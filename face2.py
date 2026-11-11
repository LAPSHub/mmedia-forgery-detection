"""
Multimodal Extraction Pipeline
==============================

This module provides a robust pipeline for synchronized extraction of facial 
regions and audio features (MFCC) from video files. It is designed to scale 
using multiprocessing and handles different operational modes: Video-only, 
Audio-only, or Synchronized Multimodal extraction.

Key Components:
---------------
* **Face Detection**: Uses MTCNN to locate faces, selecting the largest face 
  per frame with optional color space conversion (RGB, Gray, HSV, etc.).
* **Audio Processing**: Extracts audio using MoviePy and generates 
  spectrogram-like MFCC features using torchaudio's Kaldi integration.
* **Sync Logic**: When both modes are active, it ensures each saved face 
  has a corresponding audio snippet centered on the frame's timestamp.

Operational Workflow:
---------------------
1. Scans input directory for video files.
2. Initializes a pool of workers; MTCNN is only loaded if video is enabled.
3. Processes videos in parallel, saving outputs as compressed .npz files.
"""

import argparse
import os
import glob
import tempfile
import cv2
import numpy as np
import torch
import torchaudio
from mtcnn import MTCNN
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from moviepy.editor import VideoFileClip
from scipy.io import wavfile

# Global configuration constants
DTYPE_MAP = {'uint8': np.uint8,
             'float16': np.float16, 
             'float32': np.float32, 
             'float64': np.float64}

COLOR_MAP = {'bgr': None, 
             'rgb': cv2.COLOR_BGR2RGB, 
             'gray': cv2.COLOR_BGR2GRAY, 
             'hsv': cv2.COLOR_BGR2HSV, 
             'ycrcb': cv2.COLOR_BGR2YCrCb, 
             'lab': cv2.COLOR_BGR2LAB}

MFCC_CONFIG = {"sample_frequency": 16000, 
               "frame_length": 15.0, 
               "frame_shift": 4.0, 
               "num_ceps": 13,
               "use_energy": False, 
               "window_type": "hanning", 
               "num_mel_bins": 40}

process_detector = None

def init_worker(video_enabled):
    """
    Initializes a worker process for parallel execution.

    :param video_enabled: Boolean flag. If True, loads the MTCNN detector into global memory.
    """
    global process_detector
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    if video_enabled:
        torch.set_num_threads(1)
        process_detector = MTCNN()

def get_audio_waveform(video_path, target_sr=16000):
    """
    Extracts the raw audio signal from a video file.

    :param video_path: Path to the source video file.
    :param target_sr: Target sample rate for the output audio.
    :return: A tuple of (audio_array, sample_rate). Returns (None, 0) on failure.
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_audio:

            video_clip = VideoFileClip(video_path)

            if video_clip.audio is None:

                video_clip.close()

                return None, 0
            
            video_clip.audio.write_audiofile(tmp_audio.name, fps=target_sr, logger=None)
            video_clip.close()
            sr, audio_data = wavfile.read(tmp_audio.name)

            if len(audio_data.shape) > 1: audio_data = audio_data.mean(axis=1)
            return audio_data, sr
    except Exception:
        return None, 0

def generate_mfcc(audio_array, target_dtype=np.uint8):
    """
    Transforms a raw audio waveform into a standardized MFCC spectrogram.

    The process includes signal normalization, Kaldi-based MFCC extraction, 
    statistical standardization (mean/std), and channel replication to create 
    a 3-channel image-like tensor.

    :param audio_array: 1D NumPy array of the audio signal.
    :param target_dtype: Desired NumPy data type for the output.
    :return: A 3D NumPy array (H, W, 3) representing the MFCC spectrogram.
    """
    if len(audio_array) < 100: return None
    audio_array = audio_array - np.mean(audio_array)
    audio_tensor = torch.from_numpy(audio_array).float().unsqueeze(0)

    mfccs = torchaudio.compliance.kaldi.mfcc(waveform=audio_tensor, **MFCC_CONFIG)
    mfccs = (mfccs - torch.mean(mfccs, dim=0)) / (torch.std(mfccs, dim=0) + 1e-9)
    mfcc_numpy = mfccs.repeat(3, 1, 1).permute(1, 2, 0).numpy()

    norm = (mfcc_numpy - np.min(mfcc_numpy)) / (np.max(mfcc_numpy) - np.min(mfcc_numpy) + 1e-6)

    return (norm * 255).astype(np.uint8) if target_dtype == np.uint8 else norm.astype(target_dtype)

def process_video_wrapper(video_path, output_root, flags, params):
    """
    Orchestrates the extraction process for a single video.

    Depending on the flags, it extracts either the full audio MFCC, 
    specific video frames (face crops), or a synchronized combination 
    where each face is paired with a corresponding audio window.

    :param video_path: Path to the video file to be processed.
    :param output_root: Directory where the .npz files will be stored.
    :param flags: Dictionary containing 'video' and 'audio' boolean activation flags.
    :param params: Dictionary containing 'frames', 'size', 'color', and 'audio_dtype'.
    :return: A tuple containing (number_of_faces_saved, number_of_audios_saved).
    """
    global process_detector
    v_name = os.path.basename(video_path).rsplit('.', 1)[0]

    p_faces, p_audio = os.path.join(output_root, f"{v_name}_faces.npz"), os.path.join(output_root, f"{v_name}_audio.npz")

    a_dtype = DTYPE_MAP.get(params['audio_dtype'], np.uint8)

    if flags['audio'] and not flags['video']:
        wav, sr = get_audio_waveform(video_path)
        if wav is not None:
            mfcc = generate_mfcc(wav, target_dtype=a_dtype)
            if mfcc is not None: 
                np.savez_compressed(p_audio, data=mfcc)
                return 0, 1
        return 0, 0

    wav_data, sr = get_audio_waveform(video_path) if flags['audio'] else (None, 0)
    cap = cv2.VideoCapture(video_path)
    total, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    if total == 0 or fps == 0: return 0, 0

    indices = set()
    if params['frames'] != 'full':
        count = int(params['frames'])
        indices = set(np.linspace(0, total - 1, count, dtype=int)) if total > count else set(range(total))
    
    b_faces, b_audios, f_idx = [], [], 0
    conv = COLOR_MAP.get(params['color'].lower())

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break 
        if params['frames'] == 'full' or (f_idx in indices):
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                dets = process_detector.detect_faces(rgb)
                if dets:
                    d = max(dets, key=lambda x: x['box'][2] * x['box'][3])
                    x, y, w, h = d['box']
                    cx, cy, m_dim = x + w/2, y + h/2, max(w, h) * 1.2
                    x1, y1 = int(max(0, cx - m_dim/2)), int(max(0, cy - m_dim/2))
                    x2, y2 = int(min(frame.shape[1], cx + m_dim/2)), int(min(frame.shape[0], cy + m_dim/2))
                    crop = frame[y1:y2, x1:x2]
                    if crop.size != 0 and crop.shape[0] > 20:
                        res = cv2.resize(crop, (params['size'], params['size']))
                        final_v = cv2.cvtColor(res, conv) if conv else res
                        
                        a_img = None
                        if flags['audio'] and wav_data is not None:
                            win = int(0.2 * 16000)
                            c_ptr = int((f_idx / fps) * sr)
                            chunk = wav_data[max(0, c_ptr - win//2):min(len(wav_data), c_ptr + win//2)]
                            if len(chunk) < win: chunk = np.pad(chunk, (0, win - len(chunk)))
                            a_img = generate_mfcc(chunk, target_dtype=a_dtype)

                        if flags['audio'] and flags['video']:
                            if a_img is not None: b_faces.append(final_v); b_audios.append(a_img)
                        elif flags['video']: b_faces.append(final_v)
            except: pass
        f_idx += 1
    cap.release()
    
    if b_faces: np.savez_compressed(p_faces, data=np.array(b_faces, dtype=np.uint8))
    if b_audios: np.savez_compressed(p_audio, data=np.array(b_audios, dtype=a_dtype))
    return len(b_faces), len(b_audios)

def main():
    parser = argparse.ArgumentParser(description="Multimodal Pipeline")
    parser.add_argument('--input', '-i', required=True, help="Input folder with videos")
    parser.add_argument('--output', '-o', required=True, help="Output folder for NPZ files")
    parser.add_argument('--video_on', action='store_true', help="Enable face extraction")
    parser.add_argument('--audio_on', action='store_true', help="Enable audio extraction")
    parser.add_argument('--frames', '-f', default='full', help="'full' or specific frame count")
    parser.add_argument('--size', '-s', type=int, default=224, help="Resize dimension")
    parser.add_argument('--color', '-c', default='bgr', help="Color space (bgr, rgb, gray, etc.)")
    parser.add_argument('--audio_dtype', default='uint8', help="Storage precision for audio")
    args = parser.parse_args()
    
    if not args.video_on and not args.audio_on: return
    
    v_exts = ('*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm')
    v_files = []
    for ext in v_exts: v_files.extend(glob.glob(os.path.join(args.input, ext)))
    
    p_func = partial(process_video_wrapper, output_root=args.output, 
                     flags={'video': args.video_on, 'audio': args.audio_on},
                     params={'frames': args.frames, 'size': args.size, 'color': args.color, 'audio_dtype': args.audio_dtype})

    with ProcessPoolExecutor(initializer=init_worker, initargs=(args.video_on,)) as ex:
        list(tqdm(as_completed([ex.submit(p_func, v) for v in v_files]), total=len(v_files), desc="Processing"))

if __name__ == "__main__":
    main()
