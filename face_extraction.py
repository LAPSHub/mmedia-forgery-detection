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
from PIL import Image

# ==========================================
# 1. GLOBAL CONFIGURATIONS
# ==========================================

# MFCC Configuration (Mel-Frequency Cepstral Coefficients)
# OBJECTIVE: Define how the computer will "hear" and transform sound into an image.
MFCC_CONFIG = {
    "sample_frequency": 16000, # Sample rate of 16kHz (Standard for human voice and telephony).
    "frame_length": 15.0,      # Analysis window length (15ms): short enough for voice to be stable.
    "frame_shift": 4.0,        # Window shift (4ms): how much the window "moves" to the right for each calculation.
    "num_ceps": 13,            # Number of coefficients that summarize the vocal tract shape (the "identity" of the sound).
    "use_energy": False,       # If True, adds the total frame energy as extra data.
    "dither": 0.0,             # Random noise to avoid errors in absolute silence (not used here).
    "window_type": "hanning",  # Smoothing of audio cut edges to avoid mathematical noise (spectral leakage).
    "cepstral_lifter": 22,     # Constant to rescale coefficients and facilitate AI learning.
    "high_freq": -400,         # High frequency filter.
    "low_freq": 20,            # Low frequency filter (removes deep rumbles).
    "num_mel_bins": 40,        # Vertical resolution of the generated spectrogram image.
}

# Map to convert input strings (e.g., 'rgb') to OpenCV numeric constants
COLOR_MAP = {
    'bgr':   None,                
    'rgb':   cv2.COLOR_BGR2RGB,   
    'gray':  cv2.COLOR_BGR2GRAY,  
    'hsv':   cv2.COLOR_BGR2HSV,   
    'ycrcb': cv2.COLOR_BGR2YCrCb, 
    'lab':   cv2.COLOR_BGR2LAB,
    'yuv':   cv2.COLOR_BGR2YUV,
    'luv':   cv2.COLOR_BGR2LUV
}

# Global variable to hold the MTCNN neural network within each process
process_detector = None

def init_worker():
    """
    Initializes the environment for each worker in parallel processing.
    
    WHY IS THIS NECESSARY?
    Neural networks (like MTCNN) and libraries like TensorFlow/PyTorch often crash
    if you try to share the same loaded network across multiple CPU processes.
    
    This function ensures that each CPU core (worker) loads its own independent
    copy of MTCNN and configures its threads to avoid competition.
    """
    global process_detector
    # Silence unnecessary TensorFlow logs
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    # Force PyTorch to use only 1 thread per process (since we will have many processes)
    torch.set_num_threads(1)
    # Load the face detection network into the memory of this specific process
    process_detector = MTCNN()

# ==========================================
# 2. AUDIO HELPERS
# ==========================================

def get_audio_waveform(video_path, target_sr=16000):
    """
    Extracts the complete audio track from a video file.
    
    WHAT IT DOES:
    1. Uses MoviePy library to separate audio from video.
    2. Saves temporarily to disk and re-reads with Scipy (faster/safer).
    3. Converts Stereo (2 channels) to Mono (1 channel) by averaging.
    
    IMPORTANT:
    This function returns a 'Time Ruler'. It DOES NOT remove silence.
    If the video is 10 minutes long, it returns 10 minutes of audio data.
    This is crucial to maintain exact synchronization between the video frame
    and the audio moment.
    
    Args:
        video_path (str): Path to the video file.
        target_sr (int): Target sample rate (default 16kHz).
        
    Returns:
        tuple: (numpy array with audio data, sample rate)
    """
    try:
        # Create a temporary file to dump the extracted audio
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_audio:
            video_clip = VideoFileClip(video_path)
            
            # If video has no audio, return null
            if video_clip.audio is None:
                video_clip.close()
                return None, 0
            
            # Write audio to temp file forcing 16kHz
            video_clip.audio.write_audiofile(tmp_audio.name, fps=target_sr, logger=None)
            video_clip.close()
            
            # Read the generated WAV file into a numpy array
            sr, audio_data = wavfile.read(tmp_audio.name)
            
            # If more than one channel (Stereo), average them to get Mono
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
                
            return audio_data, sr
    except Exception:
        return None, 0

def generate_mfcc_image_from_snippet(audio_snippet):
    """
    Converts a small audio snippet (Raw Waveform) into a Spectral Image (MFCC).
    
    THE PROCESS:
    1. Receives a vector of numbers (sound amplitude over time).
    2. Applies mathematical transformations (Fourier -> Mel Scale -> DCT) to get MFCC.
    3. Normalizes data (subtract mean, divide by std dev).
    4. Transforms the 2D matrix into a 'fake' RGB image (3 equal channels)
       so that Computer Vision Neural Networks (like ResNet) can process it.
       
    Args:
        audio_snippet (np.array): Audio snippet (e.g., 3200 samples = 200ms).
        
    Returns:
        np.array (uint8): An image ready to save (0-255).
    """
    # Protection: If snippet is too small (e.g., end of video), ignore.
    if len(audio_snippet) < 100: 
        return None

    # Mean normalization (remove DC offset from electrical signal)
    audio_snippet = audio_snippet - np.mean(audio_snippet)
    # Convert to PyTorch Tensor
    audio_tensor = torch.from_numpy(audio_snippet).float().unsqueeze(0)
    
    # DSP Magic: Generate MFCC coefficients using Kaldi config
    mfccs = torchaudio.compliance.kaldi.mfcc(waveform=audio_tensor, **MFCC_CONFIG)
    
    # Statistical Normalization (Standardization) to help AI
    mean = torch.mean(mfccs, dim=0, keepdim=True)
    std = torch.std(mfccs, dim=0, keepdim=True)
    mfccs = (mfccs - mean) / (std + 1e-9)
    
    # Trick: MFCC is a 1-channel matrix (Grayscale). 
    # Here we repeat it 3 times to simulate an RGB image (3 channels).
    to_rgb = mfccs.repeat(3, 1, 1).permute(1, 2, 0)
    
    # Normalize values to pixel interval [0, 1] then [0, 255]
    mfcc_img = to_rgb.numpy()
    norm_img = (mfcc_img - np.min(mfcc_img)) / (np.max(mfcc_img) - np.min(mfcc_img) + 1e-6)
    
    # Convert to 8-bit integers (standard image format)
    img_uint8 = (norm_img * 255).astype(np.uint8)
    
    return img_uint8

# ==========================================
# 3. SYNCHRONIZED PROCESSING CORE
# ==========================================

def process_synchronized_video(video_path, output_root, num_frames_target, img_size, color_space, margin_scale):
    """
    Main function that processes a SINGLE video to extract pairs (Face, Voice).
    
    SYNCHRONIZATION LOGIC:
    1. Loads video and the complete audio 'ruler'.
    2. Iterates through video frames.
    3. When a face is found in frame N:
       a. Calculate exact time of this frame: T = N / FPS.
       b. Go to the audio ruler and cut a 200ms window around T.
       c. Save the face image and audio image (MFCC) with the SAME filename.
       
    Args:
        video_path (str): Path to the video file.
        output_root (str): Root folder to save data.
        num_frames_target (int/str): How many frames to extract ('full' or number).
        img_size (int): Final face image size (e.g., 224px).
        color_space (str): Color space (RGB, Gray, etc).
        margin_scale (float): How much margin to leave around the face (1.2 = +20%).
    """
    global process_detector
    
    # 1. Configure Output Folders (One folder per processed video)
    video_name = os.path.basename(video_path).rsplit('.', 1)[0]
    dir_video_out = os.path.join(output_root, video_name, "video_frames")
    dir_audio_out = os.path.join(output_root, video_name, "audio_spectrograms")
    
    os.makedirs(dir_video_out, exist_ok=True)
    os.makedirs(dir_audio_out, exist_ok=True)
    
    # 2. Load COMPLETE Audio into memory (Absolute Time Reference)
    full_audio_data, sample_rate = get_audio_waveform(video_path, target_sr=16000)
    has_audio = full_audio_data is not None
    total_samples = len(full_audio_data) if has_audio else 0

    # Audio Window Definition (200ms = 0.2s)
    # This defines how much voice context we grab for each mouth "photo".
    WINDOW_SECONDS = 0.2 
    WINDOW_SAMPLES = int(WINDOW_SECONDS * 16000) # Ex: 0.2 * 16000 = 3200 samples
    HALF_WINDOW = WINDOW_SAMPLES // 2            # Half backwards, half forwards from frame
    
    # 3. Initialize Video Reading
    cap = cv2.VideoCapture(video_path)
    total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Safety check for corrupted videos
    if total_frames_video == 0 or fps == 0:
        return 0, 0

    # Sampling Logic: Decide WHICH frames we will process
    extract_all = (num_frames_target == 'full')
    indices_set = set()
    if not extract_all:
        # If not 'full', choose equally spaced frames (linspace)
        target_count = int(num_frames_target)
        if total_frames_video <= target_count:
            indices_to_grab = np.arange(total_frames_video)
        else:
            indices_to_grab = np.linspace(0, total_frames_video - 1, target_count, dtype=int)
        indices_set = set(indices_to_grab)
    
    frame_idx = 0
    saved_count = 0
    conversion_code = COLOR_MAP.get(color_space.lower())

    # --- MAIN LOOP (FRAME BY FRAME) ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break # End of video

        # Check if this is a frame we want to process
        if extract_all or (frame_idx in indices_set):
            try:
                # A. VIDEO PROCESSING (Face)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Detect faces using MTCNN
                detections = process_detector.detect_faces(rgb_frame)

                if detections:
                    # Strategy: Always take the LARGEST face (assuming it's the protagonist)
                    det = max(detections, key=lambda x: x['box'][2] * x['box'][3])
                    x, y, w, h = det['box']
                    
                    # Calculate the Crop adding safety margin
                    center_x, center_y = x + w / 2, y + h / 2
                    max_dim = max(w, h) * margin_scale
                    x1 = int(max(0, center_x - max_dim / 2))
                    y1 = int(max(0, center_y - max_dim / 2))
                    x2 = int(min(frame.shape[1], center_x + max_dim / 2))
                    y2 = int(min(frame.shape[0], center_y + max_dim / 2))
                    
                    face_crop = frame[y1:y2, x1:x2]

                    # Validate if crop is valid and has minimum size
                    if face_crop.size != 0 and face_crop.shape[0] > 20:
                        # Resize to standard neural network size (e.g., 224x224)
                        interp = cv2.INTER_AREA if face_crop.shape[0] > img_size else cv2.INTER_CUBIC
                        face_resized = cv2.resize(face_crop, (img_size, img_size), interpolation=interp)
                        
                        # Define filename (Unique ID to pair with audio)
                        filename = f"frame_{saved_count:04d}.jpg"
                        path_face = os.path.join(dir_video_out, filename)

                        # Save face image applying color conversion if necessary
                        if color_space.lower() == 'rgb':
                            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                            Image.fromarray(face_rgb).save(path_face, quality=95)
                        elif conversion_code is not None and color_space.lower() != 'bgr':
                            final_face = cv2.cvtColor(face_resized, conversion_code)
                            cv2.imwrite(path_face, final_face, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                        else:
                            cv2.imwrite(path_face, face_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

                        # B. AUDIO PROCESSING (Synchronized)
                        if has_audio:
                            # 1. Calculate exact frame timestamp in seconds
                            timestamp_seconds = frame_idx / fps
                            
                            # 2. Convert seconds to sample index in audio matrix
                            center_sample_idx = int(timestamp_seconds * sample_rate)
                            
                            # 3. Define crop window (Start and End)
                            start_idx = center_sample_idx - HALF_WINDOW
                            end_idx = center_sample_idx + HALF_WINDOW
                            
                            # 4. Edge handling (beginning or end of file)
                            if start_idx < 0:
                                audio_chunk = full_audio_data[0:end_idx]
                            elif end_idx > total_samples:
                                audio_chunk = full_audio_data[start_idx:total_samples]
                            else:
                                # Ideal case: clean cut in the middle of ruler
                                audio_chunk = full_audio_data[start_idx:end_idx]
                            
                            # 5. Generate MFCC image (Spectrogram) from this snippet
                            mfcc_img = generate_mfcc_image_from_snippet(audio_chunk)
                            
                            if mfcc_img is not None:
                                path_audio = os.path.join(dir_audio_out, filename)
                                # Save with the SAME name as video frame
                                Image.fromarray(mfcc_img).save(path_audio)

                        # Increment saved pairs counter
                        saved_count += 1
                        
            except Exception:
                # If error on specific frame, skip and continue video
                pass

        frame_idx += 1
        # If we passed the last desired frame, stop reading video
        if not extract_all and indices_set and frame_idx > max(indices_set):
            break

    cap.release()
    return saved_count, saved_count if has_audio else 0

# ==========================================
# 4. MAIN (COMMAND LINE INTERFACE)
# ==========================================

def main():
    """
    Script entry point. Manages arguments and triggers parallel processing.
    """
    parser = argparse.ArgumentParser(description="Synchronized AV Extraction Pipeline (Context-Aware)")
    
    # Definition of arguments passed by user in terminal
    parser.add_argument('--input', '-i', type=str, required=True, help="Folder containing videos")
    parser.add_argument('--output', '-o', type=str, required=True, help="Folder to save dataset")
    parser.add_argument('--frames', '-f', type=str, default='full', help="How many frames per video? (or 'full')")
    parser.add_argument('--size', '-s', type=int, default=224, help="Face image size (px)")
    parser.add_argument('--color', '-c', type=str, default='bgr', help="Color space (rgb, gray, hsv...)")
    parser.add_argument('--workers', '-w', type=int, default=None, help="Number of CPU cores to use")
    
    args = parser.parse_args()

    # Basic input validations
    if args.color.lower() not in COLOR_MAP:
        print(f"Error: Invalid color space.")
        return

    # Configuration of how many frames to grab
    if args.frames.lower() == 'full':
        target_frames = 'full'
    else:
        target_frames = int(args.frames)

    # Check if input directory exists
    if not os.path.exists(args.input):
        print(f"Error: Input directory not found.")
        return
    
    # Search for videos with common extensions
    extensions = ('*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm')
    video_files = []
    for ext in extensions:
        video_files.extend(glob.glob(os.path.join(args.input, ext)))
    
    if not video_files:
        print(f" Zero videos found.")
        return

    # Define how many parallel processes to use (default = all PC cores)
    num_workers = args.workers if args.workers else os.cpu_count()

    print(f"{'='*60}")
    print(f"🚀 SYNCED AV PIPELINE (Strategy B: Context-Aware) ")
    print(f"{'='*60}")
    # ... info prints ...

    # Create a "frozen" version of the processing function with fixed arguments
    func_fixa = partial(
        process_synchronized_video, 
        output_root=args.output, 
        num_frames_target=target_frames, 
        img_size=args.size, 
        color_space=args.color,
        margin_scale=1.2
    )

    total_faces = 0
    total_audio = 0
    
    # Start ProcessPoolExecutor (Process Manager)
    with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker) as executor:
        # Submit all tasks (videos) to execution queue
        futures = [executor.submit(func_fixa, video_path) for video_path in video_files]
        
        # Progress bar to track completion
        for future in tqdm(as_completed(futures), total=len(video_files), desc="Processing"):
            try:
                f_count, a_count = future.result()
                total_faces += f_count
                total_audio += a_count
            except Exception as e:
                print(f"Error processing video: {e}")

    print(f"\n✅ Sync Processing Complete!")
    print(f"Total Paired Samples: {total_faces}")
    print(f"Dataset saved at: {args.output}")

if __name__ == "__main__":
    main()
