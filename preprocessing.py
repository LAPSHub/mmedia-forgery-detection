import argparse
import json
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Tuple

import cv2
import numpy as np
import torch
import torchaudio
from librosa.effects import split
from moviepy import VideoClip, VideoFileClip
from moviepy.video.compositing.CompositeVideoClip import concatenate_videoclips
from mtcnn import MTCNN
from numba import jit
from numpy.typing import NDArray
from PIL import Image
from scipy.io import wavfile
from tqdm import tqdm


def init_worker():
    global detector
    detector = MTCNN()


def check_out_video(
    index: int,
    video_path: Path,
    output_dir: Path,
):
    resume_file_path = output_dir.joinpath("resume.json")
    json_data = None
    with open(resume_file_path.as_posix()) as f:
        json_data = json.load(f)
    json_data["preprocessed_videos"].append(video_path.name)
    json_data["last_video_index"] = index
    with open(resume_file_path.as_posix(), "w") as f:
        json.dump(json_data, f, indent=4)


@jit(nopython=True)
def get_face(frame: NDArray, x1, x2, y1, y2) -> NDArray:
    return frame[y1:y2, x1:x2]


def crop_face(frame: NDArray, detector: MTCNN) -> NDArray | None:

    faces = detector.detect_faces(frame)
    if faces:
        result = faces[0]
        x, y, width, height = result["box"]

        pad_x = int(width * 0.25)
        pad_y = int(height * 0.25)

        h_frame, w_frame, _ = frame.shape
        startx = max(0, x - pad_x)
        starty = max(0, y - pad_y)
        endx = min(w_frame, x + width + pad_x)
        endy = min(h_frame, y + height + pad_y)
        face = get_face(frame, startx, endx, starty, endy)

        return face


def save_frame(path: str, frame: NDArray):
    Image.fromarray(frame).save(path)


def preprocess_batch_frame(
    b_frame: list,
    output_dir: Path,
    index: int,
):
    global detector

    to_rgb = lambda f: cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    for g, frame in enumerate(b_frame, index):
        cropped_frame = crop_face(to_rgb(frame), detector)
        if cropped_frame is None:
            print(f"crop_face return is None on frame {g}")
            continue
        save_frame(f"{output_dir}/frame_{g + 1:04d}.jpg", cropped_frame)


def extract_frames(
    video_path: str,
    output_dir: Path,
    workers: int,
) -> int | None:

    output_dir.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as executor:

        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            print(f"Error: Could not open video file {video_path}")
            exit()

        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = list(video.read()[1] for _ in range(total_frames))
        batch_size = total_frames // workers if total_frames > workers else 1
        batches = [
            [i, frames[i : i + batch_size]] for i in range(0, total_frames, batch_size)
        ]

        futures = [
            executor.submit(
                preprocess_batch_frame,
                batch[1],
                output_dir,
                batch[0],
            )
            for batch in batches
        ]
        wait(futures)

        extracted_frames_num = len(list(output_dir.glob("**/*.jpg")))

    return extracted_frames_num


def extract_audio_features(audio: NDArray[Any], **mfcc_kwargs) -> NDArray | None:

    audio = audio - np.mean(audio)
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
    mfccs = torchaudio.compliance.kaldi.mfcc(waveform=audio_tensor, **mfcc_kwargs)

    mean, std = torch.mean(
        mfccs,
        dim=0,
        keepdim=True,
    ), torch.std(
        mfccs,
        dim=0,
        keepdim=True,
    )
    mfccs = (mfccs - mean) / (std + 1e-9)
    to_rgb = mfccs.repeat(3, 1, 1).permute(1, 2, 0)
    return to_rgb.numpy()


def save_audio_features(
    audio_features: NDArray[Any],
    output_dir: Path,
    workers: int,
):

    output_dir.mkdir(parents=True, exist_ok=True)

    nor_audio_features = (audio_features - np.min(audio_features)) / (
        np.max(audio_features) - np.min(audio_features) + 1e-6
    )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for i, frame in enumerate(nor_audio_features):
            frame_array = frame.reshape(audio_features.shape[1], 1, 3)
            frame_array = (frame_array * 255).astype(np.uint8)
            executor.submit(
                save_frame,
                f"{output_dir}/frame_{i + 1:04d}.jpg",
                frame_array,
            )


def extract_audio(
    video: VideoClip, sr: int | None, channel: int = 0
) -> NDArray[Any] | None:

    audio_clip = video.audio

    if audio_clip is None:
        return None
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_audio:

        audio_clip.write_audiofile(tmp_audio.name, fps=sr, logger=None)
        _, audio_array = wavfile.read(tmp_audio.name)
        match channel:
            case 0:
                audio_array = audio_array.mean(axis=1)
            case -1:
                audio_array = audio_array[:, 0]

            case 1:
                audio_array = audio_array[:, 1]

    return audio_array


def cut_video(
    video: VideoClip,
    start: float | Tuple | str = 0.0,
    end: float | Tuple | str | None = None,
) -> VideoClip:
    return video.subclipped(start, end)


def remove_silency(video: VideoFileClip) -> VideoClip:

    audio = video.audio
    if audio is None:
        print("Warning: video with no audio")
        return video

    sr = audio.fps
    mono = audio.to_soundarray(fps=sr).mean(axis=1)
    non_silent_intervals = split(mono, top_db=30)
    clip_segments = []
    for start, end in non_silent_intervals:
        clip_segments.append(video.subclipped(start / sr, end / sr))
    edited_video = concatenate_videoclips(
        clip_segments,
    )
    return edited_video


def preprocess_data(
    index: int,
    data_path: Path,
    output_dir: Path,
    duration: float,
    workers: int,
) -> None:

    video_name_dir = data_path.stem
    video_out_dir = output_dir / video_name_dir / "video_frames"
    audio_spec_out_dir = output_dir / video_name_dir / "audio_spectrograms"

    # --- Configuration based on the paper ---
    MFCC_CONFIG = {
        "sample_frequency": 16000,
        "frame_length": 15.0,
        "frame_shift": 4.0,
        "num_ceps": 13,
        "use_energy": False,
        "dither": 0.0,
        "window_type": "hanning",
        "cepstral_lifter": 22,
        "high_freq": -400,
        "low_freq": 20,
        "num_mel_bins": 40,
    }

    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp_file:
        video = VideoFileClip(str(data_path))

        if duration:
            assert video.duration >= duration, "duration > video.duration"

        print(f"\nVideo: {data_path.name}")
        print(f"Video duration (Raw): {video.duration}s")

        edited_video = remove_silency(video)
        edited_video = cut_video(edited_video, end=duration)
        edited_video.write_videofile(tmp_file.name, logger=None, threads=workers)
        video.close()

        clean_video = VideoFileClip(tmp_file.name)

        print(f"Video duration (Edited): {clean_video.duration}s")

        audio = extract_audio(clean_video, MFCC_CONFIG["sample_frequency"])
        if audio is None:
            print(f"No audio found in video: {data_path}")
            return
        extracted_features = extract_audio_features(audio, **MFCC_CONFIG)

        if extracted_features is None:
            print("Extract features is None")
            return

        audio_features = extracted_features
        save_audio_features(audio_features, audio_spec_out_dir, workers)
        frames_num = extract_frames(tmp_file.name, video_out_dir, workers)
        if frames_num is None:
            print("Extracted frames number is None")
            return
        print(f"Extracted frames: {frames_num}\n")
        clean_video.close()
        check_out_video(index, data_path, output_dir)


def main(args):
    start_index = 0
    source_dir, output_dir, duration, workers, resume = (
        Path(args.source_dir),
        Path(args.output_dir),
        args.duration,
        args.workers_num,
        args.resume,
    )
    if not workers:
        workers = os.cpu_count() or 1
    if not source_dir.is_dir() or not any(source_dir.iterdir()):
        print(f"Error: Source directory not found or is empty at {source_dir}")
        return

    video_files = list(source_dir.glob("**/*.mp4")) + list(source_dir.glob("**/*.mov"))

    resume_file_path = output_dir.joinpath("resume.json")
    if resume and resume_file_path.is_file():
        print(f"Checking the last check point...\n")
        with open(resume_file_path.as_posix()) as f:
            json_file = json.load(f)
            preprocessed_videos = json_file["preprocessed_videos"]
            last_video_index = json_file["last_video_index"]
            filtered_videos = [
                video
                for video in video_files
                if video.name not in preprocessed_videos
            ]

            start_index = last_video_index + 1 if last_video_index else 0
            video_files = filtered_videos
            if len(video_files) == 0:
                print("\nPre-processing complete")
                return
    else:
        if resume:
            print(
                f"\nWarning: Resume is set to True, but resume.json file does not exists"
            )
        output_dir.mkdir(exist_ok=True)
        with open(resume_file_path.as_posix(), "w") as f:
            data = {"preprocessed_videos": [], "last_video_index": None}
            json.dump(data, f, indent=4)

    print(17 * "=")
    print("Summary")
    print(17 * "=")
    print(f"Source dir: {source_dir.absolute().as_posix()}")
    print(f"Output dir: {output_dir.absolute().as_posix()}")
    print(f"Videos found: {len(video_files)}")
    print(f"Workes used: {workers}\n")
    print(f"Pre-processing started. This may take a long time.\n")
    for i, video_path in tqdm(
        enumerate(video_files, start_index), desc="Processing videos"
    ):
        preprocess_data(i, video_path, output_dir, duration, workers)
    try:
        shutil.copy(
            source_dir.joinpath("metadata.json").as_posix(),
            output_dir.as_posix(),
        )
    except FileNotFoundError:
        print(f"Error: Source file metadata.json not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    print("\nPre-processing complete")
    print(f"\nDataset created at: {output_dir.absolute().as_posix()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-process videos for the AVoiD-DF model using MTCNN."
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        required=True,
        help="Directory containing 'real' and 'fake' subfolders with videos.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the processed dataset.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Duration of the video",
    )
    parser.add_argument(
        "--workers_num",
        type=int,
        default=None,
        help="Number of used workers on preprocessing",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the dataset preprocessing",
    )

    args = parser.parse_args()
    main(args)
