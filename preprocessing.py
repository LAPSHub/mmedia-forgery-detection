"""
Data Augmentation Module for Image Processing and Deepfakes.
Corrected Version: Fix in Salt&Pepper Broadcasting and Execution Order.
"""

import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm

# Define the module's public interface.
__all__ = [
    'load_image_to_ndarray',
    'add_wgn',
    'add_salt_and_pepper',
    'add_blur',
    'gerar_dataset_gaussiano',
    'gerar_dataset_sal_pimenta',
    'gerar_dataset_blur'
]

# ==========================================
# 1. UTILITY FUNCTIONS (IO and Math)
# ==========================================

def load_image_to_ndarray(file_path: str) -> np.ndarray:
    """
    Loads an image from disk returning an ndarray.
    Maintains original channel depth (does not force BGR conversion).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # IMREAD_UNCHANGED correctly loads RGB, YCrCb, or Gray
    image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    
    if image is None:
        raise ValueError(f"Failed to decode image: {file_path}")
    return image


def add_wgn(image: np.ndarray, snr: float) -> np.ndarray:
    """
    Adds Additive White Gaussian Noise (WGN).
    """
    image_float = image.astype(np.float32)
    
    # Calculates signal power (image)
    sig_power = np.mean(image_float ** 2)
    
    # Calculates noise power based on desired SNR (in dB)
    a = -0.05 * snr
    noise_power = np.sqrt(sig_power) * (10 ** a)
    
    # Generates noise
    noise = noise_power * np.random.randn(*image_float.shape)
    
    noisy_image = image_float + noise
    
    # Ensures values stay between 0 and 255
    return np.clip(noisy_image, 0, 255).astype(np.uint8)


def add_salt_and_pepper(image: np.ndarray, noise_ratio: float, salt_val=255, pepper_val=0) -> np.ndarray:
    """
    Adds Salt and Pepper (Impulsive) Noise.
    CORRECTED: Uses 2D mask to apply across all channels automatically.
    """
    noisy_image = image.copy()
    h, w = noisy_image.shape[:2]
    
    # Generates random 2D matrix
    rng = np.random.rand(h, w)
    
    # Creates 2D boolean masks
    salt_mask = rng < (noise_ratio / 2)
    pepper_mask = (rng >= (noise_ratio / 2)) & (rng < noise_ratio)
    
    # We apply the 2D mask directly. 
    # NumPy understands that if the image is 3D, it should apply the value to all 3 channels.
    noisy_image[salt_mask] = salt_val
    noisy_image[pepper_mask] = pepper_val
    
    return noisy_image


def add_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Applies simple Blur.
    """
    if kernel_size <= 0:
        raise ValueError("The kernel_size must be greater than 0.")
    return cv2.blur(image, (kernel_size, kernel_size))


# ==========================================
# 2. INTERNAL PROCESSING ENGINE
# ==========================================

def _processar_diretorio(input_dir, output_dir_name, func_ruido, **kwargs):
    """
    Traverses the directory and applies noise, saving to a new folder.
    """
    if not os.path.exists(output_dir_name):
        os.makedirs(output_dir_name)
        print(f"📁 Creating directory: {output_dir_name}")
    else:
        print(f"⚠️  Directory already exists: {output_dir_name} (Files may be overwritten)")

    files_to_process = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                files_to_process.append(os.path.join(root, file))

    if not files_to_process:
        print("❌ No valid images found in input.")
        return

    print(f"🚀 Processing {len(files_to_process)} images...")
    
    successes = 0
    errors = 0
    
    for file_path in tqdm(files_to_process):
        try:
            img = load_image_to_ndarray(file_path)
            img_processed = func_ruido(img, **kwargs)
            
            # Maintains subfolder structure
            relative_path = os.path.relpath(file_path, input_dir)
            final_path = os.path.join(output_dir_name, relative_path)
            
            os.makedirs(os.path.dirname(final_path), exist_ok=True)
            cv2.imwrite(final_path, img_processed)
            successes += 1
            
        except Exception as e:
            errors += 1
            # Print error only if critical, to avoid polluting terminal
            print(f"\n❌ Error in {os.path.basename(file_path)}: {e}")

    print(f"🏁 Finished. Successes: {successes} | Failures: {errors}\n")


# ==========================================
# 3. WRAPPERS (Call Functions)
# ==========================================

def gerar_dataset_gaussiano(diretorio_entrada: str, snr: float):
    clean_path = os.path.normpath(diretorio_entrada)
    output_name = f"{os.path.basename(clean_path)}_WGN_{snr}dB"
    print(f"--- Starting WGN (SNR={snr}) ---")
    _processar_diretorio(diretorio_entrada, output_name, add_wgn, snr=snr)

def gerar_dataset_sal_pimenta(diretorio_entrada: str, noise_ratio: float):
    clean_path = os.path.normpath(diretorio_entrada)
    pct = int(noise_ratio * 100)
    output_name = f"{os.path.basename(clean_path)}_SP_{pct}pct"
    print(f"--- Starting Salt & Pepper ({pct}%) ---")
    _processar_diretorio(diretorio_entrada, output_name, add_salt_and_pepper, noise_ratio=noise_ratio)

def gerar_dataset_blur(diretorio_entrada: str, kernel_size: int):
    clean_path = os.path.normpath(diretorio_entrada)
    output_name = f"{os.path.basename(clean_path)}_Blur_{kernel_size}px"
    print(f"--- Starting Blur (Kernel={kernel_size}) ---")
    _processar_diretorio(diretorio_entrada, output_name, add_blur, kernel_size=kernel_size)


# ==========================================
# 4. CLI (Command Line Interface)
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Augmentation Tool.")
    
    parser.add_argument("-i", "--input", required=True, help="Input directory.")
    parser.add_argument("-t", "--type", choices=["wgn", "sp", "blur", "all"], nargs='+', required=True, 
                        help="Noise types (e.g., -t wgn sp blur).")
    
    # Default configurations
    parser.add_argument("--snr", type=float, default=25.0, help="SNR (WGN). Default: 25.0")
    parser.add_argument("--ratio", type=float, default=0.05, help="Ratio (SP). Default: 0.05")
    parser.add_argument("--kernel", type=int, default=5, help="Kernel (Blur). Default: 5")

    args = parser.parse_args()

    # Prepare task list
    if "all" in args.type:
        types = ["wgn", "sp", "blur"]
    else:
        types = args.type
    
    # Remove duplicates and SORT to ensure execution order: Blur > SP > WGN
    sorted_types = sorted(list(set(types)))

    print(f"🔄 Processing queue: {sorted_types}\n")

    for type_ in sorted_types:
        if type_ == "wgn":
            gerar_dataset_gaussiano(args.input, snr=args.snr)
        elif type_ == "sp":
            gerar_dataset_sal_pimenta(args.input, noise_ratio=args.ratio)
        elif type_ == "blur":
            gerar_dataset_blur(args.input, kernel_size=args.kernel)
        
    print("✅ All processing completed.")
