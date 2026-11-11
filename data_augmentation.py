"""
Data Augmentation Module
========================

This module implements classical computer vision augmentation techniques 
specifically tailored for facial datasets stored in compressed NumPy format (.npz).

Technices included:
-------------------
* **Gaussian Noise**: Adds random variation to simulate sensor noise.
* **Salt and Pepper**: Simulates digital transmission errors and dead pixels.
* **Blur**: Applies a box filter to simulate out-of-focus or motion blur.

Data Integrity:
---------------
The module is designed to integrate with the outputs of 'face2.py'. It 
automatically detects and ignores files with the suffix '_audio.npz' to 
ensure that audio spectrograms are not corrupted by visual filters. All 
augmented outputs are saved with the original 'data' key to maintain 
compatibility with standard DataLoaders.
"""

import numpy as np
import cv2
import os
import glob
import argparse
from tqdm import tqdm

def add_gaussian_noise(image, snr_db=15.0):
    """
    Adiciona ruído Gaussiano baseado em um SNR fixo (em dB).
    
    :param image: Imagem original em uint8 (0-255).
    :param snr_db: Relação Sinal-Ruído desejada em decibéis. 
                   Valores típicos: 10 (muito ruído) a 30 (pouco ruído).
    """
    # 1. Evitar overflow convertendo para float antes dos cálculos
    image_float = image.astype(np.float32)
    
    # 2. Calcular a Potência do Sinal (Média dos quadrados dos pixels)
    signal_power = np.mean(image_float ** 2)
    
    # Prevenção: Se a imagem for preta (potência 0), retornamos a própria imagem
    if signal_power == 0:
        return image.copy()
        
    # 3. Converter o SNR de decibéis (dB) para escala linear
    snr_linear = 10 ** (snr_db / 10.0)
    
    # 4. Calcular a potência do ruído necessária
    noise_power = signal_power / snr_linear
    
    # 5. O desvio padrão (sigma) é a raiz quadrada da potência do ruído
    sigma = np.sqrt(noise_power)
    
    # 6. Gerar e adicionar o ruído
    gauss = np.random.normal(0, sigma, image.shape)
    noisy_image = np.clip(image_float + gauss, 0, 255).astype(np.uint8)
    
    return noisy_image

def add_salt_and_pepper_noise(images, prob=0.04):
    """
    Applies impulsive noise (Salt and Pepper) to a batch of images.

    :param images: NumPy array of shape (N, H, W, C).
    :param prob: Total probability of noise affecting a pixel.
    :return: Augmented uint8 NumPy array.
    """
    noisy_batch = []
    for img in images:
        out = np.copy(img)
        rnd = np.random.rand(img.shape[0], img.shape[1])
        out[rnd < prob/2] = 0
        out[rnd > 1 - prob/2] = 255
        noisy_batch.append(out)
    return np.array(noisy_batch, dtype=np.uint8)

def apply_blur(images, kernel_size=5):
    """
    Applies an average blur filter to a batch of images.

    :param images: NumPy array of shape (N, H, W, C).
    :param kernel_size: Dimension of the square blur kernel.
    :return: Augmented uint8 NumPy array.
    """
    return np.array([cv2.blur(img, (kernel_size, kernel_size)) for img in images], dtype=np.uint8)

# Mapping dictionary for CLI access
AUGMENTATIONS = {
    'gaussian':    lambda imgs: add_gaussian_noise(imgs),
    'salt_pepper': lambda imgs: add_salt_and_pepper_noise(imgs),
    'blur':        lambda imgs: apply_blur(imgs)
}

def process_augmentation(input_folder, output_folder, selected_augs):
    """
    Processes all face .npz files in a directory using the specified filters.

    The function iterates through the input folder, loads the 'data' key 
    from valid files, applies the requested transformations, and saves 
    each as a new compressed .npz file with a suffix indicating the effect.

    :param input_folder: Path containing the original .npz face files.
    :param output_folder: Path to save the augmented files.
    :param selected_augs: List of augmentation keys to apply (e.g., ['blur']).
    """
    files = [f for f in glob.glob(os.path.join(input_folder, "*.npz")) if "_audio.npz" not in f]
    os.makedirs(output_folder, exist_ok=True)

    for f_path in tqdm(files, desc="Augmenting Faces"):
        try:
            data = np.load(f_path)
            if 'data' not in data: continue
            
            for name in selected_augs:
                if name in AUGMENTATIONS:
                    aug_data = AUGMENTATIONS[name](data['data'])
                    new_n = os.path.basename(f_path).replace(".npz", f"_{name.upper()}.npz")
                    np.savez_compressed(os.path.join(output_folder, new_n), data=aug_data)
        except Exception as e:
            print(f"Error in {f_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Augmentation for Face NPZ datasets")
    parser.add_argument('--input', '-i', required=True, help="Input folder with _faces.npz files")
    parser.add_argument('--output', '-o', required=True, help="Output folder")
    parser.add_argument('--types', '-t', nargs='+', default=['all'], help="Augmentation types: gaussian, salt_pepper, blur")
    args = parser.parse_args()
    
    active_types = list(AUGMENTATIONS.keys()) if 'all' in args.types else args.types
    process_augmentation(args.input, args.output, active_types)
