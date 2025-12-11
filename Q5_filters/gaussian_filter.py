import cv2
import numpy as np
import matplotlib.pyplot as plt

def compare_gaussian_normalization_cv2(image_path, sigma):
    # --- 1. Load Image ---
    image_bgr = cv2.imread(image_path)
        
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    print(f"Original Image Shape: {image_rgb.shape}")

    # --- 2. Calculate Kernel Size Manually ---
    # User Request: Calculate k size using 2 * pi * sigma
    # 2 * pi is approx 6.28, so this is similar to the standard "6-sigma" rule 
    # which ensures the kernel covers ~99% of the Gaussian curve.
    raw_k = 2 * np.pi * sigma
    k_dim = int(np.ceil(raw_k)) # Round up to nearest integer

    # DIP Rule: Kernel size must be ODD. If even, add 1.
    if k_dim % 2 == 0:
        k_dim += 1
    

    print(f"Sigma: {sigma}")
    print(f"Calculated Kernel Size (2 * pi * sigma): {raw_k:.2f} -> Adjusted to Odd Integer: {k_dim}x{k_dim}")

    # --- 3. Calculate Un-normalization Factor ---
    # The sum of a 2D Gaussian kernel with peak 1.0 is approx 2 * pi * sigma^2
    unnorm_factor = 2 * np.pi * (sigma ** 2)
    print(f"Un-normalization factor (Brightness Multiplier): {unnorm_factor:.2f}")

    # --- 4. Apply Filters ---
    
    # A. Normalized (Standard OpenCV)
    # We now pass our calculated k_dim instead of (0,0)
    norm_img = cv2.GaussianBlur(image_rgb, ksize=(k_dim, k_dim), sigmaX=sigma, sigmaY=sigma)

    # B. Unnormalized (Reconstructed)
    # Multiply the normalized result by the factor to simulate the unnormalized summation
    unnorm_img_float = norm_img.astype(np.float32) * unnorm_factor
    
    print(f"Max value in Unnormalized: {np.max(unnorm_img_float):.2f}")
    
    # Clip to 0-255 for display
    unnorm_img_display = np.clip(unnorm_img_float, 0, 255).astype(np.uint8)

    # --- 5. Visualization ---
    plt.figure(figsize=(15, 6))
    
    titles = ['Original', f'Normalized\n(k={k_dim}x{k_dim})', f'Unnormalized\n(Factor: {unnorm_factor:.1f}x)']
    images = [image_rgb, norm_img, unnorm_img_display]
    
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(titles[i])
        plt.imshow(images[i])
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()

# --- Run ---
sigma_input = float(input("Enter Sigma value (Try 1.0 vs 5.0): "))
compare_gaussian_normalization_cv2(r"Filters/Torgya - Arunachal Festival.jpg", sigma_input)
