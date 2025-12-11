import cv2
import numpy as np
import matplotlib.pyplot as plt

def run_dip_filters(img_path, sigma_val):
    # Loading image and converting to RGB
    image_bgr = cv2.imread(img_path)
    def to_rgb(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Box filter
    # 5x5 Normalized
    box_5_norm = cv2.boxFilter(image_bgr, -1, (5, 5), normalize=True)

    # 5x5 Un-normalized 
    box_5_unnorm_f = cv2.boxFilter(image_bgr, cv2.CV_32F, (5, 5), normalize=False)
    box_5_unnorm = np.clip(box_5_unnorm_f, 0, 255).astype(np.uint8)

    # 20x20 Normalized
    box_20_norm = cv2.boxFilter(image_bgr, -1, (20, 20), normalize=True)

    # 20x20 Un-normalized
    box_20_unnorm_f = cv2.boxFilter(image_bgr, cv2.CV_32F, (20, 20), normalize=False)
    box_20_unnorm = np.clip(box_20_unnorm_f, 0, 255).astype(np.uint8)

    # Plotting
    plt.figure(num="Box Filters Comparison", figsize=(14, 8))
    plt.suptitle("Box filters Analysis")
    
    box_images = [image_bgr, box_5_norm, box_5_unnorm, box_20_norm, box_20_unnorm]
    box_titles = ["Original Image", "5x5 Normalized Image", "5x5 Unnormalized Image", "20x20 Normalized Image", "20x20 Unnormalized Image"]
    
    # Plot layout
    layout_indices = [1, 2, 3, 5, 6] 
    
    for i in range(5):
        plt.subplot(2, 3, layout_indices[i])
        plt.imshow(to_rgb(box_images[i]))
        plt.title(box_titles[i])
        plt.axis("off")
    
    plt.tight_layout()
    
    # Gaussian filter
    # Sigma calculation
    raw_k = 2 * np.pi * sigma_val 
    k_dim = int(np.ceil(raw_k))
    if k_dim % 2 == 0: 
        k_dim += 1 
    
    print(f"\nSigma value: {sigma_val}:")
    print(f"Calculated Kernel Size: {k_dim}x{k_dim}")

    # 1D filter normalized
    gauss_1d_norm = cv2.getGaussianKernel(k_dim, sigma_val)
    gauss_norm = cv2.sepFilter2D(image_bgr, -1, gauss_1d_norm, gauss_1d_norm)
    
    # 1D filter unnormalized
    k_half = k_dim // 2
    x = np.linspace(-k_half, k_half, k_dim)
    gauss_1d_raw = np.exp(- (x**2) / (2 * sigma_val**2))

    gauss_unnorm_f = cv2.sepFilter2D(image_bgr, cv2.CV_32F, gauss_1d_raw, gauss_1d_raw)
    gauss_unnorm = np.clip(gauss_unnorm_f, 0, 255).astype(np.uint8)

    # Plotting
    plt.figure(num="Gaussian Filters Comparison", figsize=(12, 5))
    plt.suptitle(f"Gaussian Filters Analysis\n(Sigma = {sigma_val}, k = {k_dim})")

    gauss_images = [image_bgr, gauss_norm, gauss_unnorm]
    gauss_titles = ["Original Image", 
                    f"Gaussian Normalized with Seperable filter", 
                    f"Gaussian Unnormalized with Seperable filter"]

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.imshow(to_rgb(gauss_images[i]))
        plt.title(gauss_titles[i])
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# Path and input
path = r"C:\Users\USER\Desktop\College stuff\Sem 6\DIP\Filters\Torgya - Arunachal Festival.jpg"

s_val = float(input("Enter Sigma for Gaussian filter: "))
run_dip_filters(path, s_val)

