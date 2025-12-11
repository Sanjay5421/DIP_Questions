import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# --- 1. Convolve a 2D Matrix (Keeps Same Size) ---
def apply_kernel(image_channel, kernel):
    height, width = image_channel.shape
    k_height, k_width = kernel.shape
    
    # Padding so output size matches input
    pad_h = k_height // 2
    pad_w = k_width // 2
    padded = np.pad(image_channel, ((pad_h, pad_h), (pad_w, pad_w)), 
                    mode='constant', constant_values=0)
    
    result = np.zeros((height, width))
    
    # Slide the kernel over the image
    for i in range(height):
        for j in range(width):
            patch = padded[i : i + k_height, j : j + k_width]
            result[i, j] = np.sum(patch * kernel)
    
    return result

# --- 2. Apply a Filter to an RGB Image ---
def filter_rgb_image(image_arr, kernel):
    # Split into channels
    red = image_arr[:, :, 0]
    green = image_arr[:, :, 1]
    blue = image_arr[:, :, 2]
    
    # Convolve each channel
    red_filtered   = apply_kernel(red, kernel)
    green_filtered = apply_kernel(green, kernel)
    blue_filtered  = apply_kernel(blue, kernel)
    
    # Merge channels and make sure values are 0-255
    filtered_image = np.zeros_like(image_arr)
    filtered_image[:, :, 0] = np.clip(red_filtered, 0, 255).astype(np.uint8)
    filtered_image[:, :, 1] = np.clip(green_filtered, 0, 255).astype(np.uint8)
    filtered_image[:, :, 2] = np.clip(blue_filtered, 0, 255).astype(np.uint8)
    
    return filtered_image

# --- 3. Load the Image ---
img_path = r"C:\Users\USER\Desktop\College stuff\Sem 6\DIP\Box_filter\Torgya - Arunachal Festival.jpg"

try:
    image = Image.open(img_path).convert("RGB")
    img_array = np.array(image)
except Exception as e:
    print(f"Could not load image. Using a random demo image instead.\n{e}")
    img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

# --- 4. Define Blurring Kernels ---
# Small 5x5 kernel
kernel_5_normalized   = np.ones((5, 5)) / 25.0  # Slight blur
kernel_5_unormalized  = np.ones((5, 5))         # Overexposed / white effect

# Large 20x20 kernel
kernel_20_normalized  = np.ones((20, 20)) / 400.0  # Strong blur
kernel_20_unormalized = np.ones((20, 20))          # Very bright / white

# --- 5. Apply Filters ---
print("Applying 5x5 normalized kernel...")
blur_5_normalized = filter_rgb_image(img_array, kernel_5_normalized)

print("Applying 5x5 un-normalized kernel...")
blur_5_unormalized = filter_rgb_image(img_array, kernel_5_unormalized)

print("Applying 20x20 normalized kernel...")
blur_20_normalized = filter_rgb_image(img_array, kernel_20_normalized)

print("Applying 20x20 un-normalized kernel...")
blur_20_unormalized = filter_rgb_image(img_array, kernel_20_unormalized)

# --- 6. Show Original and Filtered Images ---
fig, axes = plt.subplots(1, 5, figsize=(20, 5))

axes[0].imshow(img_array)
axes[0].set_title("Original Image")
axes[0].axis('off')

axes[1].imshow(blur_5_normalized)
axes[1].set_title("5x5 Normalized\n(Slight Blur)")
axes[1].axis('off')

axes[2].imshow(blur_5_unormalized)
axes[2].set_title("5x5 Un-normalized\n(Overexposed)")
axes[2].axis('off')

axes[3].imshow(blur_20_normalized)
axes[3].set_title("20x20 Normalized\n(Heavy Blur)")
axes[3].axis('off')

axes[4].imshow(blur_20_unormalized)
axes[4].set_title("20x20 Un-normalized\n(Overexposed)")
axes[4].axis('off')

plt.tight_layout()
plt.show()
