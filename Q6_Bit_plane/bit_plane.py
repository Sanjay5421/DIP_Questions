import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def process_dip_images(image_path, label):
    # Load image as grayscale
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)

    # To plot using Matplotlib
    grid_images = []
    grid_images.append((img_array, "Original Image"))

    # Extracting the bits from the image
    for i in range(8):
        bit_val = (img_array >> i) & 1
        visual_plane = bit_val * 255
        grid_images.append((visual_plane, f"Bit Plane {i}"))

    # Plotting the different bit planes using Matplotlib
    plt.figure(figsize=(12, 12))
    plt.suptitle(f"Bit Plane Decomposition: {label}", fontsize=16)

    for i in range(9):
        plt.subplot(3, 3, i + 1)
        img_data, title = grid_images[i]
        plt.imshow(img_data, cmap='gray', vmin=0, vmax=255)
        plt.title(title)
        plt.axis('off')

    # Save 3x3 grid
    save_filename = f"C:\\Users\\USER\\Desktop\\College stuff\\Sem 6\\DIP\\DIP_QUESTIONS\\Q6_Bit_plane\\{label}_Full_3x3_Grid.png"
    plt.savefig(save_filename)
    plt.close()
    print(f"Saved 3x3 Grid to: {save_filename}")

    # Union of lower three bits
    bit0 = ((img_array >> 0) & 1) * 1
    bit1 = ((img_array >> 1) & 1) * 2
    bit2 = ((img_array >> 2) & 1) * 4

    union_noise = bit0 + bit1 + bit2

    # Original image - noise
    difference_image = img_array - union_noise

    # Displaying
    plt.figure(figsize=(15, 6))
    plt.suptitle(f"Bit Plane Removal Analysis: {label}", fontsize=16)

    plt.subplot(1, 3, 1)
    plt.imshow(img_array, cmap='gray', vmin=0, vmax=255)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(union_noise, cmap='gray') 
    plt.title("Union of Noise (Bits 0–2)\n(Auto-scaled for visibility)")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(difference_image, cmap='gray', vmin=0, vmax=255)
    plt.title("Final Difference Image")
    plt.axis('off')

    plt.tight_layout()
    print(f"Generated display window for {label}.\n")


# Input
low_light_path = r"C:\Users\USER\Desktop\College stuff\Sem 6\DIP\DIP_QUESTIONS\Q6_Bit_plane\low_light_2.jpeg"
bright_light_path = r"C:\Users\USER\Desktop\College stuff\Sem 6\DIP\DIP_QUESTIONS\Q6_Bit_plane\bright_light_2.jpeg"

process_dip_images(low_light_path, "Low_Light")
process_dip_images(bright_light_path, "Bright_Light")

plt.show()
