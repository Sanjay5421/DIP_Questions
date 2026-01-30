import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def compare_sampling(path):
    # Loading image
    img = np.array(Image.open(path).convert("L"))
    h, w = img.shape

    # Computing FFT and shifting zero freq center
    f_shifted = np.fft.fftshift(np.fft.fft2(img))
    cy, cx = h // 2, w // 2  # center coordinates

    ratios = [2, 4, 8, 16] 
    plt.figure(figsize=(10, 12))

    for i, r in enumerate(ratios):
        # Spatial sampling
        spatial_img = img[::r, ::r]  

        # Frequency Sampling (Using low pass filtering concept via masking)
        mask = np.zeros_like(img)
        rh, rw = h // (2*r), w // (2*r)  
        mask[cy - rh : cy + rh, cx - rw : cx + rw] = 1  
        freq_img = np.fft.ifft2(np.fft.ifftshift(f_shifted * mask))
        freq_img = np.abs(freq_img) 

        # Plotting
        plt.subplot(4, 2, 2*i + 1)
        plt.imshow(spatial_img, cmap='gray')
        plt.title(f"Spatial 1/{r} Resolution")
        plt.axis('off')

        plt.subplot(4, 2, 2*i + 2)
        plt.imshow(freq_img, cmap='gray')
        plt.title(f"Frequency 1/{r} Bandwidth")
        plt.axis('off')

    plt.savefig(r"C:\Users\USER\Desktop\College stuff\Sem 6\DIP\DIP_Questions\Q3_sampling\Output.jpg")
    plt.show()
    

# Run
compare_sampling(r"C:\Users\USER\Desktop\College stuff\Sem 6\DIP\DIP_Questions\Q3_sampling\image.jpg")
