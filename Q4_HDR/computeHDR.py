import cv2
import numpy as np

def create_hdr(image_files, exposure_times):
    # 1. Load Images
    print("Loading images...")
    images = []
    for filename in image_files:
        img = cv2.imread(filename)
        if img is None:
            print(f"Error: Could not read image {filename}")
            return
        images.append(img)

    # Convert exposure times to numpy array (float32)
    times = np.array(exposure_times, dtype=np.float32)

    # 2. Estimate Camera Response Function (CRF)
    # The image text describes the Debevec & Malik method.
    # OpenCV has this built-in.
    print("Estimating Camera Response Function (Debevec)...")
    calibrate = cv2.createCalibrateDebevec()
    response = calibrate.process(images, times)

    # 3. Recover Irradiance Map (Merge)
    # Merges images using the calculated response curve.
    print("Merging images into Irradiance Map...")
    merge = cv2.createMergeDebevec()
    hdr_image = merge.process(images, times, response)

    # Optional: Save the raw HDR data (floating point)
    # cv2.imwrite("output.hdr", hdr_image)

    # 4. Global Tone Mapping
    # The formula in your image (Ld = Lm(1+Lm/Lw^2)/(1+Lm)) is exactly the Reinhard operator.
    # OpenCV's createTonemapReinhard implements this.
    # gamma=2.2 applies the gamma correction mentioned in your original code.
    print("Applying Global Tone Mapping (Reinhard)...")
    tonemap = cv2.createTonemapReinhard(gamma=2.2, intensity=0, light_adapt=0, color_adapt=0)
    # Note: intensity=0 makes it closer to the "Global" operator described in your text.
    # You can tweak this (e.g. intensity=1.0) if the result is too flat.

    ldr_image = tonemap.process(hdr_image)

    # 5. Save Output
    # The tonemap process returns 0-1 floats. Convert to 0-255 integers.
    print("Saving output...")
    ldr_image_8bit = np.clip(ldr_image * 255, 0, 255).astype('uint8')
    cv2.imwrite("output_hdr_2.jpg", ldr_image_8bit)
    print("Success! Saved as 'output_hdr.jpg'")

# --- USER CONFIGURATION ---
if __name__ == "__main__":
    # REPLACE these with your actual filenames
    my_images = [
        r"C:\Users\USER\Desktop\College stuff\Sem 6\DIP\Q4\Building\IMG_7189.jpg",
        r"C:\Users\USER\Desktop\College stuff\Sem 6\DIP\Q4\Building\IMG_7190.jpg",
        r"C:\Users\USER\Desktop\College stuff\Sem 6\DIP\Q4\Building\IMG_7191.jpg",
        r"C:\Users\USER\Desktop\College stuff\Sem 6\DIP\Q4\Building\IMG_7192.jpg"
    ]

    # REPLACE these with the shutter speeds for each image
    my_times = [1/4, 1, 4, 15]

    create_hdr(my_images, my_times)
