import cv2
import numpy as np

# HDR
def create_hdr(image_files, exposure_times):
    print("Loading images...")
    images = []

    for file in image_files:
        img = cv2.imread(file)
        if img is None:
            print("Error loading image:", file)
            return
        images.append(img)

    # Convert exposure times to numpy array
    times = np.array(exposure_times, dtype=np.float32)

    # Camera Response Function (Debevec method)
    print("Estimating camera response function...")
    calibrate = cv2.createCalibrateDebevec()
    response = calibrate.process(images, times)

    # Merge images to create HDR image
    print("Merging images to form HDR...")
    merge = cv2.createMergeDebevec()
    hdr_image = merge.process(images, times, response)

    # Tone Mapping (Reinhard)
    print("Applying tone mapping...")
    tonemap = cv2.createTonemapReinhard(gamma=2.2, intensity = 0, light_adapt = 0, color_adapt = 0)
    ldr_image = tonemap.process(hdr_image)

    # Convert to 8-bit image and save
    print("Saving output image...")
    ldr_image = ldr_image * 255
    ldr_image = np.clip(ldr_image, 0, 255)
    ldr_image = ldr_image.astype(np.uint8)

    cv2.imwrite(r"C:\Users\USER\Desktop\College stuff\Sem 6\DIP\DIP_Questions\Q4_HDR\output_waterfall.jpg", ldr_image)
    print("HDR image saved successfully")


# Main 
if __name__ == "__main__":

    image_list = [
        r"C:\Users\USER\Desktop\College stuff\Sem 6\DIP\DIP_QUESTIONS\Q4_HDR\Waterfall\Waterfall_1.jpg",
        r"C:\Users\USER\Desktop\College stuff\Sem 6\DIP\DIP_QUESTIONS\Q4_HDR\Waterfall\Waterfall_2.jpg",
        r"C:\Users\USER\Desktop\College stuff\Sem 6\DIP\DIP_QUESTIONS\Q4_HDR\Waterfall\Waterfall_3.jpg",
    ]

    exposure_times = [1/6, 1.3, 5]

    create_hdr(image_list, exposure_times)
