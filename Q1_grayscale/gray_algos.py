from PIL import Image
import matplotlib.pyplot as plt

# Grayscale (Desaturation)
def desaturate_gray(img):
    width, height = img.size
    gray_img = Image.new("L", (width, height))

    pixels = img.load()
    gray_pixels = gray_img.load()

    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            gray = (r + g + b) // 3
            gray_pixels[x, y] = gray

    return gray_img


# Uniform Quantization
def uniform_quantize(gray_img, levels):
    width, height = gray_img.size
    quant_img = Image.new("L", (width, height))

    pixels = gray_img.load()
    quant_pixels = quant_img.load()

    step = 255 // (levels - 1)

    for y in range(height):
        for x in range(width):
            value = pixels[x, y]
            quant_pixels[x, y] = round(value / step) * step

    return quant_img


# Median Cut Quantization 
def median_cut_quantize(gray_img, levels):
    return gray_img.convert("P",
                            palette=Image.ADAPTIVE,
                            colors=levels).convert("L")


# Octree Quantization
def octree_quantize(gray_img, levels):
    return gray_img.quantize(colors=levels,
                             method=2).convert("L")


# Main Program
input_path = r"C:\Users\USER\Desktop\College stuff\Sem 6\DIP\DIP_Questions\Q1_grayscale\image.png"
output_folder = r"C:\Users\USER\Desktop\College stuff\Sem 6\DIP\DIP_Questions\Q1_grayscale"

image = Image.open(input_path).convert("RGB")
gray_image = desaturate_gray(image)

levels_list = [16, 8, 4, 3]

for levels in levels_list:

    uniform_img = uniform_quantize(gray_image, levels)
    median_img = median_cut_quantize(gray_image, levels)
    octree_img = octree_quantize(gray_image, levels)

    plt.figure(figsize=(8, 8))

    plt.subplot(2, 2, 1)
    plt.title("Desaturation Gray")
    plt.imshow(gray_image, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.title("Uniform Quantization")
    plt.imshow(uniform_img, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.title("Median Cut")
    plt.imshow(median_img, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.title("Octree")
    plt.imshow(octree_img, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_folder + "\\quant_comparison_" + str(levels) + ".png")
    plt.close()
