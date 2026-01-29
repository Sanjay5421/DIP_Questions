from PIL import Image
import matplotlib.pyplot as plt

# ---------- Grayscale (Desaturation) ----------
def desaturate_gray(img):
    w, h = img.size
    out = Image.new("L", (w, h))
    px = img.load()
    opx = out.load()

    for y in range(h):
        for x in range(w):
            r, g, b = px[x, y]
            gray = (r + g + b) // 3
            opx[x, y] = gray
    return out

# ---------- Uniform Quantization ----------
def uniform_quantize(gray_img, levels):
    factor = 255 // (levels - 1)
    w, h = gray_img.size
    out = Image.new("L", (w, h))
    px = gray_img.load()
    opx = out.load()

    for y in range(h):
        for x in range(w):
            g = px[x, y]
            q = round(g / factor) * factor
            opx[x, y] = q
    return out

# ---------- Median Cut Quantization ----------
def median_cut_quantize(gray_img, colors):
    return gray_img.convert(
        "P", palette=Image.ADAPTIVE, colors=colors
    ).convert("L")

# ---------- Octree Quantization ----------
def octree_quantize(gray_img, colors):
    return gray_img.quantize(
        colors=colors, method=2
    ).convert("L")

# ---------- Main ----------
input_path = r"C:\Users\USER\Desktop\College stuff\Sem 6\DIP\DIP_Questions\Q1_grayscale\image.png"

img = Image.open(input_path).convert("RGB")
gray = desaturate_gray(img)

levels_list = [16, 8, 4, 3]

for levels in levels_list:
    uniform = uniform_quantize(gray, levels)
    median = median_cut_quantize(gray, levels)
    octree = octree_quantize(gray, levels)

    plt.figure(figsize=(8, 8))

    plt.subplot(2, 2, 1)
    plt.title("Desaturation Gray")
    plt.imshow(gray, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.title(f"Uniform ({levels})")
    plt.imshow(uniform, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.title(f"Median Cut ({levels})")
    plt.imshow(median, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.title(f"Octree ({levels})")
    plt.imshow(octree, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"C:\\Users\\USER\\Desktop\\College stuff\\Sem 6\\DIP\\DIP_Questions\\Q1_grayscale\\quant_comparison_{levels}.png")
    plt.close()
