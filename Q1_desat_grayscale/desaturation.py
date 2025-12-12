from PIL import Image

def desaturation_grayscale(
    in_img = "C:\\Users\\admin\\Desktop\\Sem 6\\DIP\\Desaturation_grayscale\\pre_image.jpg",
    op_img = "C:\\Users\\admin\\Desktop\\Sem 6\\DIP\\Desaturation_grayscale\\post_image.jpg"
):
    try:
        img = Image.open(in_img).convert("RGB")
        width, height = img.size

        grayscale_img = Image.new("L", (width, height))

        for x in range(width):
            for y in range(height):
                r, g, b = img.getpixel((x, y))
                max_val = max(r, g, b)
                min_val = min(r, g, b)

                gray_value = (max_val + min_val) // 2
                grayscale_img.putpixel((x, y), gray_value)

        grayscale_img.save(op_img)
        print(f"Successfully converted '{in_img}' to grayscale and saved as '{op_img}'")

    except FileNotFoundError:
        print(f"Error: Input file not found at {in_img}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    desaturation_grayscale()
