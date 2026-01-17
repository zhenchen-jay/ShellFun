import argparse
import os
import sys
from PIL import Image

# Function to parse command-line arguments
def parse_args():
    """
    Parses command-line arguments for mesh and output directories.
    
    Returns:
        argparse.Namespace: Contains parsed command-line arguments.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Convert PNG to JPG")
    known_args = sys.argv[1:]
    print (known_args)
    
    parser.add_argument('-i', '--input_png', type=str, required=True, help="Input png file (or directory containing png files)")
    parser.add_argument('-o', '--output_jpg', type=str, help="output jpg file (or directory containing jpg files)")
    
    return parser.parse_args(known_args)


def convert_png_to_jpg(png_file, jpg_file):
    """
    Converts a PNG image to JPG with a white background.
    
    Args:
        png_file (str): Path to the input PNG file.
        jpg_file (str): Path to save the output JPG file.
    """
    with Image.open(png_file) as img:
        # Create a white background image
        bg = Image.new("RGB", img.size, (255, 255, 255))
        # Paste the image onto the white background
        bg.paste(img, mask=img.split()[3])  # 3 is the alpha channel
        # Save as JPG
        bg.save(jpg_file, "JPEG")

if __name__ == "__main__":
    args = parse_args()
    if args.output_jpg is None:
        if os.path.isdir(args.input_png):
            args.output_jpg = args.input_png
        else:
            args.output_jpg = os.path.splitext(args.input_png)[0] + ".jpg"
    if os.path.isdir(args.input_png):
        for file in os.listdir(args.input_png):
            if file.endswith(".png"):
                convert_png_to_jpg(os.path.join(args.input_png, file), os.path.join(args.output_jpg, file.replace(".png", ".jpg")))
    else:
        convert_png_to_jpg(args.input_png, args.output_jpg)
