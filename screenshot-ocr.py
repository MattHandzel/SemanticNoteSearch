import subprocess
import tempfile
import sys
from PIL import Image
import pytesseract


def take_screenshot(file_path):
    """Capture screenshot using grimblast on Wayland"""
    try:
        # Use grimblast to capture a region and save it to the file_path
        subprocess.run(
            ["grimblast", "--notify", "copysave", "area", file_path], check=True
        )
    except subprocess.CalledProcessError:
        print("Screenshot capture canceled or failed")
        sys.exit(1)
    except FileNotFoundError:
        print("grimblast not found. Please install grim and grimblast.")
        print("Installation instructions: https://github.com/hyprwm/contrib")
        sys.exit(1)


def ocr_from_screenshot():
    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        screenshot_path = tmpfile.name

    # Take screenshot
    print("Select area to capture (click and drag)...")
    take_screenshot(screenshot_path)

    try:
        # Open image and perform OCR
        img = Image.open(screenshot_path)
        text = pytesseract.image_to_string(img)

        # Clean up empty lines and show results
        cleaned_text = "\n".join([line for line in text.split("\n") if line.strip()])
        print("\nExtracted Text:\n")
        print(cleaned_text)

    except Image.UnidentifiedImageError:
        print("Error: Could not read screenshot")
    finally:
        # Clean up temporary file
        subprocess.run(["rm", screenshot_path])


if __name__ == "__main__":
    # Check for Tesseract
    if not pytesseract.get_tesseract_version():
        print("Tesseract OCR not found. Please install:")
        print(" - macOS: brew install tesseract")
        print(" - Linux: sudo apt install tesseract-ocr")
        print(" - Windows: https://github.com/UB-Mannheim/tesseract/wiki")
        sys.exit(1)

    ocr_from_screenshot()
