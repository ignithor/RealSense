import cv2
import numpy as np
import sys


def is_image_dark(image: np.ndarray, threshold: int = 15) -> bool:
    """
    Determines if an image is dark by calculating the mean of its
    Value (V) channel in the HSV color space.

    Args:
        image: The input image (as a NumPy array in BGR format).
        threshold: The brightness value (0-255) to use as a cutoff.
                   Images with a mean value < threshold will be
                   considered dark.

    Returns:
        True if the image is dark, False if it is bright.
    """

    # 1. Check if the image is valid
    if image is None or image.size == 0:
        print("Error: Invalid image provided.")
        return False

    # 2. Convert the image from BGR to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 3. Extract the V (Value) channel
    # hsv_image[:, :, 2] is the V channel
    v_channel = hsv_image[:, :, 2]

    # 4. Calculate the mean (average) of the V channel
    mean_value = np.mean(v_channel)

    print(f"Mean Brightness (Value): {mean_value:.2f}")

    # 5. Compare against the threshold
    if mean_value < threshold:
        return True  # The image is dark
    else:
        return False  # The image is bright


# --- Example of how to use the function ---
if __name__ == "__main__":
    # Check if a file path was given
    if len(sys.argv) < 2:
        print("Usage: python check_brightness.py <path_to_your_image.jpg>")
    else:
        file_path = sys.argv[1]

        # Load the image from the file
        img = cv2.imread(file_path)

        if img is not None:
            # Analyze the image
            if is_image_dark(img):
                print("Result: The image is DARK.")
            else:
                print("Result: The image is BRIGHT.")
        else:
            print(f"Error: Could not load image from {file_path}")