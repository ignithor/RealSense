import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

def darken_image_to_limit(image_path: str):
    """
    Loads an image and iteratively darkens it until the mean 
    V-channel (brightness) is <= 1, saving each step.
    
    Finally, it plots all steps in a single grid.
    """
    
    # 1. Load the image
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # List to store all the frames for plotting
    # We will store (rgb_image, mean_v_value) tuples
    plot_data = []

    # 2. Convert to HSV and split channels
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Create the subtraction array (3) in the same type as v
    subtract_array = np.full(v.shape, 3, dtype=v.dtype)

    print("Darkening image...")
    
    # 3. The Core Loop
    while True:
        # Calculate the current mean brightness
        current_mean_v = np.mean(v)
        
        # --- Store current frame for plotting ---
        # We must convert it back to BGR, then to RGB for matplotlib
        hsv_current = cv2.merge([h, s, v])
        bgr_current = cv2.cvtColor(hsv_current, cv2.COLOR_HSV2BGR)
        rgb_current = cv2.cvtColor(bgr_current, cv2.COLOR_BGR2RGB)
        
        plot_data.append((rgb_current, current_mean_v))
        print(f"  Saved frame with Mean V: {current_mean_v:.2f}")

        # --- Check stop condition ---
        if current_mean_v <= 3.0:
            print("Mean brightness is <= 3.0. Stopping.")
            break
            
        # --- Apply the change for the *next* loop ---
        # Use cv2.subtract for correct saturation at 0 (no underflow)
        v = cv2.subtract(v, subtract_array)
        
        # Safety check: if all values are 0, stop
        if np.all(v == 0):
            print("All pixels are black. Stopping.")
            break

    # 4. Plotting
    print("Generating plot...")
    num_images = len(plot_data)
    
    # Calculate grid size (e.g., 16 images -> 4x4)
    cols = int(np.ceil(np.sqrt(num_images)))
    rows = int(np.ceil(num_images / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    
    # Flatten the axes array for easy iteration
    axes = axes.flat
    
    for i in range(num_images):
        img, mean_v = plot_data[i]
        ax = axes[i]
        ax.imshow(img)
        ax.set_title(f"Mean V: {mean_v:.1f}", fontsize=8, color='white')
        ax.axis('off')

    # Hide any unused subplots in the grid
    for i in range(num_images, len(axes)):
        axes[i].axis('off')
        
    fig.suptitle(f"Image Darkening Process (Decrease by 3V per step)")
    plt.tight_layout()
    plt.show()

# --- Main execution block ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python darken.py <path_to_image>")
        sys.exit(1)
        
    image_file = sys.argv[1]
    darken_image_to_limit(image_file)