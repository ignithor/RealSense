import pyrealsense2 as rs
import numpy as np
import cv2

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

# Check if the D435i is connected (or another D400 series camera)
if (device_product_line == 'D400'):
    print("Connecting to D400 series camera (e.g., D435i)")
    # Configure the RGB stream
    # We use 640x480 @ 30fps. You can change this.
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
else:
    print(f"Connected to {device_product_line}. This script is optimized for D400 series.")
    # Fallback to a common configuration
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)
print("Pipeline started. Press 'q' to quit.")

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        # If no color frame is captured, skip this loop iteration
        if not color_frame:
            continue

        # Convert the color frame data to a numpy array
        # This is the image data we can process
        color_image = np.asanyarray(color_frame.get_data())

        # ----------------------------------
        # --- YOUR PROCESSING GOES HERE ---
        # ----------------------------------
        #
        # 'color_image' is your OpenCV-compatible BGR image.
        #
        # Example: Convert to grayscale
        # gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        #
        # Example: Draw a simple red rectangle
        # cv2.rectangle(color_image, (100, 100), (200, 200), (0, 0, 255), 2)
        #
        # For this demo, we will just display the original color_image.
        #
        processed_image = color_image # Replace 'color_image' with your processed frame
        

        # Display the resulting image
        cv2.namedWindow('RealSense RGB Stream', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense RGB Stream', processed_image)

        # Wait for a key press and exit if 'q' is pressed
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

finally:
    # Stop streaming and clean up
    print("Stopping pipeline...") 
    pipeline.stop()
    cv2.destroyAllWindows()