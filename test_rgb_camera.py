import pyrealsense2 as rs
import numpy as np
import cv2

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

print("Connecting to D435i camera...")
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# --- NOTE: We are enabling IR (1) and DISABLING Depth ---
config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30) # Index 1 for Left

# Start streaming
try:
    profile = pipeline.start(config)
    print("Pipeline started (Color + IR). Press 'q' to quit.")
except RuntimeError as e:
    print(f"Error starting pipeline: {e}")
    exit()

try:
    while True:
        frames = pipeline.wait_for_frames()
        
        color_frame = frames.get_color_frame()
        ir_frame = frames.get_infrared_frame(1) # Get IR frame
        
        if not color_frame or not ir_frame:
            continue

        # --- 1. Process Video Streams ---
        color_image = np.asanyarray(color_frame.get_data())
        
        # Get IR image (it's grayscale)
        ir_image = np.asanyarray(ir_frame.get_data())
        # Convert to 3-channel BGR so we can stack it
        ir_display = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)

        
        # --- 2. Display Streams ---
        # Stack images horizontally
        combined_display = np.hstack((color_image, ir_display))

        cv2.namedWindow('D435i Video Streams (RGB + Infrared)', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('D435i Video Streams (RGB + Infrared)', combined_display)
        
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

finally:
    print("Stopping pipeline...") 
    pipeline.stop()
    cv2.destroyAllWindows()