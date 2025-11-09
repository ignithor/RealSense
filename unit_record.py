import pyrealsense2 as rs
import open3d as o3d
import numpy as np
import cv2
import datetime  # For timestamps
import time      # To allow for auto-exposure
import os        # <-- Added for directory management

# --- 1. CONFIGURATION ---
DATA_ROOT_FOLDER = "data"
SUB_FOLDER = "utest"

# --- GET USER INPUT FOR FILENAME ---
print("Ready to capture.")
base_name = input(f"Enter a name for this capture (will be saved in './{DATA_ROOT_FOLDER}/{SUB_FOLDER}/'): ")
if not base_name:
    base_name = f"capture_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"No name given, using default: {base_name}")

# --- Create the full directory paths ---
target_dir = os.path.join(DATA_ROOT_FOLDER, SUB_FOLDER, base_name)
os.makedirs(target_dir, exist_ok=True)
print(f"Saving all files to: {os.path.abspath(target_dir)}")
# ---

# --- RealSense Pipeline Setup ---
pipeline = rs.pipeline()
config = rs.config()

# Configure all streams with a STABLE combination
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.infrared, 640, 480, rs.format.y8, 30)
config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200)
config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 400)

print("Starting pipeline...")
profile = pipeline.start(config)

# Alignment and PointCloud objects
align_to = rs.stream.depth
align = rs.align(align_to)
pc = rs.pointcloud()

print("Pipeline started. Warming up camera...")

# --- Warm-up Loop ---
# Wait for 30 frames to allow auto-exposure to settle
for _ in range(30):
    pipeline.wait_for_frames()

print("Camera ready. Capturing one frame...")

try:
    # --- 1. Frame Acquisition ---
    # Get a single, valid set of frames
    frames = None
    while not frames:
        frames = pipeline.wait_for_frames(timeout_ms=5000)  # Wait 5 sec

    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    ir_frame = frames.get_infrared_frame()
    accel_frame = frames.first(rs.stream.accel, rs.format.motion_xyz32f)
    gyro_frame = frames.first(rs.stream.gyro, rs.format.motion_xyz32f)

    if not all([depth_frame, color_frame, ir_frame, accel_frame, gyro_frame]):
        print("Error: Could not capture all required frames.")
        raise Exception("Incomplete frame set")

    # --- 2. Process Data ---
    color_image = np.asanyarray(color_frame.get_data())
    ir_image = np.asanyarray(ir_frame.get_data())
    accel_data = accel_frame.as_motion_frame().get_motion_data()
    gyro_data = gyro_frame.as_motion_frame().get_motion_data()

    # --- 3. Process 3D Point Cloud ---
    pc.map_to(color_frame)
    points = pc.calculate(depth_frame)

    v = points.get_vertices()
    vertices = np.asanyarray(v).view(np.float32).reshape(-1, 3)
    vertices[:, 1] = -vertices[:, 1]  # Flip Y-axis

    # Robustness checks
    if vertices.size == 0 or np.all(vertices == 0) or np.isnan(vertices).any():
        print("Error: Captured frame contains bad 3D data.")
        raise Exception("Invalid 3D data")

    # Get texture coordinates
    t = points.get_texture_coordinates()
    tex_coords = np.asanyarray(t).view(np.float32).reshape(-1, 2)

    # Map colors
    h, w, _ = color_image.shape
    u = (tex_coords[:, 0] * w).astype(int)
    v = (tex_coords[:, 1] * h).astype(int)
    np.clip(u, 0, w - 1, out=u)
    np.clip(v, 0, h - 1, out=v)
    colors_rgb = color_image[v, u][:, [2, 1, 0]] / 255.0

    # Create Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd.colors = o3d.utility.Vector3dVector(colors_rgb)

    # --- 4. Save All Data ---
    print("Saving captured frames...")

    # --- Define file paths inside the target directory ---
    rgb_filename = os.path.join(target_dir, f"{base_name}_rgb.png")
    ir_filename = os.path.join(target_dir, f"{base_name}_ir.png")
    pcd_filename = os.path.join(target_dir, f"{base_name}_pcd.pcd")
    imu_filename = os.path.join(target_dir, f"{base_name}_imu.txt")

    # Save 2D images
    cv2.imwrite(rgb_filename, color_image)
    cv2.imwrite(ir_filename, ir_image)
    print(f"Saved: {rgb_filename}")
    print(f"Saved: {ir_filename}")

    # Save 3D Point Cloud
    o3d.io.write_point_cloud(pcd_filename, pcd)
    print(f"Saved: {pcd_filename}")

    # Save IMU data
    with open(imu_filename, "w") as f:
        f.write(f"Accel: {accel_data.x}, {accel_data.y}, {accel_data.z}\n")
        f.write(f"Gyro: {gyro_data.x}, {gyro_data.y}, {gyro_data.z}\n")
    print(f"Saved: {imu_filename}")

    print("\nRecording complete.")


except Exception as e:
    print(f"An error occurred: {e}")

finally:
    print("Stopping pipeline...")
    pipeline.stop()
    print("Pipeline stopped.")