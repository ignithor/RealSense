import pyrealsense2 as rs
import open3d as o3d
import numpy as np
import cv2
import datetime
import time
import os

# --- 1. CONFIGURATION ---
RECORD_DURATION_SEC = 3.0
DATA_ROOT_FOLDER = "data"  # <-- New top-level folder

# --- GET USER INPUT FOR FILENAME ---
print("Ready to record.")
base_name = input(f"Enter a name for this recording (will be saved in '{DATA_ROOT_FOLDER}/'): ")
if not base_name:
    base_name = f"capture_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"No name given, using default: {base_name}")

# --- Create the full directory paths ---
target_dir = os.path.join(DATA_ROOT_FOLDER, base_name)
pcd_folder = os.path.join(target_dir, "pcds")

# Create all directories (e.g., data/my_test/pcds/)
os.makedirs(pcd_folder, exist_ok=True)
print(f"Saving all files to: {os.path.abspath(target_dir)}")

# --- File output paths ---
rgb_video_file = os.path.join(target_dir, f"{base_name}_rgb.avi")
ir_video_file = os.path.join(target_dir, f"{base_name}_ir.avi")
imu_csv_file = os.path.join(target_dir, f"{base_name}_imu.csv")

# Frame properties (must match stream config)
WIDTH, HEIGHT = 640, 480
FPS = 30

# --- 2. RealSense Pipeline Setup ---
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
config.enable_stream(rs.stream.infrared, WIDTH, HEIGHT, rs.format.y8, FPS)
config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200)
config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 400)

print("Starting pipeline...")
profile = pipeline.start(config)
align_to = rs.stream.depth
align = rs.align(align_to)
pc = rs.pointcloud()
print("Pipeline started.")

# --- 3. Setup In-Memory Buffers ---
rgb_frames_for_video = []
color_frames_for_pcd = []
ir_frames_for_video = []
depth_frames_for_pcd = []
accel_data_list = []
gyro_data_list = []
timestamps = []

# --- 4. Warm-up Loop ---
print("Warming up camera...")
for _ in range(30):  # Wait for auto-exposure
    pipeline.wait_for_frames()
print("Camera ready.")

# --- 5. High-Speed Capture Loop ---
try:
    print(f"Recording for {RECORD_DURATION_SEC} seconds...")
    start_time = time.time()
    frame_count = 0
    while (time.time() - start_time) < RECORD_DURATION_SEC:

        try:
            frames = pipeline.wait_for_frames(timeout_ms=1000)  # Wait for a frame

            # Get all frames
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            ir_frame = frames.get_infrared_frame()
            accel_frame = frames.first(rs.stream.accel, rs.format.motion_xyz32f)
            gyro_frame = frames.first(rs.stream.gyro, rs.format.motion_xyz32f)

            if not depth_frame or not color_frame or not ir_frame or not accel_frame or not gyro_frame:
                print("Skipped incomplete frame set")
                continue

            # Add data to buffers
            rgb_frames_for_video.append(np.asanyarray(color_frame.get_data()))
            ir_frames_for_video.append(np.asanyarray(ir_frame.get_data()))

            # Store the objects for 3D processing
            color_frames_for_pcd.append(color_frame)
            depth_frames_for_pcd.append(depth_frame)

            accel_data_list.append(accel_frame.as_motion_frame().get_motion_data())
            gyro_data_list.append(gyro_frame.as_motion_frame().get_motion_data())
            timestamps.append(frames.get_timestamp())

            frame_count += 1

        except RuntimeError:
            print("Dropped a frame!")

    print(f"Recording complete. Captured {frame_count} frames.")

# --- 6. Stop Pipeline ---
finally:
    print("Stopping pipeline...")
    pipeline.stop()
    print("Pipeline stopped.")

# --- 7. Post-Processing and Saving (Slow) ---
print("Now processing and saving data. This may take a while...")

try:
    # A. Save RGB and IR Video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    rgb_out = cv2.VideoWriter(rgb_video_file, fourcc, FPS, (WIDTH, HEIGHT))
    ir_out = cv2.VideoWriter(ir_video_file, fourcc, FPS, (WIDTH, HEIGHT), isColor=False)

    for i in range(len(rgb_frames_for_video)):
        rgb_out.write(rgb_frames_for_video[i])
        ir_out.write(ir_frames_for_video[i])

    rgb_out.release()
    ir_out.release()
    print(f"Saved: {rgb_video_file}")
    print(f"Saved: {ir_video_file}")

    # B. Save IMU Data
    with open(imu_csv_file, "w") as f:
        f.write("timestamp_ms,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z\n")
        for i in range(len(timestamps)):
            ts = timestamps[i]
            accel = accel_data_list[i]
            gyro = gyro_data_list[i]
            f.write(f"{ts},{accel.x},{accel.y},{accel.z},{gyro.x},{gyro.y},{gyro.z}\n")
    print(f"Saved: {imu_csv_file}")

    # C. Save Point Clouds (This is the slowest part)
    print(f"Processing {len(depth_frames_for_pcd)} point clouds...")
    for i in range(len(depth_frames_for_pcd)):

        # Get the buffered frame OBJECTS
        d_frame = depth_frames_for_pcd[i]
        c_frame = color_frames_for_pcd[i]
        c_image_numpy = rgb_frames_for_video[i]  # Get the numpy version for mapping

        # --- Re-create the point cloud ---
        pc.map_to(c_frame)  # Use the color_frame object
        points = pc.calculate(d_frame)  # Use the depth_frame object

        v = points.get_vertices()
        vertices = np.asanyarray(v).view(np.float32).reshape(-1, 3)
        vertices[:, 1] = -vertices[:, 1]  # Flip Y-axis

        if vertices.size == 0 or np.all(vertices == 0) or np.isnan(vertices).any():
            continue  # Skip bad frames

        t = points.get_texture_coordinates()
        tex_coords = np.asanyarray(t).view(np.float32).reshape(-1, 2)

        h, w, _ = c_image_numpy.shape
        u = (tex_coords[:, 0] * w).astype(int)
        v = (tex_coords[:, 1] * h).astype(int)
        np.clip(u, 0, w - 1, out=u)
        np.clip(v, 0, h - 1, out=v)
        colors_rgb = c_image_numpy[v, u][:, [2, 1, 0]] / 255.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)
        pcd.colors = o3d.utility.Vector3dVector(colors_rgb)

        # Save the individual PCD file
        pcd_filename = f"frame_{i:04d}.pcd"  # e.g., frame_0001.pcd
        o3d.io.write_point_cloud(os.path.join(pcd_folder, pcd_filename), pcd)

        # Print progress
        if (i + 1) % 10 == 0:
            print(f"  ... saved {i + 1}/{len(depth_frames_for_pcd)} point clouds")

    print(f"Saved all point clouds to: {pcd_folder}")
    print("\nAll processing complete.")

except Exception as e:
    print(f"An error occurred during post-processing: {e}")