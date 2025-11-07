import pyrealsense2 as rs
import open3d as o3d
import numpy as np
import cv2

# --- Open3D Visualizer Setup ---
vis = o3d.visualization.Visualizer()
vis.create_window('L515 Colored Point Cloud', width=960, height=540)
pcd = o3d.geometry.PointCloud()
vis.add_geometry(pcd)
opt = vis.get_render_option()
opt.point_size = 1.0
# --- End of Open3D Setup ---

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
print("Pipeline started.")

# Flag for 3D view reset
first_frame = True

try:
    while True:
        # --- 1. Frame Acquisition ---
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        ir_frame = frames.get_infrared_frame()
        accel_frame = frames.first(rs.stream.accel, rs.format.motion_xyz32f)
        gyro_frame = frames.first(rs.stream.gyro, rs.format.motion_xyz32f)
        
        if not depth_frame or not color_frame or not ir_frame or not accel_frame or not gyro_frame:
            continue

        # --- 2. Process & Display 2D Streams (RGB, IR, IMU) ---
        color_image = np.asanyarray(color_frame.get_data())
        cv2.imshow('RGB Stream', color_image)

        ir_image = np.asanyarray(ir_frame.get_data())
        cv2.imshow('IR Stream', ir_image)

        accel_data = accel_frame.as_motion_frame().get_motion_data()
        gyro_data = gyro_frame.as_motion_frame().get_motion_data()
        imu_dashboard = np.zeros((120, 400, 3), dtype=np.uint8)
        accel_str = f"Accel: X={accel_data.x:.3f} Y={accel_data.y:.3f} Z={accel_data.z:.3f}"
        gyro_str  = f"Gyro:  X={gyro_data.x:.3f} Y={gyro_data.y:.3f} Z={gyro_data.z:.3f}"
        cv2.putText(imu_dashboard, accel_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(imu_dashboard, gyro_str, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow('IMU Data', imu_dashboard)


        # --- 3. Process & Display 3D Point Cloud ---
        pc.map_to(color_frame)
        points = pc.calculate(depth_frame)

        v = points.get_vertices()
        vertices = np.asanyarray(v).view(np.float32).reshape(-1, 3)

        # --- FIX FOR REVERSED POINT CLOUD ---
        # Flip the Y-axis to correct for the coordinate system difference
        vertices[:, 1] = -vertices[:, 1]
        # --- END OF FIX ---

        # Robustness checks
        if vertices.size == 0 or np.all(vertices == 0) or np.isnan(vertices).any():
            vis.poll_events(); vis.update_renderer() 
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue

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

        # Update Open3D
        pcd.points = o3d.utility.Vector3dVector(vertices)
        pcd.colors = o3d.utility.Vector3dVector(colors_rgb)
        vis.update_geometry(pcd)
        
        if first_frame:
            vis.reset_view_point(True)
            first_frame = False

        vis.poll_events()
        vis.update_renderer()
        
        # --- 4. Quit Condition ---
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    print("Stopping pipeline...")
    pipeline.stop()
    cv2.destroyAllWindows()
    vis.destroy_window()
    print("Pipeline stopped.")