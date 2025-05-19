import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt

#mariam all

# New
def traj_plot(camera_poses, scale=0.2):
    """
    Visualizes camera poses as coordinate frames and a continuous trajectory line.

    Args:
        camera_poses (list): List of 4x4 pose matrices.
        scale (float): Size of coordinate frames.
    """
    frames = []
    trajectory_points = []

    for pose in camera_poses:
        cam_to_world = np.linalg.inv(pose)
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)
        frame.transform(cam_to_world)
        frames.append(frame)
        # Extract camera position (translation component)
        position = cam_to_world[:3, 3]
        trajectory_points.append(position)

    # Create a line set for the trajectory
    trajectory_points = np.array(trajectory_points)
    lines = [[i, i + 1] for i in range(len(trajectory_points) - 1)]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(trajectory_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0] for _ in lines]))  # Red trajectory line

    o3d.visualization.draw_geometries(frames + [line_set], window_name="Camera Trajectory")


def sparse_save(point_cloud, colors, with_colors=False) -> None:
    result_path="Results"
    # Prepare point cloud: scale and optionally apply colors
    points = point_cloud.reshape(-1, 3) * 200
    if with_colors:
        colors = colors.reshape(-1, 3) / 255.0

    # Filter out distant outliers using distance from mean
    mean = np.mean(points, axis=0)
    dist = np.linalg.norm(points - mean, axis=1)
    mask = dist < (np.mean(dist) + 300)
    points = points[mask]
    if with_colors:
        colors = colors[mask]

    # Create Open3D point cloud and optionally attach colors
    pointCloud = o3d.geometry.PointCloud()
    pointCloud.points = o3d.utility.Vector3dVector(points)
    if with_colors:
        pointCloud.colors = o3d.utility.Vector3dVector(colors)

    # Visualize and export point cloud to .ply file
    o3d.visualization.draw_geometries([pointCloud])
    name = result_path + "/colors.ply" if with_colors else result_path + "/no_colors.ply"
    o3d.io.write_point_cloud(name, pointCloud)
    
def show_image(title,image):
    cv2.imshow(title,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def show_plot(img,title):
    plt.figure(figsize=(6, 4))
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.show()