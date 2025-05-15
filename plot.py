import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt


#mariam all
def traj_plot(camera_poses, scale=0.2):
    # Generate and visualize a 3D trajectory of camera coordinate frames
    frames = []
    for pose in camera_poses:
        cam_to_world = np.linalg.inv(pose)
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)
        frame.transform(cam_to_world)
        frames.append(frame)
    o3d.visualization.draw_geometries(frames, window_name="Camera Trajectory")
    

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