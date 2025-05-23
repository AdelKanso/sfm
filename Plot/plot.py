import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt


#mariam all
def traj_plot(camera_poses, scale=0.2):
    frames = []
    camera_centers = []

    for pose in camera_poses:
        cam_to_world = np.linalg.inv(pose)
        
        # Add coordinate frame at camera pose
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)
        frame.transform(cam_to_world)
        frames.append(frame)

        # Compute and store camera center (translation part)
        camera_center = cam_to_world[:3, 3]
        camera_centers.append(camera_center)

    # Build trajectory as a LineSet
    lines = [[i, i + 1] for i in range(len(camera_centers) - 1)]
    trajectory = o3d.geometry.LineSet()
    trajectory.points = o3d.utility.Vector3dVector(camera_centers)
    trajectory.lines = o3d.utility.Vector2iVector(lines)
    trajectory.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in lines])  # Red

    o3d.visualization.draw_geometries(frames + [trajectory], window_name="Camera Trajectory")

    

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
    
    
#mariam
def show_results(errors,camera_poses,total_points,total_colors):
    plt.title("Reprojection Errors")
    for i in range(len(errors)):
        plt.scatter(i, errors[i])
        
    plt.show()

    # Visualize the sparse 3D points and save
    sparse_save(total_points, total_colors,with_colors=False)
    # Show poses
    traj_plot(camera_poses)
    # Show Colorized and save
    sparse_save(total_points, total_colors,with_colors=True)