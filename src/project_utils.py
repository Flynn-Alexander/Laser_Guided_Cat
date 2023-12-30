import requests
import os
import sys
import numpy as np
from model_utils import *
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def capture_photos(save_dir, interval=5, total_photos=40):
    """
    Captures photos from the webcam and saves them to a specified directory.

    Args:
        save_dir (str): directory to save the photos to
        interval (int): interval between each photo capture (seconds)
        total_photos (int): total number of photos to capture
    """
    # Create the directory if it does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Start the webcam
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Camera not accessible")
        return

    last_saved_time = time.time()
    photo_count = 0

    try:
        while photo_count < total_photos:
            ret, frame = cap.read()
            if ret:
                # Display the frame in a window
                cv2.imshow('Webcam Stream', frame)

                # Check if it's time to save a photo
                if time.time() - last_saved_time >= interval:
                    last_saved_time = time.time()
                    photo_count += 1
                    file_name = os.path.join(save_dir, f'photo_{photo_count}.png')
                    cv2.imwrite(file_name, frame)
                    print(f"Captured {file_name}")

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("Error: Unable to capture image")
                break  # Break the loop if frame capture fails
    finally:
        # Release the camera and close the window
        cap.release()
        cv2.destroyAllWindows()


def download_file(url, filename, directory):
    """
    Download a file from a URL to a directory

    Args:
        url (str): URL of the file to download
        directory (str): directory to save the file to
    """
    # Create the combined filepath location
    file_location = os.path.join(directory, filename)

    # Make a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        print('Downloading MiDaS Model Parameters...')
        with open(file_location, 'wb') as f:
            f.write(response.content)
        print(f'Successfully downloaded {filename} parameters\n')
    else:
        print(f'Failed to download {filename} parameters')
        sys.exit()


def convert_depth_map_to_point_cloud(scene_image_rgb, inverse_depth_map, visualise=False):
    """
    Convert a depth map to a point cloud

    Args:
        scene_image_rgb (np.ndarray): RGB image
        inverse_depth_map (np.ndarray): depth map
        visualise (bool): visualise the point cloud scene?

    Returns:
        pcd (open3d.geometry.PointCloud): point cloud
    """
    # Hardcoded intrinsic parameters (Macbook Webcam)
    fx = 1439.058594
    fy = 1437.825369
    cx = 965.957705
    cy = 528.668273
    width = 1920
    height = 1080
    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    rgb_o3d = o3d.geometry.Image(scene_image_rgb.astype(np.uint8))

    depth_map = 1.0 / inverse_depth_map
    depth_map = depth_map * 3850.0  # scaling multiplier found through manual experimentation
    depth_map[depth_map >= 1.5] = np.nan # Cap the depth map at 3.5m
    depth_map *= 1000.0 # convert to mm
    depth_o3d = o3d.geometry.Image(np.uint16(depth_map))

    #display_depth_map_with_colorbar(depth_map)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_o3d, depth_o3d)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsics)
    pcd.transform([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])

    # Visualize the point cloud
    if visualise:
        if True:    # plot using O3D
            o3d.visualization.draw_geometries([pcd])
        else:   # plot using matplotlib
            points = np.asarray(pcd.points)
            points[:, 2] = -points[:, 2]
            sample = points[np.random.choice(points.shape[0], 10000), :]

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], s=1)
            ax.set_xlabel('X Axis')
            ax.set_ylabel('Y Axis')
            ax.set_zlabel('Z Axis')
            ax.set_title('3D Point Cloud')

            plt.show()

    return pcd


def filter_ground_plane(pcd, visualise=False):
    """
    Filter the ground plane from the point cloud.

    Args:
        pcd (open3d.geometry.PointCloud): The point cloud to filter.
        visualise (bool): visualise the filtered point clouds?
    """

    # Apply RANSAC to segment the largest plane (likely the floor)
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.015, ransac_n=3, num_iterations=1000)

    # Extract inliers and outliers
    floor = pcd.select_by_index(inliers)
    objects = pcd.select_by_index(inliers, invert=True)

    # Visualize the results
    floor.paint_uniform_color([1, 0, 0])  # Floor in red
    objects.paint_uniform_color([0, 1, 0])  # Remaining objects in green

    if visualise:
        o3d.visualization.draw_geometries([floor, objects])

    return floor


def rotate_open3d_point_cloud_y(pcd, angle_degrees):
    """
    Rotate an Open3D point cloud around the Y-axis by the given angle.

    Args:
    pcd (open3d.geometry.PointCloud): The point cloud to rotate.
    angle_degrees (float): The rotation angle in degrees.

    Returns:
    open3d.geometry.PointCloud: The rotated point cloud.
    """
    # Convert angle to radians
    angle_radians = np.radians(angle_degrees)

    # Define the rotation matrix about the Y-axis
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle([0, angle_radians, 0])

    # Apply the rotation
    rotated_pcd = pcd.rotate(rotation_matrix, center=(0, 0, 0))

    return rotated_pcd


def onclick(event):
    ix, iy = event.xdata, event.ydata
    print(f'x = {int(ix)}, y = {int(iy)}')


def display_depth_map_with_colorbar(depth_map):
    # Normalize the depth map (0-1)
    depth_min, depth_max = depth_map.min(), depth_map.max()
    normalized_depth = depth_map / depth_max

    # Apply a colormap (e.g., 'jet') and display the depth map
    plt.imshow(depth_map, cmap='jet')

    # Add a color bar with labels
    cbar = plt.colorbar()
    cbar.set_label('Normalized Depth')

    # Add title and labels as needed
    plt.title('Depth Map with Color Bar')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')

    plt.show()


if __name__ == "__main__":
    # Develop a collection of calibration photos
    if False:
        capture_photos('../data/calibration_photos_macbook_webcam')

    # Find the pixel coordinates in a photo
    if False:
        fig = plt.figure()
        img = mpimg.imread('depth_map.png')  # Replace with your image path
        ax = fig.add_subplot(111)
        ax.imshow(img)

        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
