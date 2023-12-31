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
from scipy.spatial.transform import Rotation as R

# Hardcoded intrinsic parameters (Macbook Webcam)
fx = 1439.058594
fy = 1437.825369
cx = 965.957705
cy = 528.668273
width = 1920
height = 1080
camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)


def onclick(event):
    ix, iy = event.xdata, event.ydata
    print(f'x = {int(ix)}, y = {int(iy)}')


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


def convert_depth_map_to_point_cloud(rgb_image, depth_map, depth_model, visualise=False):
    """
    Convert a depth map to a point cloud

    Args:
        rgb_image (np.ndarray): RGB image
        depth_map (np.ndarray): depth map
        depth_model (str): depth estimation model used to generate the depth map
        visualise (bool): visualise the point cloud scene?

    Returns:
        pcd (open3d.geometry.PointCloud): point cloud
    """

    rgb_o3d = o3d.geometry.Image(rgb_image.astype(np.uint8))

    if depth_model == 'MiDaS':
        depth_map = 1.0 / depth_map # convert inverse depth map to depth map
        depth_map = depth_map * 3850.0  # scaling multiplier found through manual experimentation
    depth_map[depth_map >= 5] = np.nan # Cap the depth map at 5m
    depth_map *= 1000.0 # convert to mm
    depth_o3d = o3d.geometry.Image(np.uint16(depth_map))

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_o3d, depth_o3d, depth_trunc=5.0)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsics)
    pcd.transform([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])

    # Visualize the point cloud
    if visualise:
        #display_depth_map_with_colorbar(depth_map)
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


def filter_ground_plane(pcd, rgb_image, visualise=False):
    """
    Filter the ground plane from the point cloud.

    Args:
        pcd (open3d.geometry.PointCloud): The point cloud to filter.
        visualise (bool): visualise the filtered point clouds?
    """

    # Apply RANSAC to segment the largest plane (likely the floor)
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.03, ransac_n=3, num_iterations=1000)

    # Determine the rotation required to align the floor plane with the Y-axis
    plane_normal = np.array(plane_model[:3])
    y_normal = np.array([0, 1, 0])
    rotation_axis = np.cross(plane_normal, y_normal)
    rotation_axis_normalized = rotation_axis / np.linalg.norm(rotation_axis)
    rotation_angle = np.arccos(np.dot(plane_normal, y_normal))
    rotation_vector = rotation_axis_normalized * rotation_angle
    rotation = R.from_rotvec(rotation_vector).as_matrix()
    points = np.asarray(pcd.points)
    rotated_points = np.dot(points, rotation.T)
    rotated_points[:, 2] = -rotated_points[:, 2]
    rotated_pcd = o3d.geometry.PointCloud()
    rotated_pcd.points = o3d.utility.Vector3dVector(rotated_points)

    sample = rotated_points[np.random.choice(points.shape[0], 10000), :]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], s=1)
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('3D Point Cloud')

    plt.show()

    floor = rotated_pcd.select_by_index(inliers)
    objects = rotated_pcd.select_by_index(inliers, invert=True)
    #TODO implement rotation and translation, a 2D view of the floor plane and its transformation back to the pixel space.

    # Extract inliers and outliers
    #floor = pcd.select_by_index(inliers)
    #objects = pcd.select_by_index(inliers, invert=True)

    # Visualize the results
    floor.paint_uniform_color([1, 0, 0])  # Floor in red
    objects.paint_uniform_color([0, 1, 0])  # Remaining objects in green

    if visualise:
        #o3d.visualization.draw_geometries([floor, objects])
        if True:    # plot using O3D
            o3d.visualization.draw_geometries([floor])
        else:   # plot using matplotlib
            points = np.asarray(floor.points)
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

    floor_binary_map = pcd_to_pixel_binary_map(floor)
    # make the rgb image red everywhere the binary map is 1
    overlayed_rgb_image = rgb_image.copy()
    overlayed_rgb_image[floor_binary_map == 1] = [255, 0, 0]
    cv2.imshow('Overlayed RGB Image', overlayed_rgb_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    return floor_binary_map


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


def project_to_pixel(point):
    # Apply the intrinsic parameters to project the 3D point to 2D
    u = (fx * point[0] / point[2]) + cx
    v = (fy * point[1] / point[2]) + cy

    # Ensure the pixel coordinates are within the image dimensions
    u = np.clip(u, 0, width - 1)
    v = np.clip(v, 0, height - 1)

    # Flip the pixel coordinates horizontally to match the image
    u = width - u

    return int(u), int(v)


def pcd_to_pixel_binary_map(pcd):
    binary_map = np.zeros((height, width), dtype=np.uint8)

    for point in np.asarray(pcd.points):
        pixel = project_to_pixel(point)
        if 0 <= pixel[0] < width and 0 <= pixel[1] < height:
            binary_map[pixel[1], pixel[0]] = 1

    return binary_map


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
