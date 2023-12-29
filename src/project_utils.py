import requests
import os
import sys
import numpy as np
from model_utils import *
import open3d as o3d


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


def convert_depth_map_to_point_cloud(inverse_depth_map):
    """
    Convert a depth map to a point cloud

    Args:
        inverse_depth_map (np.ndarray): depth map

    Returns:
        point_cloud (np.ndarray): point cloud
    """
    # Hardcoded intrinsic parameters (Macbook Webcam)
    fx = 1439.058594
    fy = 1437.825369
    cx = 965.957705
    cy = 528.668273
    intrinsic_matrix = np.array([[fx, 0, cx],
                               [0, fy, cy],
                               [0, 0, 1]])

    depth_map = 1.0/inverse_depth_map
    depth_map[inverse_depth_map == 0] = 0  # Handle infinite depth

    # Create a meshgrid of pixel coordinates
    height, width = depth_map.shape
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    x, y = np.meshgrid(x, y)

    # Reproject to 3D coordinates
    X = (x - cx) * depth_map / fx
    Y = (y - cy) * depth_map / fy
    Z = depth_map
    points_3D = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)

    # filter out invalid points
    valid_points = points_3D[~np.isinf(points_3D).any(axis=1)]

    # Convert to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_points)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])

    print('debug')
    return X, Y, Z

if __name__ == "__main__":
    # Develop a collection of calibration photos
    if True:
        capture_photos('../data/calibration_photos_macbook_webcam')