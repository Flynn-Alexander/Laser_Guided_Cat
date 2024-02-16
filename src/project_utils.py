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
from scipy.spatial import ConvexHull
import alphashape
from ultralytics import YOLO
import queue
import threading
import time

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


class MyCustomError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


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
        MyCustomError("Error: Camera not accessible")
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

                # Break the loop if 'esc' is pressed
                if cv2.waitKey(1) & 0xFF == ord('\x1b'):
                    break
            else:
                MyCustomError("Error: Unable to capture image")
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
        # Plot the depth map
        if True:
            display_depth_map_with_colorbar(depth_map)

        # Plot the point cloud
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


def isolate_ground_plane(pcd, rgb_image, visualise=False):
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
    inv_rotation = np.linalg.inv(rotation)

    # Generate a new rotated point cloud
    rotated_pcd = pcd.rotate(rotation, center=(0, 0, 0))
    """
    points = np.asarray(pcd.points)
    rotated_points = np.dot(points, rotation.T)
    rotated_points[:, 2] = -rotated_points[:, 2] # flip the Z axis
    rotated_pcd = o3d.geometry.PointCloud()
    rotated_pcd.points = o3d.utility.Vector3dVector(rotated_points)
    """

    # Determine the translation required to align the floor plane with the XZ plane
    rotated_floor_pcd = rotated_pcd.select_by_index(inliers)
    rotated_floor_points = np.asarray(rotated_floor_pcd.points)
    translation = -np.mean(rotated_floor_points[:, 1])

    # Determine the alpha shape of the floor plane
    floor_points_2D = rotated_floor_points[:, [0, 2]]
    sample = sample_spatial_points(floor_points_2D, 0.05)
    alpha_shape = alphashape.alphashape(sample, alpha=6)
    if alpha_shape.geom_type == 'MultiPolygon':
        MyCustomError("Your Floor Space is split into multiple polygons. Please try again.")

    if visualise:
        # Plot the 3D segmented point clouds
        if True:
            floor_pcd = pcd.select_by_index(inliers)
            objects_pcd = pcd.select_by_index(inliers, invert=True)
            if True:    # plot using O3D
                floor_pcd.paint_uniform_color([1, 0, 0])  # Floor in red
                objects_pcd.paint_uniform_color([0, 1, 0])  # Remaining objects in green
                o3d.visualization.draw_geometries([floor_pcd, objects_pcd])
            else:   # plot using matplotlib
                #points = np.asarray(floor_pcd.points)
                points = np.asarray(rotated_pcd.points)
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

        # Plot the 2D floor plane alpha shape
        if True:
            fig, ax = plt.subplots()
            # Plot the sampled points
            ax.scatter(*zip(*sample))
            # Plot the alpha shape
            if alpha_shape.geom_type == 'Polygon':
                x, y = alpha_shape.exterior.xy
                ax.plot(x, y, color="red")  # Red color for the alpha shape
            elif alpha_shape.geom_type == 'MultiPolygon':
                for polygon in alpha_shape.geoms:  # Use .geoms to iterate
                    x, y = polygon.exterior.xy
                    ax.plot(x, y, color="red")
            plt.show()

        # Plot the projected floor plane on the RGB image
        if True:
            # reproject the transformed floor points using the inverse transformation
            reprojected_floor_points = np.vstack((floor_points_2D[:, 0], np.zeros(floor_points_2D.shape[0]), floor_points_2D[:, 1])).T
            reprojected_floor_pcd = apply_3D_transformation(reprojected_floor_points, inv_rotation, np.array([0, translation, 0]), inverse_transformation=True)
            reprojected_floor_binary_map = pcd_to_pixel_binary_map(reprojected_floor_pcd)

            # reproject the alpha shape using the inverse transformation
            alpha_shape_points = [alpha_shape.exterior.interpolate(distance) for distance in np.arange(0, alpha_shape.length, 0.005)]
            x_coords = np.array([point.x for point in alpha_shape_points])
            z_coords = np.array([point.y for point in alpha_shape_points])
            alpha_shape_points = np.array((x_coords, np.zeros(len(alpha_shape_points)), z_coords)).T
            alpha_shape_pcd = apply_3D_transformation(alpha_shape_points, inv_rotation, np.array([0, translation, 0]), inverse_transformation=True)
            alpha_shape_binary_map = pcd_to_pixel_binary_map(alpha_shape_pcd)
            alpha_shape_binary_map = grow_binary_map(alpha_shape_binary_map, kernel_size=5)

            # recolour the RGB image
            overlayed_rgb_image = rgb_image.copy()
            overlayed_rgb_image[reprojected_floor_binary_map == 1] = [255, 0, 0]
            overlayed_rgb_image[alpha_shape_binary_map == 1] = [0, 0, 255]
            cv2.imshow('Overlayed RGB Image', overlayed_rgb_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return rotation, inv_rotation, translation, alpha_shape


def grow_binary_map(binary_map, kernel_size=3):
    """
    Grows a binary map by a specified kernel size.

    Args:
        binary_map (np.ndarray): binary map to grow
        kernel_size (int): kernel size to grow the binary map by

    Returns:
        np.ndarray: grown binary map
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(binary_map, kernel, iterations=1)


def apply_3D_transformation(points, rotation, translation, inverse_transformation=False):
    """
    Apply a transformation to a point cloud.

    Args:
        points (np.ndarray): point cloud
        rotation (np.ndarray): rotation matrix
        translation (np.ndarray): translation vector

    Returns:
        pcd (open3d.geometry.PointCloud): transformed point cloud
    """

    # input is np.ndarray
    if isinstance(points, np.ndarray):
        if inverse_transformation:     # apply the translation first
            points += -translation
            points = np.dot(points, rotation.T)
        else:       # apply the rotation first
            points = np.dot(points, rotation)
            points += translation
        transformed_pcd = o3d.geometry.PointCloud()
        transformed_pcd.points = o3d.utility.Vector3dVector(points)
        return transformed_pcd

    # input is open3d.geometry.PointCloud
    elif isinstance(points, o3d.geometry.PointCloud):
        if inverse_transformation:    # apply the translation first
            points.translate(-translation)
            points.rotate(rotation.T)
        else:       # apply the rotation first
            points.rotate(rotation)
            points.translate(translation)
        return points


def sample_spatial_points(data, grid_size):
    """
    Samples points evenly across a spatial dataset.

    Args:
        data (np.ndarray): spatial dataset
        grid_size (int): grid size to sample points from

    Returns:
        np.ndarray: sampled points
    """
    # Create grid
    x_max, y_max = np.max(data, axis=0)
    x_min, y_min = np.min(data, axis=0)
    grid_x, grid_y = np.mgrid[x_min:x_max:grid_size, y_min:y_max:grid_size]
    grid_centers = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

    # Sample closest point to each grid center
    sampled_points = []
    for center in grid_centers:
        distances = np.linalg.norm(data - center, axis=1)
        closest_point = data[np.argmin(distances)]
        sampled_points.append(closest_point)

    return np.array(sampled_points)


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


class VideoCapture:     # bufferless VideoCapture
  def __init__(self, source):
    self.cap = cv2.VideoCapture(source)
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get()


def control_loop(rotation, inv_rotation, translation, alpha_shape):
    """
    Estimate the orientation of the subject and instruct the laser to move accordingly.

    Args:
        rotation (np.ndarray): rotation matrix
        inv_rotation (np.ndarray): inverse rotation matrix (rotation back to camera space)
        translation (np.ndarray): translation vector
        alpha_shape (shapely.geometry.Polygon): alpha shape of the floor plane
    """

    # Load a YOLO model
    model = YOLO("models/YOLO/yolov8n-pose.pt")  # load a pretrained pose model

    # Initialise the camera
    print('\nInitialising Control Loop...')
    camera = VideoCapture(1)
    if not camera.cap.isOpened():
        MyCustomError("Error: Camera not accessible")
    time.sleep(2)

    # Start the control loop
    while True:
        # Read the latest frame from the camera
        frame = camera.read()
        #cv2.imshow("frame", frame/255.0)
        #if cv2.waitKey(1) & 0xFF == ord('\x1b'):
        #    break

        # Run the YOLO model
        #results = model(frame)[0]
        results = model(frame, show=True)[0]
        keypoints = results.keypoints.xy.numpy()

        for subject_num, subject_type in results.names.items():



        print('debug')


    #results = model.predict(source="1", show=True)  # predict on the webcam stream

class subject:
    def __init__(self, subject_type, id, department):
        self.subject_type = subject_type
        self.id = id
        self.keypoints = None

    def update_keypoints(self, keypoints):
        self.keypoints = keypoints

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
