from ultralytics import YOLO
from models.MiDaS.midas.model_loader import default_models, load_model
from model_utils import *
from project_utils import *
import argparse
import torch
import os
from PIL import Image
from models.ZoeDepth.zoedepth.utils.misc import colorize


def develop_3D_playground(depth_model='MiDaS'):
    """
    This function is used to develop the 3D playground (scene).

    Args:
        depth_model (str): depth estimation model to use (MiDaS or ZoeDepth)

    Returns:

    """

    # Initialise model parameters
    if depth_model == 'MiDaS':
        device, model, model_type, transform, net_w, net_h = initialise_MiDaS_model()     # MiDaS
    elif depth_model == 'ZoeDepth':
        zoe = initialise_ZoeDepth_model()

    # Collect an image frame
    print('\nDeveloping 3D Playground (parameterised floor scene)...')
    rgb_image = collect_image_frame()

    # Run the model
    if depth_model == 'MiDaS':
        inverse_depth_map = run_MiDaS_model(rgb_image, device, model, model_type, transform, net_w, net_h, visualise=False, side=False)
        pcd = convert_depth_map_to_point_cloud(rgb_image, inverse_depth_map, depth_model='MiDaS')    # Generate point cloud data of the static scene
    elif depth_model == 'ZoeDepth':
        image = Image.fromarray(rgb_image)
        depth_map = zoe.infer_pil(image)
        pcd = convert_depth_map_to_point_cloud(rgb_image, depth_map, depth_model='ZoeDepth', visualise=True)  # Generate point cloud data of the static scene

    # Filter the floor plane to establish the 3D playground
    rotation, inv_rotation, translation, alpha_shape = isolate_ground_plane(pcd, rgb_image, visualise=True)

    return rotation, inv_rotation, translation, alpha_shape


if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument('-i', '--input_path',
    #                    default=None,
    #                    help='Folder with input images (if no input path is specified, images are tried to be grabbed '
    #                         'from camera)'
    #                    )

    # Develop a 3D playground (parameterised floor scene)
    # rotation, inv_rotation, translation, alpha_shape = develop_3D_playground("MiDaS")
    rotation, inv_rotation, translation, alpha_shape = develop_3D_playground("ZoeDepth")

    # Load a YOLO model
    #model = YOLO("models/YOLO/yolov8n-pose.pt")  # load a pretrained model (recommended for training)
    #results = model.predict(source="1", show=True)  # predict on the webcam stream
