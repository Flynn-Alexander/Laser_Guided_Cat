from ultralytics import YOLO
from models.MiDaS.midas.model_loader import default_models, load_model
from model_utils import *
from project_utils import *
import argparse
import torch
import os


def develop_3D_playground():
    """
    This function is used to develop the 3D playground (scene).

    Args:

    Returns:

    """

    # Initialise MiDaS model parameters
    device, model, model_type, transform, net_w, net_h = initialise_MiDaS_model()

    # Collect an image frame
    print('\nDeveloping 3D Playground (Scene)...')
    scene_image_rgb = collect_image_frame()

    # Load a MiDaS model
    inverse_depth_map = run_MiDaS_model(scene_image_rgb, device, model, model_type, transform, net_w, net_h, visualise=False, side=True)

    X, Y, Z = convert_depth_map_to_point_cloud(inverse_depth_map)

    #

    print('debug')

if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument('-i', '--input_path',
    #                    default=None,
    #                    help='Folder with input images (if no input path is specified, images are tried to be grabbed '
    #                         'from camera)'
    #                    )

    # Load a MiDaS model
    develop_3D_playground()

    # Load a YOLO model
    #model = YOLO("models/YOLO/yolov8n-pose.pt")  # load a pretrained model (recommended for training)
    #results = model.predict(source="1", show=True)  # predict on the webcam stream
