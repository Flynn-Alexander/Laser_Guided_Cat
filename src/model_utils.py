import torch
import glob
from imutils.video import VideoStream
from models.MiDaS.midas.model_loader import default_models, load_model
from models.MiDaS.run import process, create_side_by_side
from project_utils import *
import time
import numpy as np
import cv2
import sys


def collect_image_frame(visualise=False):
    src = 1     # 1 for laptop webcam
    vs = VideoStream(src).start()

    # Allow camera to warm up
    print('Collecting Image in\n3')
    time.sleep(1)
    print('2')
    time.sleep(1)
    print('1')
    time.sleep(1)
    frame = vs.read()
    print('Success: Image Collected\n')
    vs.stop()

    original_image_rgb = frame

    # Display the image
    if visualise:
        cv2.imshow('Original RGB Image', original_image_rgb)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return original_image_rgb


def initialise_MiDaS_model():
    """
    Initialise MiDaS model parameters

    Args:

    Returns:
        device (torch.device): device to run the model on
        model (torch.nn.Module): MiDaS model
    """

    # Set the device for the model to run on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set the model path and type
    MiDaS_model_name = "dpt_beit_large_512.pt"  # Default (slow but highest accuracy)
    model_path = f'models/MiDaS/weights/{MiDaS_model_name}'
    if not os.path.exists(model_path):
        url = f'https://github.com/isl-org/MiDaS/releases/download/v3_1/{MiDaS_model_name}'
        download_file(url, MiDaS_model_name, 'models/MiDaS/weights')
    model_type = MiDaS_model_name[:-3]

    # Load a MiDaS model
    optimize = False
    height = None
    square = False
    model, transform, net_w, net_h = load_model(device, model_path, model_type, optimize, height, square)

    return device, model, model_type, transform, net_w, net_h


def run_MiDaS_model(rgb_image,
                    device,
                    model,
                    model_type,
                    transform,
                    net_w,
                    net_h,
                    visualise=False,
                    used_camera=True,
                    optimize=False,
                    side=False,
                    grayscale=False
                    ):
    """
    Convert one RGB image to a depth map using a MiDaS model

    Args:
        rgb_image (np.array): RGB image to input into forward pass of MiDaS model
        device (torch.device): device to run the model on
        model (torch.nn.Module): MiDaS model
        model_type (str): the type of the model to be loaded
        transform (torchvision.transforms.Compose): image transformation pipeline
        net_w (int): network input width
        net_h (int): network input height
        visualise (bool): visualise the depth map?
        used_camera (bool): was the image captured using a camera?
        optimize (bool): optimize the model to half-integer on CUDA?
        side (bool): create a side-by-side image?
        grayscale (bool): convert the depth map to grayscale?
    """

    with torch.no_grad():
        if rgb_image is not None:
            # Apply transformations to the input image (resizing)
            image = transform({"image": rgb_image / 255})["image"]

            # Predict and normalise the depth map
            inverse_depth_map = process(device, model, model_type, image, (net_w, net_h), rgb_image.shape[1::-1], optimize, used_camera)
            content = create_side_by_side(rgb_image, inverse_depth_map, grayscale)

            # visualise results
            if visualise or side:
                cv2.imshow('MiDaS Depth Estimation - Press Escape to close window ', content / 255)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        else:
            print('Error: No or invalid scene image provided')
            sys.exit()

    return inverse_depth_map


def initialise_ZoeDepth_model():
    """
    Initialise ZoeDepth model parameters

    Returns:
        model_zoe_nk (torch.nn.Module): ZoeDepth model
    """

    repo = "isl-org/ZoeDepth"
    model_zoe_nk = torch.hub.load(repo, "ZoeD_NK", pretrained=True)  # ZoeDepth
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model_zoe_nk.to(device)