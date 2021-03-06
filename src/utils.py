import numpy as np

import nibabel as nib
from scipy import ndimage


def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan


def normalize(volume):
    """Normalize the volume"""
    min = np.min(volume)
    max = np.max(volume)
    volume = (volume) / (max)
    volume = volume.astype("float32")
    return volume


def pretty_dict(dict_x):
    """

    :param dict_x:
    :return:
    """
    str_line = ''.join(['-'] * 100)
    str_dict = f"{str_line}"
    for key, val in dict_x.items():
        str_dict += f"\n{key}: {val} ({type(val)})"
    str_dict += f"\n{str_line}"
    return str_dict


def resize_volume(img,
                  desired_width=128,
                  desired_height=128,
                  desired_depth=32,
                  ):
    """Resize across z-axis"""
    # Set the desired depth

    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img


def process_scan(path,
                 desired_width=128,
                 desired_height=128,
                 desired_depth=32,
                 ):
    """Read and resize volume"""
    # Read scan
    volume = read_nifti_file(path)
    # Normalize
    volume = normalize(volume)
    # Resize width, height and depth
    volume = resize_volume(volume,
                           desired_width,
                           desired_height,
                           desired_depth)
    return volume
