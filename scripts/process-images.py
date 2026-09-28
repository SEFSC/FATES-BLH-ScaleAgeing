#!/usr/bin/env python

"""
Process Raw Scale Images for Ageing
-----------------------------------
This script processes raw fish scale images by cropping and padding them using
either simple binary thresholding or a Segment Anything Model (SAM). Additional
options include image normalization and grey scale color inversion. Arguments,
hyperparameters, and other settings are included in a configs.yml file. New
cropped images are saved to an output directory specified in configs.yml.

Usage:
    python Scale_Raw_Image_Preprocessing.py --config_path path/to/configs.yml
    python Scale_Raw_Image_Preprocessing.py -c path/to/configs.yml

Authors: aotian.zheng@noaa.gov (script development) and matt.grossi@noaa.gov
         (implementation, user functionality, documentation) with assistance
         from Google Gemini Coding Partner
Version: 2026.1.2
Release Date: July 2025
Last Updated: September 2026
"""

import argparse
import difflib
import os
from pathlib import Path
import yaml

import cv2 as cv
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import torch
from tqdm import tqdm

def load_yaml(file_path: str | Path) -> dict:
    """Load a YAML configuration file with fallback support for raw Windows
    backslashes.

    Attributes
    ----------
    file_path : str | Path
        Path to the YAML configuration file.
    """
    path = Path(file_path)
    content = path.read_text(encoding='utf-8')

    try:
        config = yaml.safe_load(content) or {}
        clean_and_validate_config(config=config)
        return config
    except yaml.YAMLError as exc:
        # Check if the failure is likely caused by backslash escape codes
        if '\\' in content:
            # Convert backslashes to forward slashes and attempt a second parse
            config = yaml.safe_load(content.replace('\\', '/')) or {}
            clean_and_validate_config(config=config)
            return config
        
        # If there are no backslashes, raise the actual YAML syntax error
        raise

def clean_and_validate_config(config: dict):
    """Checks for missing mandatory keys and typos in the YAML. Suggests the
    closest-match valid key for any invalid key found. Cleans any string values
    when Bools are expected and ensures image file extension, if passed, contains
    a leading ".".
    
    Arguments
    ---------
    config (dict): dictionary of configuration settings to validate
    """
    # Define expected keys
    REQUIRED_KEYS = {
        'processed_image_path', 'raw_image_path', 'input_type'
        }
    VALID_KEYS = {
        'binary_threshold',
        'bottom_pad',
        'collection_date_colname',
        'downsample',
        'fish_id_colname',
        'fish_length_colname',
        'fish_weight_colname',
        'input_type',
        'invert',
        'metadata_csv_file',
        'model_pth_file',
        'normalization',
        'output_csv_file',
        'output_type',
        'pad',
        'points_per_side',
        'processed_image_path',
        'raw_image_path',
        'sam_model_type',
        'sam_weights_path',
        'segment',
        'stability_score_thresh'
        }
    
    # Find the differences using set math
    config_keys = set(config.keys())
    missing_keys = REQUIRED_KEYS - config_keys
    unrecognized_keys = config_keys - VALID_KEYS

    error_blocks = []

    # Check for missing keys
    if missing_keys:
        missing_msg = "[!] MISSING REQUIRED SETTINGS:\n" + "\n".join(f"  - '{k}'" for k in missing_keys)
        error_blocks.append(missing_msg)

    # Check for typos or unrecognized keys
    if unrecognized_keys:
        unrecognized_msg_lines = []
        for key in unrecognized_keys:
            matches = difflib.get_close_matches(key, list(VALID_KEYS), n=1, cutoff=0.6)
            suggestion = f" (Did you mean '{matches[0]}'?)" if matches else ""
            unrecognized_msg_lines.append(f"  - '{key}'{suggestion}")
        
        unrecognized_msg = "[!] UNRECOGNIZED SETTINGS FOUND:\n" + "\n".join(unrecognized_msg_lines)
        error_blocks.append(unrecognized_msg)

    if error_blocks:
        final_error_message = (
            "\n\nCONFIGURATION ERROR(S) DETECTED:\n\n" +
            "\n\n".join(error_blocks) +
            "\n\nPlease correct your configuration file and restart the utility."
        )
        raise ValueError(final_error_message)
        
    # Fix capitalized file extensions
    for k, v in config.items():
        if '_file' in k:
            _, ext = os.path.splitext(v)
            v = v.replace(ext.upper(), ext.lower())

    # Format directories for cross-platform compatibility
    config.update(
        {k: Path(i) for k,i in config.items() if 'path' in k or 'file' in k}
        )
    
    # Check and fix image type file extensions, if necessary
    if 'input_type' in config and not config['input_type'].startswith('.'):
        config['input_type'] = '.' + config['input_type']
    if 'output_type' in config and not config['output_type'].startswith('.'):
        config['output_type'] = '.' + config['output_type']

    value_errors = []

    # Verify segment key
    segment = config.get('segment')
    if segment is not None and segment not in {'binary', 'sam'}:
        value_errors.append(f"  - 'segment' must be 'binary' or 'sam' (got: '{segment}')")

    # Verify stability score threshold (0 to 1 inclusive)
    thresh = config.get('stability_score_thresh')
    if thresh is not None:
        if not isinstance(thresh, (int, float)) or not (0 <= thresh <= 1):
            value_errors.append(f"  - 'stability_score_thresh' must be a number between 0 and 1 (got: {thresh})")

    # Verify downsample (< 1)
    downsample = config.get('downsample')
    if downsample is not None:
        if not isinstance(downsample, (int, float)) or downsample >= 1:
            value_errors.append(f"  - 'downsample' must be a number less than 1 (got: {downsample})")

    # Verify Segment Anything Model (SAM) type
    sam_type = config.get('sam_model_type')
    if sam_type is not None and sam_type not in {'vit_b', 'vit_l', 'vit_h'}:
        value_errors.append(f"  - 'sam_model_type' must be 'vit_b', 'vit_l', or 'vit_h' (got: '{sam_type}')")

    # Extract the segment value, defaulting to an empty string if missing, 
    # and safely convert it to lowercase for comparison.
    segment_type = str(config.get('segment', '')).lower()
    if segment_type == 'sam':
        # If segment is 'sam', ensure 'sam_weights_path' exists in the dictionary
        # and that it isn't completely empty/None
        if not config.get('sam_weights_path'):
            value_errors.append(
                "  - 'sam_weights_path' is missing or empty, but is required when 'segment' is set to 'sam'."
            )
        elif ".pth" not in config["sam_weights_path"]:
            value_errors.append("  - The 'sam_weights_path' key in the configuration file must include a file name ending with '.pth'.")

    # Verify normalization key
    norm = config.get('normalization')
    if norm is not None and str(norm).lower() not in {'none', 'he', 'clahe'}:
        value_errors.append(f"  - 'normalization' must be 'none', 'he', or 'clahe' (got: '{norm}')")

    # Verify invert key (handle YAML bools, ints, and case-insensitive strings)
    invert = config.get('invert')
    if invert is not None:
        # If the user put quotes around it, it's a string, so we lowercase it. 
        # Otherwise, YAML parsed it as a bool or int.
        invert_check = invert.lower() if isinstance(invert, str) else invert
        if invert_check not in {True, False, 1, 0, 'true', 'false', '1', '0'}:
            value_errors.append(f"  - 'invert' must be True, False, 1, or 0 (got: '{invert}')")

    # Add collected value errors to the main error blocks
    if value_errors:
        error_blocks.append("[!] INVALID VALUES FOUND:\n" + "\n".join(value_errors))

    # Report config issues in need of fixing
    if error_blocks:
        final_error_message = (
            "\n\nCONFIGURATION ERROR(S) DETECTED:\n\n" +
            "\n\n".join(error_blocks) +
            "\n\nPlease correct your configuration file and restart the utility."
        )
        raise ValueError(final_error_message)    

def combine_masks(annotations):
    """Combine overlapping masks into single masks.
    
    Parameters
    ----------
    annotations : list of dict
        List of annotations, where each annotation is a dictionary containing 'segmentation' (2D numpy array) and 'bbox' (list of [x, y, width, height]).

    Returns
    -------
    list of dict
        List of combined annotations, where overlapping masks have been merged into single masks.
    """

    # Calculate area of first image annotation
    img_area = annotations[0]['segmentation'].shape[0]*annotations[0]['segmentation'].shape[1]

    # Remove any background masks that cover more than 90% of the image area
    foreground_anns = []
    for ann in annotations:
        xmin, ymin, w, h = ann['bbox']
        if(w*h < 0.9*img_area):
            foreground_anns.append(ann)
    # Keep track of which masks should be deleted
    del_indices = [0]
    while(len(del_indices) > 0):
        del_indices = []
        # Loop through all pairs of masks and combine overlapping ones
        for i in range(len(foreground_anns)):
            if(i not in del_indices):
                for j in range(i+1, len(foreground_anns)):
                    # Check if bounding boxes overlap
                    if(j not in del_indices):
                        left1, top1, w1, h1 = foreground_anns[i]['bbox']
                        left2, top2, w2, h2 = foreground_anns[j]['bbox']
                        right1 = left1+w1
                        right2 = left2+w2
                        bot1= top1+h1
                        bot2 = top2+h2
                        margin = 10
                        # If bounding boxes overlap, combine the masks
                        if(right1 >= left2-margin and left1 <= right2+margin and bot1 >= top2-margin and top1 <= bot2+margin):
                            foreground_anns[i]['segmentation'] = np.logical_or(foreground_anns[i]['segmentation'], foreground_anns[j]['segmentation'])
                            left = min(left1, left2)
                            right = max(right1, right2)
                            top = min(top1, top2)
                            bot = max(bot1, bot2)
                            foreground_anns[i]['bbox'] = [left, top, right-left, bot-top]
                            del_indices.append(j)
                            break
        # Remove the masks that were combined with anther mask
        combined_anns = []
        for i in range(len(foreground_anns)):
            if i not in del_indices:
                combined_anns.append(foreground_anns[i])
        foreground_anns = combined_anns

    return combined_anns

def choose_best(annotations):
    """Choose the best mask from a list of masks based on area and distance from center.
    
    Parameters
    ----------
    anns : list of dict
        List of annotations, where each annotation is a dictionary containing 'segmentation' (2D numpy array) and 'bbox' (list of [x, y, width, height]).
    
    Returns
    -------
    dict
        The annotation dictionary corresponding to the best mask.
    """

    scores = []
    
    # Get maximum area and center of image
    h, w = annotations[0]['segmentation'].shape[0], annotations[0]['segmentation'].shape[1]
    centx = w/2
    centy = h/2
    max_area = 0
    for i in range(len(annotations)):
        bbox = annotations[i]['bbox']
        area = bbox[3]*bbox[2]
        if(area > max_area):
            max_area = area
    for i in range(len(annotations)):
        score = 0
        anno = annotations[i]
        bbox = anno["bbox"]
        # Calculate mask size score based on bbox area
        area_score = bbox[2]*bbox[3]/max_area
        # Distance from center score (closer is better, so higher score)
        distance = (bbox[0]+bbox[2]*0.5-centx)**2 + (bbox[1]+bbox[3]*0.5-centy)**2
        distance_score = 1-(distance/(centy**2 + centx**2))
        score = area_score + distance_score
        scores.append(score)
    mask_index = scores.index(max(scores))

    return annotations[mask_index]

def crop_and_pad_binary_threshold(image, kernel_size=10, threshold=100, pad=0.05, bottom_pad=0.35):
    """Crop and pad an image using binary thresholding and morphological operations.
    
    Parameters
    ----------
    image : np.ndarray
        Input image in 3-channel BGR format.
    kernel_size : int
        Size of the kernel for morphological operations. Default is 10.
    threshold : int
        Pixel value threshold for binary segmentation. Default is 100.
    pad : float
        Percentage of padding to add around the top, left, and right sides of the cropped image. Default is 0.05 (5%). Bottom padding is controlled by `bottom_pad`.
    bottom_pad : float
        Percentage of padding to add to the bottom of the cropped image. Default is 0.35 (35%). Top, left, and right padding is controlled by `pad`.

    Returns
    -------
    crop_pad : np.ndarray
        Cropped and padded image in 3-channel BGR format.       
    """

    # Binary threshold
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray_image, threshold, 255, cv.THRESH_BINARY)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Morphological operation for fine-tuning mask imperfections
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
    
    # Find contours of the scales in the image
    contours, hierarchy = cv.findContours(closing, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # If there are more than one scale in the image, use the largest one based on area
    if(len(contours) > 0):
        maxidx = 0
        maxarea = 0
        for i in range(len(contours)):
            contour = contours[i]
            if cv.contourArea(contour) > maxarea:
                maxarea = cv.contourArea(contour)
                maxidx = i
        x, y, w, h = cv.boundingRect(contours[maxidx])
        
    # Crop and Pad
    xmargin = pad
    ymargin_top = pad
    ymargin_bot = bottom_pad
    y_max, x_max, c = image.shape
    x_new = int(max(0, x-w*xmargin))
    y_new = int(max(0, y-h*ymargin_top))
    w_new = int(min(w*(1+xmargin*2), x_max-x_new))
    h_new = int(min(h*(1+ymargin_top+ymargin_bot), y_max-y_new))
    crop = image[y_new:y_new+h_new, x_new:x_new+w_new, :]
    
    # Make the crop square by padding the shorter dimension
    c_h, c_w, c = crop.shape
    left, right, top, bottom = 0, 0, 0, 0
    if(c_h > c_w):
        difference = c_h-c_w
        left = int(difference/2)
        right = int(difference/2)
    if(c_h < c_w):
        difference = c_w-c_h
        top = int(difference/2)
        bottom = int(difference/2)
    crop_pad = cv.copyMakeBorder(crop, top, bottom, left, right, cv.BORDER_REPLICATE)

    return crop_pad

def crop_and_pad_sam(image, down_scale, sam_type, sam_model_path, num_points, threshold, pad, bottom_pad):
    """Crop and pad an image using a Segment Anything Model (SAM).

    Parameters
    ----------
    image : np.ndarray
        Input image in 3-channel BGR format.
    down_scale : float
        Downscale factor for input image when using SAM segmentation. Default is 0.5 (50%).
    sam_type : str
        Type of SAM model to use. Options are "vit_h" (ViT-Huge), "vit_l" (ViT-Large), or "vit_b" (ViT-Base).
    sam_model_path : str
        Path to SAM model weights. If empty, will use default weights for the specified `sam_type`.
    num_points : int
        Number of points per side to use for SAM mask generation. Higher values yield more masks but take longer. Default is 8.
    threshold : float
        Stability score threshold for SAM mask generation. Higher values yield fewer, more stable masks. Default is 0.88.
    pad : float
        Percentage of padding to add around the top, left, and right sides of the cropped image. Default is 0.05 (5%). Bottom padding is controlled by `bottom_pad`.
    bottom_pad : float
        Percentage of padding to add to the bottom of the cropped image. Default is 0.35 (35%). Top, left, and right padding is controlled by `pad`.
    
    Returns
    -------
    crop_pad : np.ndarray
        Cropped and padded image in 3-channel BGR format.
    """

    # Generate masks using SAM and GPU, if available, and convert to gray scale
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[sam_type](checkpoint=sam_model_path)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=num_points,
        stability_score_thresh=threshold,
    )
    resized_image = cv.resize(image, (0,0), fx=down_scale, fy=down_scale)
    masks = mask_generator.generate(resized_image)
    # If no masks are found, return the original image
    if(len(masks) == 0):
        return image
    combined_masks = combine_masks(annotations=masks)
    if(len(combined_masks) == 0):
        return image
    mask = choose_best(annotations=combined_masks)
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Crop and Pad
    xmargin = pad
    ymargin_top = pad
    ymargin_bot = bottom_pad
    y_max, x_max = gray_image.shape
    x,y,w,h = mask['bbox']
    x_new = int(max(0, x-w*xmargin)/down_scale)
    y_new = int(max(0, y-h*ymargin_top)/down_scale)
    w_new = int(min(w*(1+xmargin*2)/down_scale, x_max-x_new))
    h_new = int(min(h*(1+ymargin_top+ymargin_bot)/down_scale, y_max-y_new))
    crop = image[y_new:y_new+h_new, x_new:x_new+w_new, :]
    
    # Make the crop square by padding the shorter dimension
    c_h, c_w, c = crop.shape
    left, right, top, bottom = 0,0,0,0
    if(c_h>c_w):
        difference = c_h-c_w
        left = int(difference/2)
        right = int(difference/2)
    if(c_h<c_w):
        difference = c_w-c_h
        top = int(difference/2)
        bottom = int(difference/2)
    crop_pad = cv.copyMakeBorder(crop, top, bottom, left, right, cv.BORDER_REPLICATE)

    return crop_pad

def preprocess_folder(image_dir, output_dir, seg_opt="binary", extension=".tif", out_type=".jpg", kernel_size=10, binary_threshold=100, pad=0.05, bottom_pad=0.35, normalization="none", invert=False, down_scale=0.5, sam_type="vit_h", sam_model_path="", num_points=8, sam_threshold=0.88):
    """Preprocess all images in a folder and save to output folder.
    
    Parameters
    ----------
    image_dir : str
        Full path to directory containing images to process
    output_dir : str
        Full path to directory to save cropped images
    seg_opt : str
        Method of segmentation to use: "binary" for simple binary (two-toned) segmentation, "sam" to use a segment anything model (SAM), or "none" for no segmentation.
    extension : str
        File extension of input images. Default is ".tif".
    out_type : str
        File extension of output cropped images. Default is ".jpg".
    kernel_size : int
        Kernel size for morphological operations in binary segmentation. Default is 10.
    binary_threshold : int
        For binary segmentation, the pixel value that defines the mask. Default is 100.
    pad : float
        Percentage of padding to add around the top, left, and right sides of the cropped image. Default is 0.05 (5%). Bottom padding is controlled by `bottom_pad`.
    bottom_pad : float
        Percentage of padding to add to the bottom of the cropped image. Default is 0.35 (35%). Top, left, and right padding is controlled by `pad`.
    normalization : str
        Method of normalization method to apply to cropped image. Options are "none" for no normalization, "he" for simple histogram equalization, or "clahe" for contrast limited adaptive histogram equalization. Default is "none".
    invert : bool
        Whether to invert the grayscale image before applying normalization. This will make dark regions light and light regions dark. Default is False.
    down_scale : float
        Downscale factor for input image when using SAM segmentation. Default is 0.5 (50%).
    sam_type : str
        Type of SAM model to use. Options are "vit_h" (ViT-Huge), "vit_l" (ViT-Large), or "vit_b" (ViT-Base). Default is "vit_h".
    sam_model_path : str
        Path to SAM model weights. If empty, will use default weights for the specified `sam_type`. Default is "".
    num_points : int
        Number of points per side to use for SAM mask generation. Higher values yield more masks but take longer. Default is 8.
    sam_threshold : float
        Stability score threshold for SAM mask generation. Higher values yield fewer, more stable masks. Default is 0.88.
    """

    # Create output directory if it doesn't exist
    if(not os.path.exists(output_dir)):
        os.mkdir(output_dir)

    # Loop through all images in the image directory
    for file in tqdm(os.listdir(image_dir), desc="Processing images"):
        if file.endswith(extension):
            # Read in the original image
            image = cv.imread(os.path.join(image_dir, file))

            # Crop and pad the image, applying segmentation if specified
            if(seg_opt == "binary"):
                cropped_image = crop_and_pad_binary_threshold(image=image, kernel_size=kernel_size, threshold=binary_threshold, pad=pad, bottom_pad=bottom_pad)
            elif(seg_opt == "sam"):
                cropped_image = crop_and_pad_sam(image=image, down_scale=down_scale, sam_type=sam_type, sam_model_path=sam_model_path, num_points=num_points, threshold=sam_threshold, pad=pad, bottom_pad=bottom_pad)
            else:
                cropped_image = image

            # Apply normalization and color inversion to the cropped image, if specified.
            # Image is converted to grayscale, cropped if desired, normalized, then converted back to BGR for tensorflow.
            if(normalization == "he"):
                cropped_image = cv.cvtColor(cropped_image, cv.COLOR_BGR2GRAY)
                if(invert):
                    cropped_image = 255-cropped_image
                cropped_image = cv.equalizeHist(cropped_image)
                cropped_image = cv.cvtColor(cropped_image, cv.COLOR_GRAY2BGR)
            elif(normalization =="clahe"):
                cropped_image = cv.cvtColor(cropped_image, cv.COLOR_BGR2GRAY)
                if(invert):
                    cropped_image = 255-cropped_image
                clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                cropped_image = clahe.apply(cropped_image)
                cropped_image = cv.cvtColor(cropped_image, cv.COLOR_GRAY2BGR)

            # Write the new cropped image to the output directory
            cv.imwrite(os.path.join(output_dir, os.path.splitext(file)[0]+out_type), cropped_image)

# Parse command line arguments. Currently only requires a path to a configuration yaml file.
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config_path", help="path to configuration yaml file")
args = parser.parse_args()

# Open the configuration file and read in parameters
try:
    config = load_yaml(file_path=args.config_path)
except FileNotFoundError:
    raise FileNotFoundError(f"Error: The configuration file was not found at {args.config_path}")

# Set defaults for settings that can also be set in the YAML configuration file
CONFIG_DEFAULTS = {
    "binary_threshold": 100,
    "bottom_pad": 0.35,
    "downsample": 0.5,
    "input_type": ".tif",
    "invert": False,
    "normalization": "none",
    "output_type": ".jpg",
    "pad": 0.05,
    "points_per_side": 8,
    "sam_model_type": "vit_h",
    "sam_weights_path": "",
    "segment": "binary",
    "stability_score_thresh": 0.88,
}
# Merge default settings into configuration file
# (If a key exists in both dictionaries, the value from the second dictionary,
# `config`, replaces the value from the first dictionary, the default value.)
config = CONFIG_DEFAULTS | config

# Run the preprocessing function with parameters from the configuration file
preprocess_folder(
    image_dir=config["raw_image_path"],
    output_dir=config["processed_image_path"],
    seg_opt=config["segment"],
    extension=config["input_type"],
    out_type=config["output_type"],
    kernel_size=10,
    binary_threshold=config["binary_threshold"],
    pad=config["pad"],
    bottom_pad=config["bottom_pad"],
    normalization=config["normalization"],
    invert=config["invert"],
    down_scale=config["downsample"],
    sam_type=config["sam_model_type"],
    sam_model_path=config["sam_weights_path"],
    num_points=config["points_per_side"],
    sam_threshold=config["stability_score_thresh"]
)
