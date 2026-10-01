#!/usr/bin/env python

"""
Predict Menhaden Ages (from image only)
---------------------------------------
This script predicts the age of fish scale images using a pre-trained ResNet18
model. User settings are included in a configs.yml file. Predicted ages are
written to a CSV file.

Usage:
    python predict-ages-images.py --config_path path/to/configurations.yml
    python predict-ages-images.py -c path/to/configurations.yml

Authors: aotian.zheng@noaa.gov (model development, training, validation, testing)
         and matt.grossi@noaa.gov (model testing, implementation, code
         refactoring for user functionality, documentation) with assistance
         from Google Gemini Coding Partner
Version: 2026.1.2
Release Date: September 2025
Last Updated: September 2026
"""

import argparse
import difflib
import os
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset  # For custom datasets
from torchvision import transforms
from torchvision.models import resnet18
from tqdm import tqdm
import yaml

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
        'output_csv_file', 'processed_image_path'
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
    for k, v in config.items():
        if 'path' in k or 'file' in k:
            if isinstance(v, str):
                # Strip accidental literal quotes/spaces and convert backslashes to forward slashes
                clean_path = v.strip(" \t\"'").replace('\\', '/')
                config[k] = Path(clean_path).expanduser()
            else:
                # Fallback if the path is already a Path object or unexpected type
                config[k] = Path(v)
     
    # Check and fix image type file extensions, if necessary
    if 'input_type' in config and not config['input_type'].startswith('.'):
        config['input_type'] = '.' + config['input_type']
    if 'output_type' in config and not config['output_type'].startswith('.'):
        config['output_type'] = '.' + config['output_type']

    # Check for file names included in config paths where needed
    if "model_pth_file" in config and config["model_pth_file"].suffix.lower() != ".pth":
        raise ValueError("The 'model_pth_file' key in the configuration file must include a file name ending with '.pth'.")
    if "output_csv_file" in config and config["output_csv_file"].suffix.lower() != ".csv":
        raise ValueError("The 'output_csv_file' key in the configuration file must include a file name ending with '.csv'.")

class FishTestDataset(Dataset):
    """Custom Dataset for loading fish scale images for age inference.
    
    Attributes
    ----------
    image_dir : str
        Path to the directory containing images.
    image_name : list
        List of image filenames in the directory.
    transforms : callable, optional
        A function/transform that takes in a PIL image and returns a transformed
        version.
    
    Methods
    -------
    __len__ : returns the number of images in the dataset.
    __getitem__(index) : returns the image and its filename at the specified
        index.
    """
    def __init__(self, image_dir, transform=None):
        """
        Parameters
        ----------
        image_dir : str
            Path to the directory containing images.
        transform : callable, optional
            A function/transform that takes in a PIL image and returns a
            transformed version.
        """

        # Get the directory of the images to age
        self.image_dir = image_dir

        # Get the transform methods
        self.transforms = transform

        # Image Name
        self.image_name = [
            f.name for f in self.image_dir.iterdir() if f.is_file()
            ]

    def __len__(self):
        """Returns the number of images in the dataset."""
        return len(self.image_name)

    def __getitem__(self, index):
        """Returns the image, metadata, and label at the specified index."""
        img_path = self.image_dir / str(self.image_name[index])
        
        try:
            # Open and transform the specified image
            image = Image.open(img_path)
            if self.transforms:
                image = self.transforms(image)
        except Exception as e:
            # Print the exact file causing the crash before raising the error
            print(f"\n[!] CRASHED WHILE READING IMAGE: {img_path}")
            raise e

        return image, self.image_name[index]

def main():
    """Main function to execute the inference script.""" 
    # Parse command line arguments. Currently only requires a path to a configuration yaml file.
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", help="path to configuration yaml file")
    args = parser.parse_args()

    # PWD absolute path
    script_dir = Path(__file__).resolve().parent

    # Open the configuration file and read in the parameters
    try:
        config = load_yaml(file_path=args.config_path)
    except FileNotFoundError:
        print(f"Error: The configuration file was not found at {args.config_path}")
        return

    # Set defaults for settings that can also be set in the YAML configuration file
    CONFIG_DEFAULTS = {
        "model_pth_file": script_dir / 'weights' / 'image-model-v2025.pth',
        "output_type": ".jpg"
    }
    # Merge default settings into configuration file
    # (If a key exists in both dictionaries, the value from the second dictionary,
    # `config`, replaces the value from the first dictionary, the default value.)
    config = CONFIG_DEFAULTS | config

    # Image transformations: resizing, cropping, normalization
    data_transforms = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    test_dataset = FishTestDataset(
        image_dir=config["processed_image_path"],
        transform=data_transforms
    )
    test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False, drop_last=False)

    # Load the model using GPU, if available, in evaluation mode.
    # Number of classes corresponds to the number of age classes.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = resnet18(num_classes=5)

    # Load the pre-trained model weights
    try:
        model.load_state_dict(torch.load(config["model_pth_file"]))
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return
    model.eval()    
    model.to(device)

    # Create output file and write header
    with open(config["output_csv_file"], 'w') as file:
        file.write("Image Name, Predicted Age\n")

        # Loop through the dataset and make predictions
        for images, img_path in tqdm(test_loader, desc="Predicting ages"):
            images = images.to(device)
            outputs = model(images)
            outputs = torch.squeeze(outputs)
            _, preds = torch.max(outputs, 1)
            preds = preds.cpu().detach().numpy()
            for i in range(preds.shape[0]):
                age = str(preds[i])
                # Change the maximum age class to "4+"
                if(preds[i] == 4):
                    age = "4+"
                file.write("%s,%s\n" % (img_path[i], age))
    print(f'Inference complete. Results saved to {config["output_csv_file"]}')
    
if __name__ == "__main__":
    main()