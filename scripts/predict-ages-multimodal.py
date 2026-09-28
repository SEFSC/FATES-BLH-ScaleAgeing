#!/usr/bin/env python

"""
Predict Menhaden Ages (from images and metadata)
------------------------------------------------
This script predicts the age of Menhaden fish samples using scale images and
associated metadata (fish length, weight, and month of catch). The model's
architecture and settings are controlled via a configurations.yml file.
Predicted ages are written to a CSV file.

Usage:
    python predict-ages-multimodal.py --config_path path/to/configurations.yml
    python predict-ages-multimodal.py -c path/to/configurations.yml

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
import yaml

import numpy as np
import pandas as pd
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from tqdm import tqdm
from typing import Any, Callable, List, Optional, Type, Union

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
        'collection_date_colname', 'fish_id_colname', 'fish_length_colname',
        'fish_weight_colname', 'metadata_csv_file', 'output_csv_file', 'processed_image_path'
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

    # Check for file names included in config paths where needed
    if config["model_pth_file"].suffix.lower() != ".pth":
        raise ValueError("The 'model_pth_file' key in the configuration file must include a file name ending with '.pth'.")
    if config["output_csv_file"].suffix.lower() != ".csv":
        raise ValueError("The 'output_csv_file' key in the configuration file must include a file name ending with '.csv'.")
    if config["metadata_csv_file"].suffix.lower() != ".csv":
        raise ValueError("The 'metadata_csv_file' key in the configuration file must include a file name ending with '.csv'.")

# Function to create a 3x3 convolutional layer
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """Creates a 3x3 convolutional layer with padding. See torch.nn.Conv2d docs for more details.

    Attributes
    ----------
    in_planes : int
        Number of input channels.
    out_planes : int
        Number of output channels.
    stride : int, optional
        Stride value for the convolution. Default is 1.
    groups : int, optional
        Number of groups for grouped convolution. Default is 1.
    dilation : int, optional
        Dilation rate for the convolution. Default is 1.
    
    Returns
    -------
    nn.Conv2d
        A 3x3 convolutional layer.
    """
    # Create a 3x3 convolutional layer with the specified parameters
    conv_layer = nn.Conv2d(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )
    return conv_layer

# Function to create a 1x1 convolutional layer
def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """Creates a 1x1 convolutional layer with padding. See torch.nn.Conv2d docs for more details.

    Attributes
    ----------
    in_planes : int
        Number of input channels.
    out_planes : int
        Number of output channels.
    stride : int, optional
        Stride value for the convolution. Default is 1.
    
    Returns
    -------
    nn.Conv2d
        A 1x1 convolutional layer.
    """
    # Create a 1x1 convolutional layer with the specified parameters
    conv_layer = nn.Conv2d(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
    )
    return conv_layer

# Module to define a BasicBlock residual block for the ResNet model
class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        """Basic residual block implementation used in ResNet.

        Attributes
        ----------
        in_planes : int
            Number of input channels.
        planes : int
            Number of layer output channels.
        stride : int, optional
            Stride value for the convolutional layers. Default is 1.
        downsample : Optional[nn.Module], optional
            Downsampling layer to match dimensions. Default is None.
        groups : int, optional
            Number of groups for grouped convolution. Default is 1.
        base_width : int, optional
            Base width for grouped convolution. Default is 64.
        dilation : int, optional
            Dilation rate for dilated convolution. Default is 1.
        norm_layer : Optional[Callable[..., nn.Module]], optional
            Normalization layer to use. Default is None, which uses nn.BatchNorm2d.
        """
        super(BasicBlock, self).__init__()

        # Use batch normalization if no other normalization layer is provided
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # BasicBlock only supports groups=1 and base_width=64
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")

        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Create model layers
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        # 3x3 convolutional layer
        self.conv1 = conv3x3(in_planes=in_planes, out_planes=planes, stride=stride)
        # Batch normalization
        self.bn1 = norm_layer(planes)
        # ReLU activation function
        self.relu = nn.ReLU(inplace=True)
        # 3x3 convolutional layer
        self.conv2 = conv3x3(in_planes=planes, out_planes=planes)
        # Batch normalization
        self.bn2 = norm_layer(planes)
        # Downsample module
        self.downsample = downsample
        # Stride value for the convolutional layers
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        """Feed an example through the model.
        
        Attributes
        ----------
        x : Tensor
            Example in the form of a tensor to pass through the block.
        
        Returns
        -------
        Tensor
            The output prediction.
        """
        identity = x

        # First convolution layer
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Second convolution layer
        out = self.conv2(out)
        out = self.bn2(out)

        # Downsample the input, if needed
        if self.downsample is not None:
            identity = self.downsample(x)

        # Add the identity (skip connection) and apply ReLU activation
        out += identity
        out = self.relu(out)

        return out

# Module to define a Bottleneck residual block for the ResNet model
class Bottleneck(nn.Module):
    """Bottleneck in torchvision places the stride for downsampling at 3x3
    convolution (self.conv2) while the original implementation places the
    stride at the first 1x1 convolution (self.conv1) according to "Deep
    residual learning for image recognition" (https://arxiv.org/abs/1512.03385).
    This variant is also known as ResNet V1.5 and improves accuracy according
    to https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
    """
    expansion: int = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        """Bottleneck residual block implementation used in ResNet.

        Attributes
        ----------
        in_planes : int
            Number of input channels.
        planes : int
            Number of layer output channels.
        stride : int, optional
            Stride of the cross-correlation. Default is 1.
        downsample : Optional[nn.Module], optional
            Downsampling layer to match dimensions. Default is None.
        groups : int, optional
            Number of blocked connections from input channels to output channels. Default is 1.
        base_width : int, optional
            Base width for the bottleneck layer. Default is 64.
        dilation : int, optional
            Spacing between kernel elements. Default is 1.
        norm_layer : Optional[Callable[..., nn.Module]], optional
            Normalization layer to use. Default is None, which uses nn.BatchNorm2d.
        """
        super(Bottleneck, self).__init__()

        # Use batch normalization if no other normalization layer is provided
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # Adjust planes based on groups and base_width
        width = int(planes * (base_width / 64.0)) * groups

        # Create model layers
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        # 1x1 convolutional layer
        self.conv1 = conv1x1(in_planes=in_planes, out_planes=width)
        # Batch normalization
        self.bn1 = norm_layer(width)
        # 3x3 convolutional layer
        self.conv2 = conv3x3(in_planes=width, out_planes=width, stride=stride, groups=groups, dilation=dilation)
        # Batch normalization
        self.bn2 = norm_layer(width)
        # 1x1 convolutional layer
        self.conv3 = conv1x1(in_planes=width, out_planes=planes * self.expansion)
        # Batch normalization
        self.bn3 = norm_layer(planes * self.expansion)
        # ReLU activation function
        self.relu = nn.ReLU(inplace=True)
        # Downsample module
        self.downsample = downsample
        # Stride value for the convolutional layers
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        """Feed an example through the model.
        
        Attributes
        ----------
        x : Tensor
            Example in the form of a tensor to pass through the block.
        
        Returns
        -------
        Tensor
            The output prediction.
        """
        identity = x

        # First 1x1 convolution layer
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Second 3x3 convolution layer
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # Third 1x1 convolution layer
        out = self.conv3(out)
        out = self.bn3(out)

        # Downsample layer, if specified
        if self.downsample is not None:
            identity = self.downsample(x)

        # Add the identity (skip connection) and apply ReLU activation
        out += identity
        out = self.relu(out)

        return out

# Module defining modified RESNET backbone for multimodal data to include a metadata input branch
class ResNet(nn.Module):
    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            num_classes: int = 5,
            img_size: int = 64,
            metadata_size: int = 32,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
        ) -> None:
        """ResNet model adapted for multimodal data (images plus metadata).

        Attributes
        ----------
        block : Type[Union[BasicBlock, Bottleneck]]
            Type of residual block to use (BasicBlock or Bottleneck).
        layers : List[int]
            List specifying the number of blocks in each layer.
        num_classes : int, optional
            Number of output classes. Default is 5 (ages 0, 1, 2, 3, 4+).
        img_size : int, optional
            Size of the image feature vector after processing. Default is 64.
        metadata_size : int, optional
            Size of the metadata feature vector after processing. Default is 32.
        zero_init_residual : bool, optional
            If True, initializes the last batch norm in each residual branch to zero. Default is False.
        groups : int, optional
            Number of groups for grouped convolution. Default is 1.
        width_per_group : int, optional
            Base width for the bottleneck layer. Default is 64.
        replace_stride_with_dilation : Optional[List[bool]], optional
            List indicating whether to replace stride with dilation in each layer. Default is None.
        norm_layer : Optional[Callable[..., nn.Module]], optional
            Normalization layer to use. Default is None, which uses nn.BatchNorm2d.
        """
        super(ResNet, self).__init__()

        # Use batch normalization if no other normalization layer is provided
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        # Set initial number of input channels and dilation
        self.inplanes = 64
        self.dilation = 1
        
        # Image and metadata feature sizes
        self.img_size = img_size
        self.metadata_size = metadata_size

        # If replace_stride_with_dilation is not provided, set it to a default value of [False, False, False]
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        # If it is provided, ensure it has exactly three elements
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )

        # Set groups and base width for convolutions
        self.groups = groups
        self.base_width = width_per_group

        # Initial convolutional layer, batch norm, ReLU, and max pooling
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Four residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        # Adaptive average pooling and fully connected layers for image and metadata branches
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_meta = nn.Linear(3, metadata_size)
        self.fc_img = nn.Linear(512 * block.expansion, img_size)
        self.fc_combined = nn.Linear(metadata_size + img_size, num_classes)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(p=0.5)
        self.soft = nn.Softmax(dim = 1)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch, so that the
        # residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to
        # https://arxiv.org/abs/1706.02677.
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        """Creates a layer of residual blocks.
        
        Attributes
        ----------
        block : Type[Union[BasicBlock, Bottleneck]]
            Type of residual block to use (BasicBlock or Bottleneck).
        planes : int
            Number of layer output channels.
        blocks : int
            Number of blocks to create in the layer.
        stride : int, optional
            Stride value for the convolutional layers. Default is 1.
        dilate : bool, optional
            If True, applies dilation to the convolutional layers. Default is False.
        
        Returns
        -------
        nn.Sequential
            A sequential container of the created residual blocks.
        """

        # Get the normalization layer
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        # Adust dilation and stride if dilate is True
        if dilate:
            self.dilation *= stride
            stride = 1

        # Create downsample layer if stride is not 1 or number of input channels is different from output channels
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        # Create the specified number of blocks
        layers = []
        # Add the first block of the layer with potential downsampling
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion

        # Add the remaining blocks of the layer
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor, metadata: Tensor) -> Tensor:
        """Forward pass of the ResNet model.

        Attributes
        ----------
        x : Tensor
            Input image tensor.
        metadata : Tensor
            Metadata tensor.
        
        Returns
        -------
        Tensor
            Output tensor.
        """
        metadata = F.relu(self.fc_meta(metadata))

        # First convolutional layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Four residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Average pooling and fully connected layer
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc_img(x)

        # Combine image and metadata features
        combined_features = torch.cat((x, metadata), dim=1)
        combined_features = self.dropout(combined_features)
        x = self.fc_combined(combined_features)
        return x

    def forward(self, x: Tensor, metadata: Tensor) -> Tensor:
        """Forward pass of the ResNet model.

        Attributes
        ----------
        x : Tensor
            Input image tensor.
        metadata : Tensor
            Metadata tensor.
        
        Returns
        -------
        Tensor
            Output tensor.
        """
        return self._forward_impl(x, metadata)

# Function to create a new ResNet model
def resnet_merge(block: Type[Union[BasicBlock, Bottleneck]],
               layers: List[int],
               pretrained: bool = False,
               num_classes: int = 5,
               metadata_size: int = 32,
               img_size: int = 64,
               progress: bool = True,
               **kwargs: Any) -> ResNet:
    """
    Creates a new ResNet model adapted for multimodal data.

    Attributes
    ----------
    block : Type[Union[BasicBlock, Bottleneck]]
        Type of residual block to use (BasicBlock or Bottleneck).
    layers : List[int]
        List specifying the number of blocks in each layer of the network.
    pretrained : bool, optional
        If True, returns a model pre-trained on ImageNet. Default is False.
    num_classes : int, optional
        Number of output classes. Default is 5 (ages 0, 1, 2, 3, 4+).
    metadata_size : int, optional
        Size of the metadata feature vector after processing. Default is 32.
    img_size : int, optional
        Size of the image feature vector after processing. Default is 64.
    progress : bool, optional
        If True, displays a progress bar when downloading pretrained weights.
        Default is True.
    **kwargs : Any
        Additional keyword arguments for the ResNet class.
    
    Returns
    -------
    ResNet
        A ResNet model adapted for multimodal data.
    """
    # Download pretrained weights, if desired
    if pretrained:
        # Load a pretrained ResNet18 model
        model = resnet.resnet18(pretrained=True, progress=progress)
        # Update the final fully connected layer for the desired number of classes
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    # Update the kwargs dictionary with the specified parameters
    kwargs['metadata_size'] = metadata_size
    kwargs['img_size'] = img_size
    kwargs['block'] = block
    kwargs['layers'] = layers

    # Create a new ResNet model with modified parameters
    model = ResNet(**kwargs)
    return model

# Custom Dataset for loading fish scale images and metadata for inference
class FishTestDataset(Dataset):
    """Custom Dataset for loading fish scale images and metadata.
    
    Attributes
    ----------
    image_dir : str
        Path to the directory containing images.
    csv_path : str
        Path to the CSV file with age and metadata information.
    image_name : list
        List of image filenames.
    length : numpy.ndarray
        Array of fish lengths.
    wt : numpy.ndarray
        Array of fish weights.
    month : numpy.ndarray
        Array of collection months.
    transforms : callable, optional
        A function/transform that takes in a PIL image and returns a transformed version.
    """
    def __init__(self, image_dir, csv_path, csv_cols, file_extension, transform=None):
        """
        Attributes
        ----------
        image_dir : str
            Path to the directory containing images.
        csv_path : str
            Path to the CSV file with age and metadata information.
        csv_cols : dict
            Dictionary with keys ['fish_id_colname', 'fish_length_colname',
            'fish_weight_colname', 'collection_date_colname'] mapped to actual
            column names in the file passed to `csv_path`.
        file_extension: str
            The expected file extension for images (e.g., ".jpg").
        transform : callable, optional
            A function/transform for image transformations.
        """
        # Read the metadata CSV file, store the image dataset directory, and store the transformation methods 
        date_col = csv_cols['collection_date_colname']
        data_info = pd.read_csv(csv_path, header=0,
            usecols=list(csv_cols.values()),
            encoding="iso-8859-1").dropna(axis=0, how='all')
        try:
            # If it's the current format, this will succeed
            data_info['Collection Month'] = pd.to_datetime(
                data_info[date_col], 
                format='%d-%b-%y'
            ).dt.month
        except ValueError:
            # If it throws a ValueError, it's the legacy CSV format
            data_info['Collection Month'] = data_info[date_col]
        
        self.data_info = data_info.astype('int64')
        self.image_dir = image_dir
        self.transforms = transform

        # Append the file extension to the image names from the CSV
        # self.image_name = np.asarray([f"{name}{file_extension}" if file_extension not in str(name) else str(name) for name in self.data_info.loc[:, 'Fish nbr']])
        self.image_name = np.asarray([f"{str(nbr)}{file_extension}" for nbr in self.data_info.loc[:, csv_cols['fish_id_colname']]])

        # Check for metadata but missing image
        available_images = [f for f in os.listdir(self.image_dir) if f.endswith(file_extension)]
        missing_images = [x for x in self.image_name if x not in set(available_images)]
        if len(missing_images) > 1:
            raise FileNotFoundError(f"The following files appear in the metadata but not in `processed_image_path`: {missing_images}")

        # Check for image but missing metadata
        missing_metadata = [x for x in available_images if x not in set(self.image_name)]
        if len(missing_metadata) > 1:
            raise AssertionError(f"The following files found in `processed_image_path` are missing metadata and cannot be aged: {missing_metadata}")

        # Extract metadata attributes: fish length, weight, month of catch
        self.length = np.asarray(self.data_info.loc[:, csv_cols['fish_length_colname']])
        self.wt = np.asarray(self.data_info.loc[:, csv_cols['fish_weight_colname']])
        self.month = np.asarray(self.data_info.loc[:, 'Collection Month'])

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.image_name)

    def __getitem__(self, index):
        """Returns the image, metadata, and label at the specified index."""
        # Open the specified image
        img_path = os.path.join(self.image_dir, str(self.image_name[index]))
        image = Image.open(img_path)
        
        # Normalize metadata
        metadata = torch.tensor([(self.wt[index] - 163)/(82), (self.length[index] - 211)/ (35.5), (self.month[index]-7.4)/(1.9)]).type(torch.FloatTensor)
        
        # Transform the image, if transforms are provided
        if self.transforms:
            image = self.transforms(image)

        return (image, metadata), self.image_name[index]

def main():
    """Main function to run the inference script."""
    # Parse command line arguments.
    parser = argparse.ArgumentParser(description="Predict fish age using a pre-trained multimodal model.")
    parser.add_argument("-c", "--config_path", help="Path to configuration yaml file", required=True)
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
        "model_pth_file": str(script_dir / 'weights' / 'multimodal-model-v2025.pth'),
    }
    # Merge default settings into configuration file
    # (If a key exists in both dictionaries, the value from the second dictionary,
    # `config`, replaces the value from the first dictionary, the default value.)
    config = CONFIG_DEFAULTS | config

    # Image transformations: resizing, cropping, normalization
    data_transforms = transforms.Compose(
        [
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Load the dataset for inference
    csv_cols = [
        'fish_id_colname',
        'fish_length_colname',
        'fish_weight_colname',
        'collection_date_colname'
        ]
    test_dataset = FishTestDataset(
        image_dir=config["processed_image_path"],
        csv_path=config["metadata_csv_file"],
        csv_cols={k:config[k] for k in csv_cols if k in config},
        file_extension=config["output_type"],
        transform=data_transforms
    )
    test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False, drop_last=False)
    
    # Load the model using GPU, if available, in evaluation mode.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Create the multimodal ResNet model
    model = resnet_merge(
        BasicBlock,
        [2, 2, 2, 2],
        num_classes=5,
        metadata_size=32,
        img_size=64,
    )

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
    try:
        with open(config["output_csv_file"], 'w') as file:
            file.write("Image Name, Predicted Age\n")

            # Loop through the dataset and make predictions
            for (images, meta), img_path in tqdm(test_loader, desc="Predicting ages"):
                images = images.to(device)
                meta = meta.to(device)

                with torch.no_grad():
                    outputs = model(images, meta)
                
                _, preds = torch.max(outputs, 1)
                preds = preds.cpu().detach().numpy()
                
                # Write predictions to the output file
                for i in range(preds.shape[0]):
                    age = str(preds[i])
                    # Change the maximum age class to "4+"
                    if preds[i] == 4:
                        age = "4+"
                    file.write(f"{img_path[i]},{age}\n")
        print(f'Inference complete. Results saved to {config["output_csv_file"]}')

    except Exception as e:
        print(f"An error occurred during inference: {e}")

if __name__ == '__main__':
    main()
