# UNeXtplus
"UNeXt++: A parallel-cascade hybridized UNeXt for rapid medical image segmentation", ICPR 2024
# Project Overview

This repository contains a deep learning model for image segmentation, using several architectures including UNet, UNet++, UNeXt, and UNeXt++. The codebase includes scripts for data loading, model construction, training, and evaluation.

## Dataset Structure
Before getting started, please ensure that your dataset is organized as shown below to facilitate smooth model training.
```
<dataset_name>
├── images
│   ├── 0a7e06.jpg
│   ├── 0aab0a.jpg
│   ├── 0b1761.jpg
│   └── ...
└── masks
    ├── 0
    │   ├── 0a7e06.png
    │   ├── 0aab0a.png
    │   ├── 0b1761.png
    │   └── ...
    ├── 1
    │   ├── 0a7e06.png
    │   ├── 0aab0a.png
    │   ├── 0b1761.png
    │   └── ...
    └── ...
```

### Code File Descriptions

- **`dataset.py`**: Data loading and processing script.
- **`model.py`**: Contains the architecture for UNet, UNet++, UNeXt, and UNeXt++ models.
- **`train.py`**: Script to train the models.
- **`utils.py`**: Contains utility functions used during model training and evaluation.
- **`model/`**: Folder storing training logs.
- **`models/`**: Folder for saving model weights.
- **`outputs/`**: Folder for storing image segmentation results.

## How to Train the Model

Follow these steps to train the model with your own dataset:

1. **Set Dataset Paths**
    In `train.py`, update the following parameters to match the locations of your dataset:

   - `--images_path`: Path to the images directory.
   - `--masks_path`: Path to the masks directory.

2. **Select the Model**
    Change the model architecture by setting the `--method` argument:

   - `0`: UNet
   - `1`: UNet++
   - `2`: TransUNet
   - `3`: UNeXt
   - `4`: UNeXtPlus (default)

3. **Set Dataset Name**
    Update the `--dataset` argument with the name of your dataset (e.g., `DDTI`, `BUSI`, `ISIC2018`, `ISIC2020`).

4. **Set Image and Mask Extensions**
    Update the following parameters to match your file extensions:

   - `--img_ext`: Image file extension (default `.PNG`).
   - `--mask_ext`: Mask file extension (default `.PNG`).

5. **Save Model Weights**
    The trained model weights will be saved at:

   ```
   models/{dataset}/{model_name}/model.pth
   ```

   where `{dataset}` is the dataset name you provided, and `{model_name}` corresponds to the selected model architecture.

## How to Test the Model

To evaluate the trained model on a test set:

1. **Set Dataset Name**
    In `test.py`, update the `--dataset` argument with the name of the dataset used for training.
2. **Set Test Dataset Paths**
    Specify the paths for the test images and masks:
   - `--test_images`: Path to the test images directory.
   - `--test_masks`: Path to the test masks directory.
3. **Set Model Weights Path**
    Specify the location of the trained model weights:
   - `--yml_path`: Path to the models folder.
   - `--model_path`: Path to the saved model weights, e.g., `models/{dataset}/{model_name}/model.pth`.
