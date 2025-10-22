# Dataset Setup Guide

This guide explains how to prepare and organize datasets for training and inference with this project.

## Data Format Overview

All data should be stored as **`.npy` files** (NumPy array format). This includes:
- **MRI volumes**: Intensity normalized 3D arrays
- **Segmentation masks**: Label maps with corresponding class values

## Directory Structure

### For Training
```
project_root/
├── data/
│   ├── mri_data/
│   │   ├── train/
│   │   │   ├── patient_001.npy
│   │   │   ├── patient_002.npy
│   │   │   └── ...
│   │   ├── val/
│   │   │   ├── patient_101.npy
│   │   │   └── ...
│   │   └── test/
│   │       └── ...
│   ├── segmentations/
│   │   ├── breast_npy/
│   │   │   ├── train/
│   │   │   │   ├── patient_001.npy
│   │   │   │   └── ...
│   │   │   └── val/
│   │   │       └── ...
│   │   └── dv_npy/  (FGT and blood vessel masks)
│   │       ├── train/
│   │       │   ├── patient_001.npy
│   │       │   └── ...
│   │       └── val/
│   │           └── ...
│   └── preds/  (Predictions from breast model used as input for DV model)
│       └── breast_model_preds/
│           ├── train/
│           │   └── ...
│           └── val/
│               └── ...
```

### For Prediction
```
project_root/
├── mri_volumes/
│   ├── volume_001.npy
│   └── ...
└── predicted_masks/  (Output directory)
    ├── volume_001_breast.npy
    └── ...
```

## Data Preparation Steps

### Step 1: Download Dataset
The original data comes from the **Duke-Breast-Cancer-MRI** dataset available at:
https://doi.org/10.7937/TCIA.e3sv-re93

Download the dataset with:
- MRI DICOM files (precontrast sequences)
- Annotation files (NRRD format)

### Step 2: Convert to NPY Format
Use the preprocessing functions in `preprocessing.py`:

```python
from preprocessing import *
import numpy as np
from pathlib import Path

# Read mapping file (available with dataset download)
fpath_mapping_df = clean_filepath_filename_mapping_csv(
    'Breast-Cancer-MRI-filepath_filename-mapping.csv'
)

# Process each subject
subject_id = 'Breast_MRI_001'
image_array, dcm_data, nrrd_breast_data, nrrd_dv_data = read_precontrast_mri_and_segmentation(
    subject_id,
    'Duke-Breast-Cancer-MRI',  # Path to downloaded dataset
    fpath_mapping_df,
    'train_annotations'  # Path to annotation directory
)

# Normalize images
image_array = zscore_image(normalize_image(image_array))

# Save as NPY
output_dir = Path('data/mri_data/train')
output_dir.mkdir(parents=True, exist_ok=True)

np.save(output_dir / f'{subject_id}.npy', image_array)
np.save(Path('data/segmentations/breast_npy/train') / f'{subject_id}.npy', nrrd_breast_data)
np.save(Path('data/segmentations/dv_npy/train') / f'{subject_id}.npy', nrrd_dv_data)
```

### Step 3: Train/Validation Split
Organize data into `train/` and `val/` directories (typically 80/20 split):

```python
import shutil
import numpy as np
from pathlib import Path

data_dir = Path('data/mri_data/train')
files = sorted(list(data_dir.glob('*.npy')))

# Split into 80/20
split_idx = int(len(files) * 0.8)
train_files = files[:split_idx]
val_files = files[split_idx:]

# Copy to appropriate directories
val_dir = Path('data/mri_data/val')
val_dir.mkdir(parents=True, exist_ok=True)

for file in val_files:
    shutil.move(str(file), str(val_dir / file.name))
```

## Data Specifications

### Image Array Properties
- **Shape**: 3D array (X, Y, Z) - e.g., (512, 512, 160)
- **Data Type**: float32
- **Value Range**: Should be intensity normalized (typically 0-1 or standardized)
- **Content**: Raw MRI voxel intensities

### Mask Array Properties
- **Shape**: Same as image array (X, Y, Z)
- **Data Type**: uint8 or int32
- **Value Range**: 
  - **Breast masks**: 0 (background), 1 (breast)
  - **DV masks**: 0 (background), 1 (FGT/dense tissue), 2 (blood vessels)

## Dataset Classes

The project provides several dataset classes in `dataset_3d.py`:

### 1. Dataset3DSimple
Uses full volume for each input (no patching).
```python
from dataset_3d import Dataset3DSimple

dataset = Dataset3DSimple(
    image_dir='data/mri_data/train',
    mask_dir='data/segmentations/breast_npy/train',
    transforms=None
)
```

### 2. Dataset3DRandom (Recommended for training)
Randomly samples subvolumes of fixed size from each volume.
```python
from dataset_3d import Dataset3DRandom

dataset = Dataset3DRandom(
    image_dir='data/mri_data/train',
    mask_dir='data/segmentations/breast_npy/train',
    input_dim=96,  # Cube size: 96x96x96
    total_samples=20000  # Total samples per epoch
)
```

### 3. Dataset3DDivided
Systematically divides volumes into overlapping patches.
```python
from dataset_3d import Dataset3DDivided

dataset = Dataset3DDivided(
    image_dir='data/mri_data/train',
    mask_dir='data/segmentations/breast_npy/train',
    input_dim=96,
    x_y_divisions=4,  # 4x4 grid in XY plane
    z_division=2      # 2 divisions in Z axis
)
```

### 4. Dataset3DVerticalStack
Stacks consecutive slices along Z axis.
```python
from dataset_3d import Dataset3DVerticalStack

dataset = Dataset3DVerticalStack(
    image_dir='data/mri_data/train',
    mask_dir='data/segmentations/breast_npy/train',
    z_input_dim=32,   # Stack 32 slices
    z_step_size=4     # Step by 4 slices
)
```

## For Prediction Only

If you just want to use pre-trained models for prediction:

1. **Prepare your MRI volumes as `.npy` files**
   ```python
   import numpy as np
   from preprocessing import normalize_image, zscore_image
   
   # Load and preprocess your MRI (e.g., from DICOM)
   mri_volume = ...  # Load your MRI data
   mri_volume = zscore_image(normalize_image(mri_volume))
   
   # Save as NPY
   np.save('mri_volumes/your_volume.npy', mri_volume)
   ```

2. **Run prediction** (no mask directory needed)
   ```bash
   python predict.py --target-tissue breast \
                     --image mri_volumes/your_volume.npy \
                     --save-masks-dir output_masks/ \
                     --model-save-path trained_models/breast_model.pth
   ```

## Example: Complete Training Workflow

```python
from dataset_3d import Dataset3DRandom
from train import train_model
from unet import UNet3D
import torchio as tio

# Configuration
input_dim = 96
total_train_samples = 20000
total_val_samples = 4000
batch_size = 16
epochs = 20
n_channels = 1  # Breast: 1 channel
n_classes = 1   # Breast: 1 class

# Create datasets
train_dataset = Dataset3DRandom(
    image_dir='data/mri_data/train',
    mask_dir='data/segmentations/breast_npy/train',
    input_dim=input_dim,
    total_samples=total_train_samples,
    transforms=tio.Compose([
        tio.Resize((input_dim, input_dim, input_dim)),
        tio.RandomBiasField(),
        tio.RandomMotion(degrees=5, translation=5, num_transforms=2),
    ]),
    one_hot_mask=True
)

val_dataset = Dataset3DRandom(
    image_dir='data/mri_data/val',
    mask_dir='data/segmentations/breast_npy/val',
    input_dim=input_dim,
    total_samples=total_val_samples,
    transforms=tio.Compose([
        tio.Resize((input_dim, input_dim, input_dim)),
    ]),
    one_hot_mask=True
)

# Create model
model = UNet3D(
    in_channels=n_channels,
    out_classes=n_classes,
    num_encoding_blocks=3,
    padding=True,
    normalization='batch'
)

# Train
trained_model = train_model(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    n_classes=n_classes,
    n_channels=n_channels,
    batch_size=batch_size,
    learning_rate=3e-4,
    epochs=epochs,
    model_save_dir='trained_models/',
    model_save_name='my_breast_model.pth'
)
```

## Notes

- **Filenames**: Must end with `.npy` (same filename for corresponding image and mask)
- **Directory organization**: Exactly as shown above for the code to find files
- **Normalization**: Images must be normalized before saving as NPY
- **Memory**: Full datasets are loaded into RAM, ensure sufficient memory
- **GPU**: Training benefits greatly from GPU acceleration
