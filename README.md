# MEAF-YOLO

MEAF-YOLO is a trimmed Ultralytics-based object detection project. The repository keeps the MEAF-YOLO model structure, training code, validation code, and the required custom modules.

### 1. Install

Create a Python environment. Python 3.10 is recommended.

```bash
conda create -n meaf-yolo python=3.10 -y
conda activate meaf-yolo
```

Install PyTorch according to your CUDA version, then install the project requirements:

```bash
pip install -r requirements.txt
```

This project already contains a trimmed local `ultralytics` package, so installing the official `ultralytics` package is not required.

### 2. Usage

#### 2.1 Train

```bash
python train.py
```

The default training script uses:

```text
ultralytics/cfg/MEAF-YOLO/MEAF-YOLO.yaml
ultralytics/cfg/datasets/VisDrone.yaml
```

#### 2.2 Val

```bash
python val.py
```

The default validation script loads:

```text
runs/detect/train/weights/best.pt
```

Modify `val.py` if your weight path or dataset yaml is different.

### 3. Key Directory Structure

#### 3.1 Configuration Files (`ultralytics/cfg`)

Model architecture: located in `ultralytics/cfg/MEAF-YOLO`, containing the MEAF-YOLO network configuration.

Dataset settings: stored in `ultralytics/cfg/datasets`, currently keeping:

```text
VisDrone.yaml
AITOD.yaml
```

Training/testing setup: managed by `ultralytics/cfg/default.yaml`, including common hyperparameters, optimizer settings, and runtime options.

#### 3.2 Module Implementations (`ultralytics/nn/modules`)

Core network blocks are implemented in:

```text
ultralytics/nn/modules/block.py
ultralytics/nn/modules/conv.py
ultralytics/nn/modules/head.py
```

The main MEAF-YOLO custom modules include:

```text
TDC_C2f (paper: TDC-C2f)
  - TDCE
    - MRSE
    - SDM
    - FSDE
  - CDGR
PLKF
TD / BU (the top-down and bottom-up modules of CD-PAN)
AMSFHead (paper: AMSF-Head; composed of AMSFBlock modules)
```


#### 3.3 Model Parsing (`ultralytics/nn`)

Model construction and yaml parsing are handled by:

```text
ultralytics/nn/tasks.py
```

This file registers the modules used by `MEAF-YOLO.yaml`.

#### 3.4 Training and Validation Entrances

```text
train.py
val.py
```

`train.py` starts model training, and `val.py` evaluates trained weights.
