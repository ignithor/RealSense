# RealSense


## Setup Instructions

Tested on Ubuntu 20.04 and Windows 11
```bash
conda create -n realsense python=3.9
conda activate realsense
```

```bash
pip install open3d opencv-python numpy
pip install ultralytics transformers torch
pip install pillow
```

if Linux (Ubuntu 20.04):
```bash
conda install -c conda-forge librealsense=2.50.0
```

if Windows:
```powershell
pip install pyrealsense2==2.50.0.3812
```


