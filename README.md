# Converting Normal Photos to Spatial Photos

## Project Overview
This project implements an advanced system for converting 2D photos into spatial (3D) photos using state-of-the-art deep learning techniques. The system combines depth estimation and stereo image generation to create immersive 3D experiences from single images.

Key Features:
- Multiple output format support (SBS, HEIC, JPS, Depth Maps)
- Deep learning-based depth estimation
- High-quality stereo image generation
- Support for various devices and viewing methods
- Real-time processing capabilities
- Batch processing support
- Cross-platform compatibility

## Directory Structure
```
RVCE_24CX02RV_GenAI_Converting_normal_photos_to_spatial_photos/
├── different formats/                    # Format conversion modules
│   ├── sbs.py                          # Side-by-side stereo format
│   ├── stero_heic.swift                # HEIC stereo format
│   ├── convertjps.py                   # JPS format
│   └── depth.py                        # Depth map generator
│
├── model/                              # Deep learning implementation
│   ├── model.py                        # Main model
│   ├── weights/                        # Pre-trained weights
│   └── utils/                          # Utility functions
│
├── scripts/                            # Utility scripts
├── tests/                              # Test suite
├── requirements.txt                    # Dependencies
└── README.md                          # Documentation
```

## Model Architecture
![dbf3666e-fc68-40c1-a314-298faeb4d71b](https://github.ecodesamsung.com/SRIB-PRISM/RVCE_24CX02RV_GenAI_Converting_normal_photos_to_spatial_photos/assets/33436/28b61fef-96a4-433c-9159-d4b7173e247c)


### Core Components

1. **Encoder Network**
   - Input: Single RGB image (224x224x3)
   - Architecture: ResNet-50 backbone
   - Features: Multi-scale feature extraction
   - Output: 512-dimensional feature vector

2. **Depth Estimation Network**
   - Input: Encoded features
   - Architecture: U-Net with skip connections
   - Features: 
     - 5 encoder blocks
     - 5 decoder blocks
     - Skip connections for detail preservation
   - Output: Depth map (224x224x1)

3. **Stereo Generation Network**
   - Input: RGB image + Depth map
   - Architecture: Custom CNN
   - Features:
     - Left/right view generation
     - Parallax effect simulation
     - View consistency check
   - Output: Stereo image pair

### Training Process
- Pre-training on synthetic data
- Fine-tuning on real-world images
- Loss functions:
  - Depth loss (L1)
  - Stereo consistency loss
  - Edge-aware smoothness loss

## Requirements

### Software Requirements
- Python 3.8+
- PyTorch 1.9.0+
- OpenCV 4.5.0+
- NumPy 1.19.0+
- Swift (for HEIC support)

### Hardware Requirements
- CPU: Intel i5/AMD Ryzen 5+
- RAM: 8GB minimum (16GB recommended)
- GPU: NVIDIA GPU with CUDA support
- Storage: 10GB free space
- Display: 1920x1080 minimum resolution

### Dependencies
```
torch>=1.9.0
torchvision>=0.10.0
opencv-python>=4.5.0
numpy>=1.19.0
pillow>=8.0.0
tqdm>=4.62.0
```

## Installation and Setup

1. **Clone Repository**
```bash
git clone https://github.com/yourusername/RVCE_24CX02RV_GenAI_Converting_normal_photos_to_spatial_photos.git
cd RVCE_24CX02RV_GenAI_Converting_normal_photos_to_spatial_photos
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Download Models**
```bash
python scripts/download_models.py
```

5. **Verify Installation**
```bash
python test_setup.py
```

### Optional Setup
- CUDA toolkit for GPU acceleration
- FFmpeg for video processing
- ImageMagick for image manipulation

### Environment Variables
```bash
export CUDA_VISIBLE_DEVICES=0  # For GPU usage
export MODEL_PATH=/path/to/models
export OUTPUT_DIR=/path/to/output
```

Zip file of the full code
https://drive.google.com/drive/folders/1L1xd09eOWz4k5DQY4yRRJfD-2flHuopi?usp=sharing
All the 50 test images files are there in the output folder
