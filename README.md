# 🚶 YOLO Pedestrian Detection with CBAM Attention

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![YOLOv11](https://img.shields.io/badge/YOLOv11-Medium-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**A state-of-the-art pedestrian detection system using YOLOv11 enhanced with CBAM attention mechanisms**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Results](#-results) • [Architecture](#-architecture) • [Dataset](#-dataset)

</div>

---

## 📋 Overview

This project implements an advanced pedestrian detection system using **YOLOv11-Medium** as the base architecture, enhanced with **Convolutional Block Attention Module (CBAM)** for improved feature extraction and detection accuracy. The model is trained on the **Caltech Pedestrian Dataset**, one of the most challenging benchmarks for pedestrian detection.

### ✨ Key Highlights

- 🎯 **Enhanced YOLOv11**: Custom architecture with CBAM attention mechanisms
- 🔥 **CBAM Integration**: Channel and spatial attention for better feature representation
- 📊 **Comprehensive Training**: 50 epochs with batch size optimization
- 🎨 **Visualization Tools**: Advanced tracking and detection visualization
- 📈 **Performance Metrics**: Detailed precision, recall, mAP tracking
- 🚀 **Production Ready**: Optimized for real-time inference

---

## 🌟 Features

### Core Capabilities
- ✅ Real-time pedestrian detection in images and videos
- ✅ Custom YOLOv11-CBAM architecture with attention mechanisms
- ✅ Multi-scale feature extraction and fusion
- ✅ Comprehensive evaluation metrics (Precision, Recall, mAP50, mAP50-95)
- ✅ Confidence threshold filtering (≥0.5)
- ✅ Visualization of detection results with bounding boxes

### Technical Features
- **CBAM Attention Module**: Dual attention mechanism (channel + spatial)
- **Transfer Learning**: Leverages pretrained YOLOv11-Medium weights
- **Data Augmentation**: Built-in YOLO augmentation pipeline
- **Custom Dataset Pipeline**: Automated Caltech dataset processing
- **Performance Tracking**: Real-time metrics visualization with Plotly

---

## 🏗️ Architecture

### YOLOv11-CBAM Model

The model integrates CBAM attention modules into the YOLOv11-Medium backbone at strategic layers (layers 2 and 4) to enhance feature representation:

```
Input Image (640×480)
    ↓
YOLOv11 Backbone
    ↓
Layer 2 + CBAM → Enhanced Features
    ↓
Layer 4 + CBAM → Enhanced Features
    ↓
YOLOv11 Neck (PANet)
    ↓
YOLOv11 Head
    ↓
Detection Output
```

### CBAM Attention Mechanism

```python
class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    - Channel Attention: avg_pool + max_pool → FC layers → attention weights
    - Spatial Attention: channel-wise pooling → conv → attention map
    - Output: input_features × channel_attention × spatial_attention
```

**Benefits**:
- Focuses on informative features while suppressing irrelevant ones
- Improves detection of pedestrians at various scales
- Enhances performance in challenging scenarios (occlusion, small objects)

---

## 📊 Dataset

### Caltech Pedestrian Dataset

The model is trained on the **Caltech Pedestrian Detection Benchmark**, which contains:

- **Training Set**: Multiple video sequences from set00-set10
- **Validation Set**: Separate test sequences
- **Image Size**: 640×480 pixels
- **Annotations**: Bounding boxes in YOLO format (normalized xywh)
- **Class**: Single class - `person`

#### Dataset Structure
```
datasets/
├── images/
│   ├── train/
│   │   └── caltechpedestriandataset/
│   │       ├── set00/
│   │       ├── set01/
│   │       └── ...
│   └── val/
│       └── caltechpedestriandataset/
└── labels/
    ├── train/
    └── val/
```

#### Preprocessing Pipeline
1. **Annotation Conversion**: Matlab annotations → YOLO format
2. **Box Format**: `(x_center, y_center, width, height)` normalized to [0, 1]
3. **Filtering**: Remove occluded and partially visible pedestrians
4. **Frame Sampling**: Strategic frame selection from video sequences

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Setup

```bash
# Clone the repository
git clone https://github.com/Gemechu90/yolo-pedestrian-detection.git
cd yolo-pedestrian-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```txt
ultralytics>=8.0.0
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
plotly>=5.14.0
tqdm>=4.65.0
matplotlib>=3.7.0
```

---

## 💻 Usage

### 1. Prepare the Dataset

```python
# The notebook includes automated dataset preparation
# Converts Caltech annotations to YOLO format
python prepare_dataset.py
```

### 2. Train the Model

#### Standard YOLOv11-Medium
```python
from ultralytics import YOLO

model = YOLO("yolo11m.pt")
model.train(
    data="custom_dataset.yaml",
    epochs=50,
    batch=32,
    verbose=True
)
```

#### YOLOv11-CBAM (Enhanced)
```python
from custom_ultralytics.models import YOLO11m_CBAM

# Initialize custom model
cbam_model = YOLO11m_CBAM()

# Load pretrained weights
base_model = YOLO("yolo11m.pt")
cbam_model.load_pretrained(base_model)

# Train
cbam_model.train(
    data="custom_dataset.yaml",
    epochs=50,
    batch=32
)
```

### 3. Run Inference

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('runs/detect/train/weights/best.pt')

# Predict on images
results = model.predict(
    source='path/to/images',
    conf=0.5,  # Confidence threshold
    save=True   # Save results
)

# Predict on video
results = model.predict(
    source='path/to/video.mp4',
    conf=0.5,
    save=True
)
```

### 4. Evaluate Performance

```python
# Validation
metrics = model.val()

print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")
print(f"Precision: {metrics.box.p}")
print(f"Recall: {metrics.box.r}")
```

---

## 📈 Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Precision** | Tracked across 50 epochs |
| **Recall** | Tracked across 50 epochs |
| **mAP@0.5** | Tracked across 50 epochs |
| **mAP@0.5:0.95** | Tracked across 50 epochs |
| **Inference Speed** | Real-time capable |

### Training Configuration
- **Base Model**: YOLOv11-Medium
- **Epochs**: 50
- **Batch Size**: 32
- **Image Size**: 640×480 (native Caltech resolution)
- **Optimizer**: AdamW (default YOLO)
- **Learning Rate**: Auto-scheduled

### Visualization Examples

The notebook includes comprehensive visualization tools:
- ✅ Training/validation loss curves
- ✅ Precision-Recall curves
- ✅ mAP progression over epochs
- ✅ Detection results with bounding boxes
- ✅ Frame-by-frame tracking sequences

---

## 🔬 Model Architecture Details

### CBAM Module Implementation

```python
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
        # Spatial Attention
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )
```

### Integration Strategy

1. **Layer Selection**: CBAM modules added after layers 2 and 4
2. **Channel Reduction**: 16× reduction ratio for efficiency
3. **Activation**: Sigmoid for attention weights
4. **Sequential Integration**: Maintains gradient flow

---

## 📁 Project Structure

```
yolo-pedestrian-detection/
├── custom_ultralytics/
│   ├── nn/
│   │   └── modules/
│   │       └── cbam.py          # CBAM implementation
│   └── models/
│       └── yolo11m_cbam.py      # Custom YOLO model
├── datasets/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
├── runs/
│   └── detect/
│       └── train/
│           └── weights/
│               ├── best.pt      # Best model weights
│               └── last.pt      # Last checkpoint
├── notebooks/
│   └── yolo-pedestrian.ipynb    # Main training notebook
├── custom_dataset.yaml          # Dataset configuration
├── requirements.txt
└── README.md
```

---

## 🛠️ Advanced Features

### Custom Dataset Configuration

```yaml
# custom_dataset.yaml
path: /path/to/datasets
train: /path/to/datasets/images/train
val: /path/to/datasets/images/val

nc: 1  # number of classes
names:
  0: person
```

### Detection Pipeline

```python
def detect_people(frame_list):
    """
    Detects pedestrians in a list of frames
    
    Args:
        frame_list: List of image file paths
        
    Returns:
        all_box_list: List of bounding boxes per frame
        all_conf_list: List of confidence scores per frame
    """
    model = YOLO('runs/detect/train/weights/best.pt')
    results = model.predict(frame_list, verbose=False)
    
    all_boxes = []
    all_confs = []
    
    for result in results:
        boxes = result.boxes
        frame_boxes = []
        frame_confs = []
        
        for box in boxes:
            if box.conf >= 0.5:  # Confidence threshold
                frame_boxes.append(box.xyxy[0].cpu().numpy())
                frame_confs.append(float(box.conf))
        
        all_boxes.append(frame_boxes)
        all_confs.append(frame_confs)
    
    return all_boxes, all_confs
```

---

## 🎯 Use Cases

- **Autonomous Vehicles**: Pedestrian detection for self-driving cars
- **Surveillance Systems**: Crowd monitoring and tracking
- **Smart Cities**: Traffic analysis and pedestrian flow management
- **Retail Analytics**: Customer counting and behavior analysis
- **Safety Applications**: Construction site monitoring, crosswalk safety

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Add unit tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting PR

---

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@misc{yolo-pedestrian-cbam,
  author = {Your Name},
  title = {YOLO Pedestrian Detection with CBAM Attention},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/yolo-pedestrian-detection}
}
```

### References
- **YOLOv11**: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- **CBAM**: [Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018](https://arxiv.org/abs/1807.06521)
- **Caltech Dataset**: [Dollár et al., "Pedestrian Detection: An Evaluation of the State of the Art", PAMI 2012](https://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Ultralytics** for the excellent YOLOv11 implementation
- **Caltech Vision Lab** for the pedestrian detection benchmark
- **PyTorch Team** for the deep learning framework
- **CBAM Authors** for the attention mechanism architecture

---

## 📧 Contact

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)

---

## 🔄 Updates & Roadmap

### Current Version: 1.0.0

### Planned Features
- [ ] Multi-class pedestrian detection (walking, running, standing)
- [ ] Real-time video stream processing
- [ ] Model quantization for edge deployment
- [ ] Integration with ROS for robotics applications
- [ ] Web interface for easy inference
- [ ] TensorRT optimization for NVIDIA GPUs
- [ ] ONNX export for cross-platform deployment

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ by [Gemechu Geleta]

</div>
