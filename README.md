# 🔍 Computer Vision - Object Detection Portfolio

**A collection of real-world object detection projects using YOLO across multiple domains**

This repository showcases various computer vision applications leveraging YOLO (You Only Look Once) architecture for different use cases, from agricultural monitoring to workplace analytics and beyond.

---

## 📋 Projects

### 1. 🌽 LeafScan - Agricultural Disease Detection
AI-powered corn leaf disease detection system for early identification and classification of plant health issues, enabling farmers to take preventive action before significant crop damage.

**Key Features:**
- Real-time disease detection and classification
- Multi-class disease identification
- Confidence scoring for predictions
- Batch processing support

---

### 2. 💻 UrDesk - Workspace Analysis Tool
Computer vision system that monitors and analyzes workspace conditions, detecting objects on desk surfaces to assess ergonomics and organization levels.

**Key Features:**
- Real-time workspace monitoring
- Object detection and counting
- Workspace organization scoring
- Ergonomic assessment

---

### 🚧 More Projects Coming Soon
This repository is actively expanding with new object detection applications across various domains.

---

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/rioooranteai/YOLO-computer-vision-object-detection.git
cd YOLO-computer-vision-object-detection

# Install dependencies
pip install ultralytics opencv-python numpy pillow

# Navigate to specific project
cd LeafScan  # or cd Urdesk

# Run detection
python detect.py --source path/to/image.jpg
```

---

## 📁 Repository Structure

```
YOLO-computer-vision-object-detection/
│
├── LeafScan/                  # Corn leaf disease detection
├── Urdesk/                    # Workspace analysis tool
├── [Future Projects]/         # More projects to be added
│
└── README.md
```

Each project folder contains:
- Detection/inference scripts
- Trained model weights
- Dataset information
- Project-specific documentation

---

## 🛠️ Core Technologies

- **YOLO**: State-of-the-art object detection
- **Python 3.8+**: Primary language
- **OpenCV**: Computer vision operations
- **Ultralytics**: YOLO framework
- **NumPy & PIL**: Data processing
