# 🌊 Marine AI - Embedded Intelligent Microscopy System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![Gradio](https://img.shields.io/badge/Gradio-4.0%2B-orange.svg)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**An intelligent embedded microscopy platform powered by AI for automated identification, classification, and enumeration of marine microorganisms.**

Created by **Team CodeFather** for Smart India Hackathon 2024

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Hardware Requirements](#hardware-requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Demo Video](#demo)
- [Model Information](#model-information)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Contact](#contact)

---

## 🌐 Overview

Traditional marine biodiversity assessments depend on manual microscopic examination of planktonic organisms—a methodology that proves **time-intensive** (10-20 minutes per specimen), **resource-demanding**, **prone to subjectivity**, and **difficult to scale**. 

Our innovation delivers an **offline-capable, cost-efficient embedded AI platform** powered by Raspberry Pi 5 that enables:
- ✅ **Detection** of marine microorganisms utilizing YOLOv8
- ✅ **Classification** through high-precision deep learning algorithms
- ✅ **Enumeration** with comprehensive statistical analysis
- ✅ **Real-time inference** on embedded computing hardware
- ✅ **Intuitive web interface** for seamless operation

### Why This Solution Matters
- ⏱️ **Accelerates analysis** from 20 minutes to 20 seconds
- 💰 **Reduces operational costs** by 10x versus conventional approaches
- 🚢 **Field-ready deployment** for vessels and remote coastal laboratories
- 🇮🇳 **Indigenous innovation** advancing Make in India and Digital India initiatives
- 🌍 **UN SDG compliance** (SDG 6: Clean Water, SDG 14: Life Below Water)

---

## ✨ Features

### Core Capabilities
- **Real-time Detection**: YOLOv8 identifies multiple overlapping organisms simultaneously
- **High-Precision Classification**: Trained on comprehensive marine zooplankton datasets
- **Automated Enumeration**: Systematic counting of organisms by taxonomic classification
- **Offline Functionality**: Complete operation without internet connectivity
- **User-Centric Dashboard**: Gradio-powered web interface with visual analytics
- **Comprehensive Analytics**: Species diversity metrics, abundance patterns, confidence scoring

### Technical Highlights
- 🖥️ **Embedded AI**: Optimized for Raspberry Pi 5 (8GB)
- ⚡ **Enhanced Inference**: Production-ready YOLO model implementation
- 🎯 **Multi-Scale Detection**: Processes organisms ranging from 2µm to 200µm
- 📊 **Visual Reporting**: Color-coded results with detailed statistics
- 🎨 **Professional Interface**: Contemporary gradient-based user experience

---

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────────────┐
│                  USB Digital Microscope                │
│                    (1080p/4K Imaging)                  │
└───────────────────────┬────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────┐
│              Raspberry Pi 5 (8GB RAM)                  │
│  ┌──────────────────────────────────────────────────┐  │
│  │         Image Preprocessing Pipeline             │  │
│  │  • Quality Assessment  • Normalization           │  │
│  └───────────────────┬──────────────────────────────┘  │
│                      │                                 │
│  ┌───────────────────▼──────────────────────────────┐  │
│  │     YOLOv8 Detection & Classification            │  │
│  │  • Multi-scale Detection  • Real-time Inference  │  │
│  └───────────────────┬──────────────────────────────┘  │
│                      │                                 │
│  ┌───────────────────▼──────────────────────────────┐  │
│  │    Post-Processing & Analytics Engine            │  │
│  │  • Counting  • Statistics  • Visualization       │  │
│  └───────────────────┬──────────────────────────────┘  │
└────────────────────┬─┴─────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
┌──────────────┐         ┌──────────────┐
│   Gradio     │         │   Export     │
│   Dashboard  │         │   Results    │
│   127.0.0.1  │         │   (Images)   │
└──────────────┘         └──────────────┘
```

---

## 🔧 Hardware Requirements

### Development Environment (Current Configuration)
- **Computer**: Windows/Linux/Mac with Python 3.9 or higher
- **RAM**: 8GB minimum (16GB recommended for optimal performance)
- **Storage**: 10GB available disk space
- **GPU**: Optional (CUDA-compatible for accelerated inference)

### Production Deployment (Raspberry Pi Configuration)
| Component | Specification | Cost (INR) |
|-----------|--------------|-----------|
| **Raspberry Pi 5** | 8GB RAM | ₹8,000 |
| **USB Microscope** | 1080p/4K Digital | ₹2,000 |
| **Power Supply** | 27W USB-C PSU | ₹800 |
| **Storage** | 64GB microSD Card | ₹600 |
| **Cooling** | Active Fan/Heatsink | ₹400 |
| **Case** | Protective Enclosure | ₹200 |
| **Total** | - | **₹12,000** |

---

## 📦 Installation

### Prerequisites
```bash
# Verify Python version (3.9 or higher required)
python --version

# Verify pip installation
pip --version
```

### Step 1: Acquire Project Files
```bash
# Using Git
git clone https://github.com/YourUsername/Marine-AI.git
cd Marine-AI

# Alternatively, download and extract to D:\Marine-AI
```

### Step 2: Configure Virtual Environment (Recommended)
```bash
# Navigate to project directory
cd D:\Marine-AI

# Initialize virtual environment
python -m venv Marine-AI

# Activate environment
# Windows Command Prompt:
Marine-AI\Scripts\activate

# Windows PowerShell:
Marine-AI\Scripts\Activate.ps1

# Linux/Mac:
source Marine-AI/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note**: Installation duration may range from 5-10 minutes based on network speed. PyTorch comprises the largest package (~2GB).

### Step 4: Validate Installation
```bash
# Verify package installation
pip list

# Expected packages: ultralytics, gradio, torch, opencv-python, etc.
```

### Step 5: Position Model File
Confirm your trained model resides at the designated location:
```
D:\Marine-AI\models\best.pt
```

---

## 🚀 Usage

### Quick Start Guide

#### Method 1: Virtual Environment Execution (Recommended)
```bash
# Launch Command Prompt/PowerShell
cd D:\Marine-AI

# Activate virtual environment
Marine-AI\Scripts\activate

# Execute application
python app.py
```

#### Method 2: Direct Python Execution
```bash
# Execute from any directory
python D:\Marine-AI\app.py

# Alternatively, use Python 3 explicitly
python3 D:\Marine-AI\app.py
```

#### Method 3: From Activated Environment
```bash
# If (Marine-AI) environment is already active
cd D:\Marine-AI
python app.py
```

### Expected Console Output
```
Loading YOLOv8 model from: D:\Marine-AI\models\best.pt
Model loaded successfully!
Found free port: 7860
Starting Clean Gradio App...
Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://xxxxx.gradio.live

To create a public link, set `share=True` in `launch()`.
```

### Accessing the Application

1. **Local Access** (Same Device):
   - Navigate to: `http://127.0.0.1:7860` in your browser
   
2. **Network Access** (Remote Devices on Identical Network):
   - Utilize the public Gradio link displayed in terminal
   - Link remains active for 72 hours

### Interface Operation Guide

**Complete Workflow:**

1. **Upload Sample Image**
   - Select "Select Marine Sample Image"
   - Choose an image from `D:\Marine-AI\test_img\` directory
   - Supported formats: JPG, PNG, JPEG

2. **Initiate Analysis**
   - Click "Analyze Microorganism sample" button
   - Processing time: 2-5 seconds

3. **Review Results**
   - **Before Detection**: Original microscopy image
   - **After Detection**: Annotated image featuring colored bounding boxes
   - **Analysis Summary**: 
     - Total organism count
     - Species diversity metrics
     - Dominant species identification
     - Detailed taxonomic breakdown with counts and confidence scores

4. **Color Interpretation**
   - Each species features a distinctive color code
   - Color legend provided in detailed breakdown
   - Bounding boxes correspond to species-specific colors

5. **Export Results** (Optional)
   - Right-click annotated image → "Save Image As"
   - Copy analysis summary for documentation

---

## DEMO VIDEO:

[![Title](https://img.youtube.com/vi/WEvNaXpfy-I/maxresdefault.jpg)](https://youtu.be/WEvNaXpfy-I)

---

## 🧠 Model Information

### Supported Marine Species (13 Classifications)

| Species | Color Code | Description |
|---------|-----------|-------------|
| **Chaetognath** | 🔴 Red | Arrow worms, predatory zooplankton |
| **Larval Fish** | 🟢 Green | Early developmental stage fish |
| **Hydromedusa** | 🔵 Blue | Jellyfish-like organisms |
| **Lobate Ctenophore** | 🟡 Yellow | Comb jellies featuring lobes |
| **Pleurobrachia** | 🟣 Magenta | Sea gooseberry ctenophore |
| **Shrimp** | 🔷 Cyan | Decapod crustaceans |
| **Siphonophore** | 🟣 Purple | Colonial marine organisms |
| **Stomatopod Larva** | 🌊 Teal | Mantis shrimp larvae |
| **Thaliac** | 🟠 Orange | Salps and related organisms |
| **Polychaete Worm** | 💜 Indigo | Segmented marine worms |
| **Cumacean** | 🌸 Violet | Diminutive crustaceans |
| **Ctenophore** | 🌳 Dark Green | Comb jellies |
| **Unknown** | ⚪ Gray | Unclassified organisms |

### Model Specifications

**YOLOv8 Detection Architecture**
- **Framework**: YOLOv8 (Ultralytics)
- **Model File**: `D:\Marine-AI\models\best.pt`
- **Input Dimensions**: Variable (auto-resized to 640x640)
- **Confidence Threshold**: 0.3 (30%)
- **Output Format**: Bounding boxes with taxonomic labels and confidence scores

**Performance Metrics**
- **Detection Speed**: ~2-5 seconds per image (CPU operation)
- **Accuracy**: High precision on training dataset
- **Batch Processing**: Supported functionality

---

## 📁 Project Structure

```
D:\Marine-AI\
│
├── app.py                      # Primary Gradio application (Entry point)
├── requirements.txt            # Python dependency specifications
├── README.md                   # Documentation file
│
├── models\
│   └── best.pt                 # Trained YOLOv8 model weights
│
├── test_img\                   # Test imagery directory
│   ├── sample1.jpg             # Example microscope captures
│   ├── sample2.png
│   └── ...
│
└── Marine-AI\                  # Virtual environment (if configured)
    ├── Scripts\
    ├── Lib\
    └── ...
```

### File Descriptions

- **`app.py`**: Primary application featuring Gradio UI and YOLO detection logic
- **`requirements.txt`**: Complete Python package dependencies
- **`models\best.pt`**: Pre-trained YOLOv8 model (required component)
- **`test_img\`**: Sample images for system validation
- **`Marine-AI\`**: Virtual environment directory (optional but recommended)

---

## 🔧 Troubleshooting

### Common Issues and Resolutions

#### 1. Model File Not Located
```
ERROR: Model file not found at D:\Marine-AI\models\best.pt
```
**Resolution:**
- Confirm model file exists at precisely `D:\Marine-AI\models\best.pt`
- Verify filename is `best.pt` (case-sensitive on Linux systems)
- Ensure possession of trained model file

#### 2. Missing Module Error
```
ModuleNotFoundError: No module named 'ultralytics'
```
**Resolution:**
```bash
# Reinstall requirements
pip install -r requirements.txt

# Or install packages individually
pip install ultralytics gradio opencv-python torch
```

#### 3. Port Conflict
```
OSError: [Errno 48] Address already in use
```
**Resolution:**
- Terminate other applications utilizing port 7860
- Application will automatically locate available port

#### 4. CUDA/GPU Errors (Optional GPU Configuration)
```
RuntimeError: CUDA out of memory
```
**Resolution:**
- Expected behavior on CPU-only systems
- YOLOv8 automatically defaults to CPU operation
- Single-image performance remains satisfactory

#### 5. Gradio Public Link Inaccessible
**Resolution:**
- Utilize local URL: `http://127.0.0.1:7860` instead
- Verify firewall configurations
- Restart application

#### 6. Reduced Inference Performance
**Resolution:**
- Utilize smaller images (resize to 1920x1080 or below)
- Terminate unnecessary applications
- Consider GPU utilization if available
- On Raspberry Pi: ensure active cooling is operational

### Support Resources

If issues persist:
1. Examine error messages thoroughly
2. Verify correct file placement
3. Confirm virtual environment activation
4. Attempt dependency reinstallation
5. Submit GitHub issue with comprehensive error logs

---

## 📊 Performance Metrics

### Current System (Development Configuration)
- **Detection Duration**: 2-5 seconds per image
- **Supported Resolutions**: Up to 4K resolution
- **Concurrent Users**: Single user (Gradio limitation)
- **Accuracy**: High precision (based on training dataset)

### Target Deployment (Raspberry Pi 5)
- **Detection Duration**: ~10-15 seconds per image (with optimization)
- **Power Consumption**: 10-12W
- **Offline Capability**: 100% functional
- **Storage Options**: SD card or external SSD

---

## 🌍 Impact & Applications

### Marine Research
- 🔬 Automated biodiversity surveillance
- 📊 Long-term ecological research programs
- 🌡️ Climate change impact evaluation
- 🗺️ Spatial distribution mapping

### Aquaculture & Fisheries Management
- 💧 Water quality surveillance
- 🦐 Plankton abundance monitoring
- 🐟 Feed optimization strategies
- ⚠️ Early warning systems for blooms

### Environmental Conservation
- 🌊 Harmful algal bloom identification
- 🏭 Pollution indicator surveillance
- 🌴 Coastal ecosystem health assessment
- 🛡️ Marine conservation initiatives

### Education & Capacity Building
- 🎓 Affordable AI microscopy for academic institutions
- 👨‍🎓 Practical embedded systems training
- 📚 Open-source research infrastructure
- 🔬 STEM education resource

---

## 🤝 Contributing

Contributions are welcomed! Areas for contribution:

- 🐛 **Bug Reports**: Discovered an issue? Inform us
- 💡 **Feature Proposals**: Enhancement suggestions
- 📝 **Documentation**: Improve guides and tutorials
- 🔬 **Dataset Contributions**: Share marine organism imagery
- 💻 **Code**: Submit pull requests

---

## 👥 Team CodeFather

**Smart India Hackathon 2024**

Developed with 💙 for Marine Conservation

---

## 👥 Authors

**DIVYESH HARI G**  
📧 divyesh02208@gmail.com  
🔗 [github.com/DIVYESH-HARI](https://github.com/DIVYESH-HARI)

**VIJAYA KARTHICK RAJA U M**  
📧 vkr3056@gmail.com  
🔗 [github.com/KARTHICK-3056](https://github.com/KARTHICK-3056)

**S.S.MADHAVAN**  
📧 ssmadhavan006@gmail.com  
🔗 [github.com/ssmadhavan006](https://github.com/ssmadhavan006)

**G.K.AKASHGAUTHAM**  
📧 gkakash2006@gmail.com  
🔗 [github.com/Akashgautham](https://github.com/Akashgautham)

**K.RAKSHITHASRI**  
📧 rakshiekt@gmail.com  
🔗 [github.com/rakshithasri-k](https://github.com/rakshithasri06)

**M.N.RAKSHA**  
📧 rakshanathan006@gmail.com  
🔗 [github.com/raksha006](https://github.com/raksha006)

---

## 🙏 Acknowledgments

- **Ultralytics**: For the exceptional YOLOv8 framework
- **Gradio Team**: For the remarkable web interface library
- **Marine Biology Community**: Dataset provision and validation support
- **Smart India Hackathon**: Platform and opportunity
- **Open Source Community**: For diverse tools and libraries

---

<div align="center">

### 🌊 **Safeguarding Marine Biodiversity Through AI Innovation** 🌊

**Made in India 🇮🇳 | For the Ocean 🌊 | Open Source 💻**

---

**[⭐ Star this project](https://github.com/YourUsername/Marine-AI)** | **[📖 Documentation](README.md)** | **[🐛 Report Bug](https://github.com/YourUsername/Marine-AI/issues)**

</div>
