# CAM Visualization Tool for ASL Recognition

This repository contains tools for visualizing Class Activation Maps (CAM) on an American Sign Language (ASL) recognition model. It includes both an interactive web application and a command-line comparison tool.

## 📋 Features

- **Web Application (`app.py`)**: Interactive Flask-based web interface to browse through test images with multiple CAM visualizations.
- **Command-Line Tool (`cam_comparison.py`)**: Interactive terminal-based tool to compare different CAM methods.

### CAM Methods Included:
- **Original Grad-CAM**: Standard Gradient-weighted Class Activation Mapping
- **MD-CAM**: MultiDeep-CAM (biases deeper layers)
- **MF-CAM**: MultiFine-CAM (biases shallower layers)
- **Gaussian-CAM**: Gaussian-CAM (emphasizes middle layers)

## 🔧 Prerequisites

- Python 3.7 or higher
- TensorFlow 2.x
- Flask (for web app)
- Matplotlib
- NumPy
- PIL (Pillow)

## 📦 Installation

1. Clone this repository:
```bash
git clone <your-repository-url>
cd <repository-name>
```

2. Install required dependencies:
```bash
pip install tensorflow flask matplotlib numpy pillow
```

3. Download the model file:
   - The trained model (`model.h5`) is too large for GitHub
   - Download it from: [Google Drive Link](https://drive.google.com/file/d/1aIBktNzCUGlPwp9j7q6V22zTZln0N6k7/view?usp=sharing)
   - Place the downloaded `model.h5` file in the root directory of this repository

## 📁 Repository Structure

```
.
├── app.py                  # Flask web application
├── cam_comparison.py       # Command-line comparison tool
├── model.h5               # Trained model (download separately)
├── test_images/           # Folder containing test images
├── templates/
│   └── index.html        # Web interface template
└── README.md             # This file
```

## 🚀 Usage

### Option 1: Web Application (app.py)

The web application provides an interactive interface to browse through images and view CAM visualizations.

**Run the application:**
```bash
python app.py
```

**Access the interface:**
1. Open your web browser
2. Navigate to: `http://localhost:5000`
3. Use the navigation controls to browse through test images
4. View multiple CAM visualizations for each image

**Features:**
- Browse through all test images with Next/Previous buttons
- View prediction results with confidence scores
- Compare Original Grad-CAM, MD-CAM, MF-CAM, and G-CAM
- See both heatmaps and overlay visualizations
- View top-5 predictions for each image
- Automatic detection of model architecture

**Requirements:**
- `model.h5` must be in the same directory
- `test_images/` folder must contain test images
- `templates/index.html` must be present

---

### Option 2: Command-Line Tool (cam_comparison.py)

The command-line tool provides an interactive matplotlib-based interface for comparing CAM methods.

**Run the tool:**
```bash
python cam_comparison.py
```

**How to use:**
1. The script will automatically load the model and analyze its architecture
2. It will find all images in the `test_images/` folder
3. A matplotlib window will open showing the first image with all CAM comparisons
4. **Press 'N'** to view the next image
5. **Press 'Q'** to quit the application

**Features:**
- Interactive keyboard controls (N for next, Q for quit)
- Side-by-side comparison of all CAM methods
- Displays prediction results with confidence
- Shows layer weights used for each multi-layer CAM method
- Automatic model architecture detection and analysis

**Requirements:**
- `model.h5` must be in the same directory
- `test_images/` folder must contain test images (`.jpg`, `.png`, `.jpeg`)

---

## 🖼️ Test Images

Place your test images in the `test_images/` folder. The scripts will automatically detect and load all images with the following extensions:
- `.jpg` / `.JPG`
- `.jpeg` / `.JPEG`
- `.png` / `.PNG`

**Image naming convention (optional):**
For automatic label detection, name your images with the expected label as a prefix:
- `A_sample1.jpg` → Expected label: A
- `B_test.png` → Expected label: B
- `C_example.jpg` → Expected label: C

## 🎯 ASL Classes

The model recognizes 24 ASL classes:
- Letters: A-Z, except J and Z

## ⚙️ Configuration

### app.py Configuration
Edit these variables at the top of `app.py`:
```python
MODEL_PATH = 'model.h5'        # Path to your model
IMAGE_FOLDER = 'test_images'   # Path to test images folder
POWER = 2.0                    # Power parameter for MD-CAM/MF-CAM
```

### cam_comparison.py Configuration
Edit these variables at the top of `cam_comparison.py`:
```python
MODEL_PATH = 'model.h5'   # Path to your model
POWER = 2.0               # Power parameter for MD-CAM/MF-CAM
SIGMA = POWER             # Sigma for Gaussian-CAM (None = use num_layers)
```

## 🐛 Troubleshooting

**"Model file not found" error:**
- Ensure `model.h5` is downloaded and placed in the repository root directory

**"No images found" error:**
- Check that the `test_images/` folder exists
- Verify that it contains image files with supported extensions

**Import errors:**
- Install all required dependencies: `pip install tensorflow flask matplotlib numpy pillow`

**Web app not loading:**
- Check that port 5000 is not being used by another application
- Try accessing `http://127.0.0.1:5000` instead of `localhost`

## 📝 License

Under the CC BY-NC 4.0 license, you are free to:
- Share: copy and redistribute the material in any medium or format.
- Adapt: remix, transform, and build upon the material.
Under the following terms:
- Attribution: You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
- NonCommercial: You may not use the material for commercial purposes.


## 👥 Authors

Elisa Cabana, CUNEF Universidad, Madrid, Spain
elisa.cabana@cunef.edu

## 🙏 Acknowledgments

- The model used here was developed in the following study:
Cabana, E. (2025). Advancing Accessible AI: A Comprehensive Dataset and Neural Models for Real-Time American Sign Language Alphabet Classification. In: Arai, K. (eds) Intelligent Systems and Applications. IntelliSys 2025. Lecture Notes in Networks and Systems, vol 1567. Springer, Cham. https://doi.org/10.1007/978-3-032-00071-2_15. Dataset and model: https://github.com/ecabestadistica/SignLanguageRecognition/  
