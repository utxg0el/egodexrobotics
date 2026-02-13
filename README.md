# EgoDex Robotics: Egocentric Vision for Robot Learning

**Extracting structure, semantics, and object permanence from egocentric video.**

This repository contains tools and scripts to analyze the **Apple EgoDex dataset**. It provides pipelines for:
- **Monocular Depth Estimation** (using Depth Anything)
- **3D Point Cloud Reconstruction** (from single-view RGB-D)
- **Object Tracking with Memory** (Hamer + Foundation Pose)
- **Segmentation** (SAM 2)

## 📂 Dataset
The code is designed to work with the **Apple EgoDex dataset**. 
- Place your video samples in `video_learning_samples/`.
- Processed outputs will be saved to `outputs/`.

## 🚀 Installation

1. **Clone the repository** (or download the source).
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: For GPU acceleration (CUDA/MPS), ensure you have the appropriate version of `torch` installed.*

## 🛠️ Usage

### 1. Depth Estimation
Generate a side-by-side RGB + Depth video from an input video.
```bash
python scripts/depth/generate_depth.py
```

### 2. 3D Reconstruction
Back-project a specific frame into a 3D Point Cloud (`.ply` format).
```bash
python scripts/3d.py
```
*Output: `outputs/3d_reconstruction/reconstructed_scene.ply` (View with MeshLab or Open3D)*

### 3. Object Tracking & Permanence
Track objects (Bottles, Cups, Bowls) and maintain their location in memory even when occluded.
```bash
python scripts/objects/bounding.py
```

## 🧠 Project Structure
- `scripts/`: Core logic for depth, 3D, and tracking.
- `notebooks/`: Exploratory Jupyter notebooks (SAM2, CoTracker, FoundationPose).
- `outputs/`: Generated results (Ignored by Git).

## 📄 License
MIT License.
