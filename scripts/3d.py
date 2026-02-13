import cv2
import numpy as np
import os

"""
3D Point Cloud Reconstruction Module.

This script manually back-projects a single RGB-D frame into 3D space using a pinhole camera model.
It generates a .ply file that can be visualized in MeshLab or Open3D.
"""

# === SETTINGS ===
BASE_DIR = "../"
VIDEO_SOURCE = os.path.join(BASE_DIR, "video_learning_samples/add_remove_lid/0.mp4")
DEPTH_SOURCE = os.path.join(BASE_DIR, "outputs/output_depth_vis.mp4")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs/3d_reconstruction")
FRAME_IDX = 45
FX, FY = 1000.0, 1000.0 # Focal lengths

def process_frame():
    """
    Reads RGB and Depth frames, computes 3D coordinates, and saves a .ply point cloud.
    Also generates an artifact visualization for debugging flying pixels.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"🚀 Starting Manual 3D Extraction...")

    # 1. READ IMAGES
    cap_rgb = cv2.VideoCapture(VIDEO_SOURCE)
    cap_rgb.set(cv2.CAP_PROP_POS_FRAMES, FRAME_IDX)
    _, rgb = cap_rgb.read()
    cap_rgb.release()

    cap_depth = cv2.VideoCapture(DEPTH_SOURCE)
    cap_depth.set(cv2.CAP_PROP_POS_FRAMES, FRAME_IDX)
    _, depth_vis = cap_depth.read()
    cap_depth.release()

    if rgb is None or depth_vis is None: return print("❌ Error: Could not read frames.")

    # 2. RESIZE & FORMAT
    h, w = rgb.shape[:2]
    if depth_vis.shape[:2] != (h, w):
        depth_vis = cv2.resize(depth_vis, (w, h), interpolation=cv2.INTER_NEAREST)

    # 3. GENERATE ARTIFACT IMAGE (Flying Pixels)
    depth_gray = cv2.cvtColor(depth_vis, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(depth_gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(depth_gray, cv2.CV_64F, 0, 1, ksize=5)
    mag = np.sqrt(sobelx**2 + sobely**2)
    
    # Save the "Red Artifacts" image for your slide
    mask = ((mag > 50) & (mag < 150)).astype(np.uint8)
    artifact_vis = rgb.copy()
    artifact_vis[mask > 0] = [0, 0, 255]
    cv2.imwrite(os.path.join(OUTPUT_DIR, "critique_flying_pixels.jpg"), artifact_vis)
    print(f"   ✅ Saved Slide Image: critique_flying_pixels.jpg")

    # 4. MANUAL 3D BACK-PROJECTION (No Open3D!)
    print("   ✨ Calculating 3D Points (NumPy)...")
    
    # Create Pixel Grid
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    
    # Normalize depth (0-255 -> 0.0-5.0 meters)
    z_metric = depth_gray.astype(np.float32) / 255.0 * 5.0
    
    # Filter: Remove background (infinity) for cleaner look
    valid_mask = z_metric > 0.1
    
    xx = xx[valid_mask]
    yy = yy[valid_mask]
    z = z_metric[valid_mask]
    colors = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)[valid_mask] / 255.0

    # Pinhole Camera Math: X = (u - cx) * Z / fx
    x = (xx - w/2) * z / FX
    y = (yy - h/2) * z / FY
    
    # Stack points (N, 3) and colors (N, 3)
    points = np.stack((x, -y, -z), axis=1) # Note: Flip Y and Z for correct view
    
    # 5. WRITE PLY FILE MANUALLY
    ply_path = os.path.join(OUTPUT_DIR, "reconstructed_scene.ply")
    print(f"   💾 Writing {len(points)} points to PLY...")
    
    with open(ply_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # Write data row by row
        for p, c in zip(points, colors):
            r, g, b = int(c[0]*255), int(c[1]*255), int(c[2]*255)
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {r} {g} {b}\n")

    print(f"   ✅ Success! Saved: {ply_path}")

if __name__ == "__main__":
    process_frame()