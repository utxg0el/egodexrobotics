import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt

# --- CONFIGURATION ---
H5_PATH = '/Users/utx/Desktop/code/egodexrobotics/video_learning_samples/add_remove_lid/0.hdf5'

# TUNING (The only numbers you need)
MIN_GRASP_WIDTH = 0.01  # 1cm (If smaller, hand is just closed/empty)
MAX_GRASP_WIDTH = 0.08  # 8cm (If larger, hand is just open)
STABILITY_THRESH = 0.002 # 2mm (How much jitter is allowed?)
DURATION_FRAMES = 15    # Must hold still for 0.5 seconds (30fps)

def main():
    with h5py.File(H5_PATH, 'r') as f:
        thumb = f['transforms']['rightThumbTip'][:, :3, 3]
        index = f['transforms']['rightIndexFingerTip'][:, :3, 3]
        
    # 1. Calculate Aperture
    aperture = np.linalg.norm(thumb - index, axis=1)
    # Smooth it slightly to remove sensor noise
    aperture = medfilt(aperture, 5)
    
    # 2. Calculate "Rate of Change" (Velocity of fingers)
    # How much did the aperture change per frame?
    aperture_velocity = np.abs(np.diff(aperture, prepend=aperture[0]))
    
    # 3. THE LOGIC: Find "Stalled" Frames
    # A grasp is defined as:
    #   (a) Fingers are NOT moving (Velocity near 0)
    #   (b) Fingers are NOT fully closed (Width > 1cm)
    #   (c) Fingers are NOT fully open (Width < 8cm)
    
    is_stable = aperture_velocity < STABILITY_THRESH
    is_holding_something = (aperture > MIN_GRASP_WIDTH) & (aperture < MAX_GRASP_WIDTH)
    
    potential_grasp = is_stable & is_holding_something
    
    # 4. Filter: Remove flickers (Must be stable for N frames)
    # We use a convolution to count consecutive True values
    kernel = np.ones(DURATION_FRAMES)
    convolved = np.convolve(potential_grasp.astype(int), kernel, mode='same')
    
    # If the sum is close to N, it means we had N consecutive Trues
    final_grasp_mask = convolved > (DURATION_FRAMES - 2)

    # 5. VISUALIZATION
    plt.figure(figsize=(10, 5))
    plt.plot(aperture, color='gray', label='Finger Distance', alpha=0.5)
    
    # Highlight the Grasp Zones
    # Identify start/end of grasp blocks for clean plotting
    grasp_indices = np.where(final_grasp_mask)[0]
    
    if len(grasp_indices) > 0:
         # Split into segments where indices are not continuous
        splits = np.where(np.diff(grasp_indices) > 1)[0] + 1
        segments = np.split(grasp_indices, splits)
        
        for seg in segments:
            start, end = seg[0], seg[-1]
            plt.axvspan(start, end, color='orange', alpha=0.4, label='Stall Detected (Object Held)')
            # Add text label
            mid = (start + end) // 2
            width = aperture[mid]
            plt.text(mid, width + 0.01, f"{width*100:.1f}cm", ha='center', color='darkorange', fontweight='bold')

    # Remove duplicate legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    plt.title("Physics-Based Stall Detection (Clutter-Proof)")
    plt.ylabel("Aperture (m)")
    plt.xlabel("Frame")
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()