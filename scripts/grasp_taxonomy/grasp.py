import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# --- CONFIGURATION ---
H5_PATH = '/Users/utx/Desktop/code/egodexrobotics/video_learning_samples/add_remove_lid/0.hdf5'

def get_smart_joint(h5_group, possible_names):
    """
    Tries to find a joint data using a list of possible names.
    Returns the translation vector (x, y, z).
    """
    for name in possible_names:
        if name in h5_group:
            return h5_group[name][:, :3, 3]
    
    available_keys = list(h5_group.keys())
    raise KeyError(
        f"\n\n❌ CRITICAL ERROR: Could not find any of these joints: {possible_names}\n"
        f"   Available joints in file: {available_keys[:10]}... (Total {len(available_keys)})\n"
    )

def classify_grasps(h5_path):
    with h5py.File(h5_path, 'r') as f:
        if 'transforms' in f:
            base = f['transforms']
        else:
            base = f
            
        # --- ROBUST JOINT LOADING ---
        wrist = get_smart_joint(base, ['rightHand', 'rightWrist', 'wrist', 'hand_right'])
        thumb_tip = get_smart_joint(base, ['rightThumbTip', 'thumb_tip', 'rightThumb4'])
        thumb_ip = get_smart_joint(base, ['rightThumbIP', 'rightThumbIntermediateTip', 'rightThumbDistal', 'rightThumb2'])
        index_tip = get_smart_joint(base, ['rightIndexFingerTip', 'index_tip', 'rightIndex4'])
        index_pip = get_smart_joint(base, ['rightIndexFingerPIP', 'rightIndexFingerIntermediateBase', 'rightIndexIntermediate', 'rightIndex1'])
        middle_tip = get_smart_joint(base, ['rightMiddleFingerTip', 'middle_tip', 'rightMiddle4'])
        ring_tip   = get_smart_joint(base, ['rightRingFingerTip', 'ring_tip', 'rightRing4'])

    # --- FEATURE ENGINEERING ---
    dist_thumb_index = np.linalg.norm(thumb_tip - index_tip, axis=1)
    dist_thumb_side = np.linalg.norm(thumb_tip - index_pip, axis=1)
    
    dist_middle_wrist = np.linalg.norm(middle_tip - wrist, axis=1)
    dist_ring_wrist   = np.linalg.norm(ring_tip - wrist, axis=1)
    avg_curl_dist = (dist_middle_wrist + dist_ring_wrist) / 2.0
    
    # --- CLASSIFICATION LOGIC ---
    PRECISION_THRESH = 0.04
    POWER_CURL_THRESH = 0.08 
    
    classifications = []
    
    for i in range(len(wrist)):
        d_tip = dist_thumb_index[i]
        d_side = dist_thumb_side[i]
        curl = avg_curl_dist[i]
        
        label = "IDLE"
        color = "lightgray"
        
        if d_tip < PRECISION_THRESH or d_side < PRECISION_THRESH:
            if curl < POWER_CURL_THRESH and d_tip > 0.03:
                label = "✊ POWER GRASP"
                color = "#ff7f0e"
            elif d_tip < (d_side + 0.01): 
                label = "👌 PRECISION PINCH"
                color = "#2ca02c"
            else:
                label = "🔑 LATERAL PINCH"
                color = "#1f77b4"
        
        classifications.append({'frame': i, 'label': label, 'color': color})
        
    return pd.DataFrame(classifications)

def main():
    print("🧠 Analyzing Kinematic Geometry...")
    try:
        df = classify_grasps(H5_PATH)
    except KeyError as e:
        print(e)
        return
    except OSError:
        print(f"❌ Could not open file: {H5_PATH}")
        return

    # --- FIX: NUMERIC SMOOTHING ---
    print("📊 Smoothing Labels...")
    
    # 1. Convert Text Labels to Numbers (0, 1, 2...)
    # This prevents the "could not convert string to float" error
    df['label'] = df['label'].astype('category')
    df['label_code'] = df['label'].cat.codes
    
    # 2. Smooth the Numbers (find the most common number in the last 10 frames)
    # Using scipy mode for speed and reliability
    def get_mode(x):
        return stats.mode(x, keepdims=True)[0][0]

    df['smooth_code'] = df['label_code'].rolling(window=10).apply(get_mode, raw=True)
    
    # 3. Handle NaNs (First 10 frames will be empty)
    df['smooth_code'] = df['smooth_code'].fillna(0).astype(int)
    
    # 4. Convert Numbers back to Text
    # We use the category map from step 1
    categories = df['label'].cat.categories
    df['smooth_label'] = df['smooth_code'].apply(lambda x: categories[x])
    
    # Also recover the color (using the first color found for that label)
    # Create a quick lookup map: Label -> Color
    color_map = df.drop_duplicates('label').set_index('label')['color'].to_dict()
    df['smooth_color'] = df['smooth_label'].map(color_map)

    # --- VISUALIZATION ---
    plt.figure(figsize=(15, 4))
    
    # Group by the SMOOTHED labels
    changes = df['smooth_label'].ne(df['smooth_label'].shift()).cumsum()
    groups = df.groupby(changes)
    
    for _, group in groups:
        start = group['frame'].iloc[0]
        end = group['frame'].iloc[-1]
        label = group['smooth_label'].iloc[0]
        color = group['smooth_color'].iloc[0]
        
        if label != "IDLE":
            plt.axvspan(start, end, color=color, alpha=0.8)
            if (end - start) > 15: # Only label if the segment is long enough
                mid = (start + end) / 2
                plt.text(mid, 0.5, label, ha='center', va='center', 
                         color='white', fontweight='bold', fontsize=9)

    plt.title("Grasp Taxonomy Classification (Auto-Detected Joints)", fontsize=14)
    plt.xlabel("Frame Index")
    plt.yticks([])
    plt.ylim(0, 1)
    
    # Custom Legend
    patches = [
        mpatches.Patch(color='#2ca02c', label='Precision (Pen)'),
        mpatches.Patch(color='#1f77b4', label='Lateral (Key)'),
        mpatches.Patch(color='#ff7f0e', label='Power (Bottle)')
    ]
    plt.legend(handles=patches, loc='upper right')
    
    plt.grid(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()