import os
import json
import shutil

# ============================================================================
# MERGE CONFIGURATION - with this file we can merge the different 
# datasets together into one master dataset for training
# ============================================================================
REAL_IMG_DIR = "img"                 
REAL_LABELS = "labels.json"             

SYNTH_IMG_DIR = "synthetic_data/img"     
SYNTH_LABELS = "synthetic_data/labels.json"

OUT_IMG_DIR = "DATA/img"    
OUT_LABELS = "DATA/labels.json"

def merge_data():
    print("\n" + "="*60)
    print("MERGING REAL AND SYNTHETIC DATASETS (NO SCALING)")
    print("="*60)
    
    os.makedirs(OUT_IMG_DIR, exist_ok=True)
    combined_labels = []
    
    # 1. PROCESS REAL DATA
    if os.path.exists(REAL_LABELS) and os.path.exists(REAL_IMG_DIR):
        print("Processing REAL data...")
        with open(REAL_LABELS, 'r') as f:
            real_data = json.load(f)
            
        success_count = 0
        for item in real_data:
            img_name = item['img_name']
            src_path = os.path.join(REAL_IMG_DIR, img_name)
            dst_path = os.path.join(OUT_IMG_DIR, img_name)
            
            if os.path.exists(src_path):
                # Copy image
                shutil.copy2(src_path, dst_path)
                
                combined_labels.append({
                    "img_name": img_name,
                    "heading_angle": item.get('heading_angle', 0.0),
                    "confidence": item.get('confidence', 0.5)
                })
                success_count += 1
        print(f"✓ Added {success_count} real images (Headings untouched).")
    else:
        print("⚠️  Real data not found. Skipping...")

    # 2. PROCESS SYNTHETIC DATA
    if os.path.exists(SYNTH_LABELS) and os.path.exists(SYNTH_IMG_DIR):
        print("\nProcessing SYNTHETIC data...")
        with open(SYNTH_LABELS, 'r') as f:
            synth_data = json.load(f)
            
        success_count = 0
        for item in synth_data:
            img_name = item['img_name']
            src_path = os.path.join(SYNTH_IMG_DIR, img_name)
            dst_path = os.path.join(OUT_IMG_DIR, img_name)
            
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                combined_labels.append(item)
                success_count += 1
        print(f"✓ Added {success_count} synthetic images.")
    else:
        print("⚠️  Synthetic data not found. Skipping...")

    # 3. SAVE MASTER JSON
    with open(OUT_LABELS, 'w') as f:
        json.dump(combined_labels, f, indent=4)
        
    print("\n" + "="*60)
    print(f"✓ MERGE COMPLETE! Total Master Dataset: {len(combined_labels)} images.")
    print("="*60)

if __name__ == "__main__":
    merge_data()
