import cv2
import os
import glob
import numpy as np
from bebop_gate_net import GateDetector

# ============================================================================
# CONFIGURATION -- this file is not in use anymore go to inference.py for the new viewer
# ============================================================================
TEST_DIR = './cyberzoo_test'         
MODEL_PATH = 'best_gate_model.keras' 
HSV_LOWER = [50, 50, 100]             
HSV_UPPER = [125, 255, 255]

def main():
    print("\n" + "="*60)
    print("STARTING BEBOP GATE CLASSIFICATION VIEWER")
    print("="*60)
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Could not find model at {MODEL_PATH}")
        return
        
    print(f"Loading model from {MODEL_PATH}...")
    # The detector now handles the dictionary output internally
    detector = GateDetector(MODEL_PATH, HSV_LOWER, HSV_UPPER)
    
    image_paths = glob.glob(os.path.join(TEST_DIR, '*.jpg')) 
    if not image_paths:
        print(f"❌ No images found in '{TEST_DIR}/'")
        return
        
    print(f"✓ Found {len(image_paths)} images.")
    print("\nCONTROLS:")
    print(" - Press any key to see the next image")
    print(" - Press 'q' or 'Esc' to quit\n")
    
    cv2.namedWindow('Bebop Final Prediction')
    cv2.moveWindow('Bebop Final Prediction', 50, 50)  
    
    cv2.namedWindow('What the Network Sees')
    cv2.moveWindow('What the Network Sees', 600, 50)  

    for img_path in image_paths:
        img = cv2.imread(img_path)
        hsv_debug = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        cv2.imshow("Hue Channel", hsv_debug[:,:,0]) # Show only the colors
        if img is None: continue
            
        h, w = img.shape[:2]

        direction_name, prob, confidence, masked_img = detector.detect(img_path)

       
        # --- 1. DISPLAY NETWORK INPUT ---
        if masked_img is not None:
            network_view = (masked_img * 255).astype(np.uint8)
            cv2.imshow('What the Network Sees', network_view)
        

        cv2.rectangle(img, (0, 0), (w, 110), (0, 0, 0), -1)
        
        color = (0, 255, 0) if confidence > 0.5 else (0, 0, 255)
    
        probs = prob if isinstance(prob, (list, np.ndarray)) else [0, 0, 0]
        
        labels = ["LEFT", "STRAIGHT", "RIGHT"]
        for i, label in enumerate(labels):
            p_val = probs[i]
            bar_w = int(p_val * (w - 150)) # Scale bar to image width
            
            # Text label
            cv2.putText(img, f"{label}:", (10, 25 + (i * 25)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Probability Bar
            cv2.rectangle(img, (100, 15 + (i * 25)), (100 + bar_w, 30 + (i * 25)), color, -1)
            
            # Percentage Text
            cv2.putText(img, f"{p_val:.1%}", (110 + bar_w, 25 + (i * 25)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Confidence readout at the bottom of the bar
        cv2.putText(img, f"GATE CONFIDENCE: {confidence:.2f}", (10, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow('Bebop Final Prediction', img)
        
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q') or key == 27: break
if __name__ == "__main__":
    main()
