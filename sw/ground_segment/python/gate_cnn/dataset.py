import json
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class GateDataset(Dataset):
    def __init__(self, json_file, img_dir):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.img_dir = img_dir
        
        # HSV color range for gate detection
        self.lower_hsv = np.array([50, 90, 100])  
        self.upper_hsv = np.array([150, 255, 255]) 
        
        # Angle threshold for "straight" classification
        self.straight_threshold = 0.15 # ±0.15 radians ≈ ±8.6 degrees

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_name = item['img_name']
        
        img_bgr = cv2.imread(f"{self.img_dir}/{img_name}")
        
        if img_bgr is None:
            # Return blank if image missing
            # Original: 520 high × 240 wide
            # Resized: 260 high × 120 wide (divide by 2)
            mask = np.zeros((260, 120), dtype=np.uint8)
        else:
            # 1. HSV Conversion
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            
            # 2. Color filtering
            mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
            
            # 3. Morphological operations to clean up the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # 4. Resize: divide both dimensions by 2
            # Original: 520 high × 240 wide
            # Resized: 260 high × 120 wide
            mask = cv2.resize(mask, (120, 260), interpolation=cv2.INTER_LINEAR)
        
        # Convert to Tensor (Normalized 0.0 to 1.0)
        # Shape: 1 × 260 × 120 (channels × height × width)
        img_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) / 255.0
        
        # Label Mapping
        if item['confidence'] == 0.0:
            label = 0  # No gate
        else:
            angle = item['heading_angle']
            if angle < -self.straight_threshold:
                label = 1  # Left
            elif angle > self.straight_threshold:
                label = 3  # Right
            else:
                label = 2  # Straight
                
        return img_tensor, torch.tensor(label, dtype=torch.long)
