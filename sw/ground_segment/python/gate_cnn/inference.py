import cv2
import torch
import numpy as np
import os
import glob
from model import GateNet # Imports your model from model.py

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_WEIGHTS = "gate_model_best.pth"  # Path to your trained weights
IMAGE_FOLDER = "./cyberzoo_test"       # Folder containing images to test
CLASSES = ["None", "Left", "Straight", "Right"]
DISPLAY_SCALE = 2.0                    # Multiplier to make the displayed image bigger

# Global flag for the mouse click event
advance_frame = False

def mouse_click(event, x, y, flags, param):
    """Callback function to detect mouse clicks."""
    global advance_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        advance_frame = True

def main():
    global advance_frame
    
    # 1. Setup Device & Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")
    
    model = GateNet().to(device)
    try:
        model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
        model.eval() # Set to evaluation mode (turns off dropout)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    # 2. Get list of images
    image_paths = sorted(glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg")))
    if not image_paths:
        print(f"No images found in {IMAGE_FOLDER}!")
        return
        
    print(f"Found {len(image_paths)} images. Click the window or press any key to advance. Press 'ESC' to quit.")

    # 3. Setup OpenCV Window and Mouse Callback
    window_name = "Drone Gate Inference Test"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_click)

    # 4. Loop through images
    for img_path in image_paths:
        # Read the raw BGR image
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        # --- PREPROCESSING (Uses original unrotated image) ---
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        lower_hsv = np.array([50, 90, 100])  
        upper_hsv = np.array([150, 255, 255]) 
        
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Resize to 120x260
        mask_resized = cv2.resize(mask, (120, 260), interpolation=cv2.INTER_LINEAR)
        
        # Convert to Tensor: Shape [1, 1, 260, 120] (Batch x Channel x Height x Width)
        img_tensor = torch.tensor(mask_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(device)
        
        # --- INFERENCE ---
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.softmax(output, dim=1)[0]
            pred_idx = output.argmax(dim=1).item()
            confidence = probabilities[pred_idx].item()
            
        prediction_text = CLASSES[pred_idx]
        
        # --- VISUALIZATION (Rotate, Resize, and Draw ONLY for display) ---
        display_img = img_bgr.copy()
        
        # 1. Rotate 90 degrees counter-clockwise
        display_img = cv2.rotate(display_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # 2. Resize to make the displayed image bigger
        display_img = cv2.resize(display_img, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, interpolation=cv2.INTER_LINEAR)

        # 3. Draw background box and Text
        color = (0, 255, 0) if prediction_text == "Straight" else (0, 165, 255)
        if prediction_text == "None":
            color = (0, 0, 255)

        cv2.rectangle(display_img, (10, 10), (280, 80), (0, 0, 0), -1)
        cv2.putText(display_img, f"Gate: {prediction_text}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(display_img, f"Conf: {confidence*100:.1f}%", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 4. Draw two vertical lines at +/- 10% from the center
        height, width = display_img.shape[:2]
        center_x = width // 2
        offset = int(width * 0.10) # 10% of total width
        
        left_line_x = center_x - offset
        right_line_x = center_x + offset
        
        cv2.line(display_img, (left_line_x, 0), (left_line_x, height), (255, 0, 0), 2)
        cv2.line(display_img, (right_line_x, 0), (right_line_x, height), (255, 0, 0), 2)

        # Show the image
        cv2.imshow(window_name, display_img)
        
        # --- WAIT FOR CLICK OR KEYPRESS ---
        advance_frame = False
        while not advance_frame:
            key = cv2.waitKey(10) & 0xFF
            if key == 27: # ESC key to quit early
                print("Exiting...")
                cv2.destroyAllWindows()
                return
            elif key != 255: # Any other key also advances the frame
                advance_frame = True

    cv2.destroyAllWindows()
    print("Finished testing all images!")

if __name__ == "__main__":
    main()
