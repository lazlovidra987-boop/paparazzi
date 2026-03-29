import os
import json
import shutil
import argparse
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
from collections import defaultdict

class ImageLabeler:
    def __init__(self, input_folder, output_folder):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.images_folder = self.output_folder / "images"
        self.labels = []
        self.current_image_path = None
        self.current_image_name = None
        self.current_index = None
        self.window_name = "Image Labeler - Click for heading angle, 'N' for no gate, 'Q' to quit"
        self.click_x = None
        self.image_cv = None
        self.display_height = None
        self.display_width = None
        
        # Create output directories
        self.images_folder.mkdir(parents=True, exist_ok=True)
        
        # Collect all images from input folder and subfolders
        self.image_files = self._collect_images()
        self.total_images = len(self.image_files)
        
    def _collect_images(self):
        """Recursively collect all image files from input folder"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
        image_files = []
        
        for root, dirs, files in os.walk(self.input_folder):
            for file in sorted(files):
                if Path(file).suffix.lower() in image_extensions:
                    image_files.append(Path(root) / file)
        
        return sorted(image_files)
    
    def _copy_and_rename_image(self, index, source_path):
        """Copy image to output folder with new name"""
        source_img = Image.open(source_path)
        new_name = f"img_cz2-{index:04d}.jpg"
        output_path = self.images_folder / new_name
        
        # Convert to RGB if necessary (for PNG with alpha, etc.)
        if source_img.mode in ('RGBA', 'LA', 'P'):
            rgb_img = Image.new('RGB', source_img.size, (255, 255, 255))
            rgb_img.paste(source_img, mask=source_img.split()[-1] if source_img.mode == 'RGBA' else None)
            rgb_img.save(output_path, 'JPEG', quality=95)
        else:
            source_img.save(output_path, 'JPEG', quality=95)
        
        return new_name, output_path
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for click detection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_x = x
            # Redraw image immediately with line
            self._redraw_display()
    
    def _redraw_display(self):
        """Redraw the image with the line and stats"""
        display_img = self.image_cv.copy()
        if self.click_x is not None:
            # Draw vertical line at click position
            cv2.line(display_img, (self.click_x, 0), (self.click_x, self.display_height), (0, 255, 0), 8)
            
            # Calculate heading angle from click position
            heading = self._normalize_angle(self.click_x, self.display_width)
            confidence = 1  # Gate detected
            heading = round(heading, 3)
            
            # Draw small text with stats on image
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            font_thickness = 1
            text_color = (0, 255, 0)  # Green
            outline_color = (0, 0, 0)  # Black outline for contrast
            
            heading_text = f"Heading: {heading:.3f}"
            conf_text = f"Confidence: {confidence}"
            
            # Draw heading text with outline
            cv2.putText(display_img, heading_text, (20, 40), font, font_scale, outline_color, font_thickness + 1)
            cv2.putText(display_img, heading_text, (20, 40), font, font_scale, text_color, font_thickness)
            
            # Draw confidence text with outline
            cv2.putText(display_img, conf_text, (20, 65), font, font_scale, outline_color, font_thickness + 1)
            cv2.putText(display_img, conf_text, (20, 65), font, font_scale, text_color, font_thickness)
        
        cv2.imshow(self.window_name, display_img)
    
    def _normalize_angle(self, click_x, image_width):
        """Convert click position to heading angle between -1 and 1"""
        # Map click position to angle: left side = -1, center = 0, right side = 1
        normalized = (click_x / image_width) * 2 - 1
        # Clamp to [-1, 1]
        return max(-1, min(1, normalized))
    
    def _show_image_and_get_label(self, image_name, image_path, index):
        """Display image and wait for user input"""
        self.current_image_name = image_name
        self.current_image_path = image_path
        self.current_index = index
        self.click_x = None
        
        # Read image with OpenCV
        self.image_cv = cv2.imread(str(image_path))
        if self.image_cv is None:
            print(f"Warning: Could not read {image_path}")
            return None
        
        # Rotate image 90 degrees counter-clockwise
        self.image_cv = cv2.rotate(self.image_cv, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Create fullscreen window and set mouse callback
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        height, width = self.image_cv.shape[:2]
        self.display_height = height
        self.display_width = width
        
        print(f"\n[{index}/{self.total_images}] {image_name}")
        print("Instructions:")
        print("  - Click on the image where the heading angle should be")
        print("  - Press 'N' if there is no gate in the image")
        print("  - Press 'Q' to quit without saving")
        print("  - Press 'Enter'/'Space' to confirm and go to next image")
        
        heading = None
        confidence = None
        
        # Show initial image
        display_img = self.image_cv.copy()
        cv2.imshow(self.window_name, display_img)
        
        while True:
            key = cv2.waitKey(100) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                # Quit
                cv2.destroyAllWindows()
                return None
            
            elif key == ord('n') or key == ord('N'):
                # No gate in image
                heading = 0.0
                confidence = 0.0
                print("  -> No gate detected. Heading=0, Confidence=0")
                break
            
            elif key == 13 or key == 32:  # Enter or Space
                if self.click_x is None:
                    print("  Please click on the image or press 'N' for no gate")
                    continue
                
                # Calculate heading angle from click position (on rotated image)
                heading = self._normalize_angle(self.click_x, self.display_width)
                confidence = 1  # Gate detected = confidence 1
                heading = round(heading, 3)
                
                print(f"  -> Click at x={self.click_x}, Heading={heading}, Confidence={confidence}")
                break
        
        cv2.destroyAllWindows()
        
        if heading is not None and confidence is not None:
            return {
                "img_name": image_name,
                "heading_angle": heading,
                "confidence": confidence
            }
        
        return None
    
    def run(self):
        """Main labeling workflow"""
        if self.total_images == 0:
            print("No images found in the input folder!")
            return
        
        print(f"Found {self.total_images} image(s) to process")
        print(f"Output folder: {self.output_folder}")
        print(f"Images will be saved to: {self.images_folder}")
        
        for index, image_path in enumerate(self.image_files, start=1):
            # Copy and rename image
            new_name, output_path = self._copy_and_rename_image(index, image_path)
            print(f"\nProcessing: {image_path.name} -> {new_name}")
            
            # Show image and get label
            label = self._show_image_and_get_label(new_name, output_path, index)
            
            if label is None:
                print("Quitting without saving remaining images...")
                # Remove already processed images if user quits
                shutil.rmtree(self.output_folder)
                print(f"Removed incomplete output folder: {self.output_folder}")
                return
            
            self.labels.append(label)
        
        # Save labels to JSON
        self._save_labels()
        print(f"\n✓ All done! Processed {len(self.labels)} images")
        print(f"✓ Images saved to: {self.images_folder}")
        print(f"✓ Labels saved to: {self.output_folder / 'labels.json'}")
    
    def _save_labels(self):
        """Save labels to JSON file"""
        output_path = self.output_folder / "labels.json"
        
        with open(output_path, 'w') as f:
            json.dump(self.labels, f, indent=2)
        
        print(f"Labels file created: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect images, rename them, and annotate with heading angles"
    )
    parser.add_argument(
        "input_folder",
        help="Path to folder containing images (searches subfolders too)"
    )
    parser.add_argument(
        "output_folder",
        help="Path where organized images and labels will be saved"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_folder)
    if not input_path.exists():
        print(f"Error: Input folder '{input_path}' does not exist")
        return
    
    labeler = ImageLabeler(input_path, args.output_folder)
    labeler.run()


if __name__ == "__main__":
    main()