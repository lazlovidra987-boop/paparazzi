"""
Color Classifier using scikit-learn
========================================================
Trains Decision Tree classifiers on YUV color data from labeled images.

Features
--------
- Arrow-key scrollable image viewer (← / →) in a single figure
- Press R to rotate the current image 90 degrees clockwise
- ROC scatter plot: one point per depth in DEPTH_SWEEP
  · X-axis = False Positive Rate: fraction of non-ground pixels wrongly
             called ground  (want this LOW)
  · Y-axis = True Positive Rate:  fraction of actual ground pixels correctly
             detected  (want this HIGH)
  -> The ideal classifier sits in the top-left corner.
  -> Use the plot to pick the shallowest depth that still gives a good result:
     a depth-d tree needs at most 2^d - 1 comparisons per pixel in C, so
     depth 3 costs <= 7 comparisons while depth 10 costs <= 1023.
- DISPLAY_DEPTH controls which trained tree is printed and shown in the viewer.

Usage:
    Place your images in a ./data/ folder:
      - Image files must end with 'c.jpg'   (e.g. 1c.jpg)
      - Mask files must end with 'm.jpg'    (e.g. 1m.jpg, same base name)
    Then run:
        python ground_classifier.py

Requirements:
    pip install opencv-python numpy scikit-learn matplotlib
"""

import cv2
import numpy as np
import glob
import random
from random import randrange
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt


# ── Configuration ─────────────────────────────────────────────────────────────

random.seed(42)

DATA_DIR          = "./data"      # datasim for gazebo
SAMPLES_PER_IMAGE = 75_000
MAX_FILES         = 500
RANDOM_STATE      = 0
TEST_SIZE         = 0.2

# Depths to sweep for the ROC curve
DEPTH_SWEEP       = [1, 2, 3, 4, 5, 6, 8, 10]

# Which depth to use for the image viewer. Must be in DEPTH_SWEEP.
DISPLAY_DEPTH     = 5

# Set to True to print the decision tree for DISPLAY_DEPTH to the terminal.
PRINT_TREE        = True


# ── Data loading ──────────────────────────────────────────────────────────────

def load_training_data(data_dir, samples_per_image, max_files):
    """Read image/mask pairs and build feature (X) and label (y) vectors."""
    images = sorted(glob.glob(f"{data_dir}/*c.jpg"))
    labels = sorted(glob.glob(f"{data_dir}/*m.jpg"))

    print(f"Found {len(images)} image(s): {images}")

    X_vec, y_vec = [], []
    files_remaining = max_files

    for f in images:
        lf = f.replace("c.jpg", "m.jpg")
        if lf not in labels:
            continue

        files_remaining -= 1
        if files_remaining <= 0:
            break

        img = cv2.imread(f)
        msk = cv2.imread(lf)
        h, w, _ = img.shape
        print(f"  img={f}  lbl={lf}  {w}x{h}")

        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

        for _ in range(samples_per_image):
            x = randrange(2, w - 3)
            y = randrange(4, h - 2)

            p = yuv[y, x]
            m = int(msk[y, x, 0])
            m = 0 if m < 127 else 255

            X_vec.append([int(p[0]), int(p[1]), int(p[2])])
            y_vec.append(m)

    print(f"Dataset size: {len(X_vec)} samples")
    return X_vec, y_vec


# ── Classifier helpers ────────────────────────────────────────────────────────

def train_classifier(X_train, y_train, max_depth, random_state=RANDOM_STATE):
    """Train and return a Decision Tree classifier."""
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    dt.fit(X_train, y_train)
    return dt


def tpr_fpr(dt, X_test, y_test):
    """Return (TPR, FPR) for the positive class (label == 255)."""
    y_pred = dt.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 255]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return tpr, fpr


def export_tree(dt, feature_names=("Y", "U", "V")):
    """Print a human-readable text representation of the decision tree."""
    text = export_text(dt, feature_names=list(feature_names))
    print("\nDecision tree:\n")
    print(text)



# ── Build classified overlays ─────────────────────────────────────────────────

def build_overlays(dt, image_paths):
    """
    Return a list of RGB overlay images (classified green-channel overlay)
    and the corresponding file names.
    """
    overlays, names = [], []
    for f in image_paths:
        img = cv2.imread(f)
        if img is None:
            continue
        h, w, d = img.shape
        yuv    = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        X_run  = yuv.reshape(h * w, d)
        y_pred = dt.predict(X_run).reshape(h, w)

        overlay = img.copy()
        overlay[:, :, 1] = y_pred
        overlays.append(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        names.append(f)
    return overlays, names


# ── ROC figure ────────────────────────────────────────────────────────────────

def show_roc(roc_points):
    """Open a separate figure with the ROC scatter plot."""
    fig, ax = plt.subplots(figsize=(6, 6))

    fprs   = [p[0] for p in roc_points]
    tprs   = [p[1] for p in roc_points]
    depths = [p[2] for p in roc_points]

    sc = ax.scatter(fprs, tprs, c=depths, cmap="plasma",
                    s=80, zorder=3, edgecolors="k", linewidths=0.5)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("max_depth", fontsize=9)

    """
    for fpr, tpr, d in roc_points:
        ax.annotate(f"d={d}", (fpr, tpr),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
                    
    """

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False Positive Rate\n(non-ground wrongly called ground)")
    ax.set_ylabel("True Positive Rate\n(ground pixels correctly detected)")
    ax.set_title("ROC — depth sweep\n(top-left = ideal)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()


# ── Interactive scrollable viewer ─────────────────────────────────────────────

class ScrollableViewer:
    """
    Show one image at a time in its own figure; use ← / → (or A / D) to navigate.
    Press R to rotate the current image 90 degrees clockwise.
    Each image remembers its own rotation independently.
    """

    def __init__(self, overlays, names, display_depth):
        self.overlays      = overlays
        self.names         = names
        self.display_depth = display_depth
        self.idx           = 0
        self.rotations     = [0] * len(overlays)  # per-image clockwise degrees

        self.fig, self.ax_img = plt.subplots(figsize=(8, 6))
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self._show_image()
        plt.tight_layout()
        plt.show()

    def _show_image(self):
        ax  = self.ax_img
        ax.clear()
        img = self.overlays[self.idx]

        # np.rot90 is counter-clockwise, so convert clockwise steps
        k_cw = (self.rotations[self.idx] // 90) % 4
        if k_cw:
            img = np.rot90(img, k=4 - k_cw)

        ax.imshow(img)
        total = len(self.overlays)
        rot   = self.rotations[self.idx]
        ax.set_title(
            f"{self.names[self.idx]}\n"
            f"Image {self.idx + 1} / {total}  |  max_depth={self.display_depth}"
            f"  |  rotation {rot}°\n"
            f"(← / → navigate,  R rotate 90° clockwise)",
            fontsize=9,
        )
        ax.axis("off")
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        if event.key in ("right", "d"):
            prev_idx = self.idx
            self.idx = (self.idx + 1) % len(self.overlays)
            self.rotations[self.idx] = self.rotations[prev_idx]
            self._show_image()
        elif event.key in ("left", "a"):
            prev_idx = self.idx
            self.idx = (self.idx - 1) % len(self.overlays)
            self.rotations[self.idx] = self.rotations[prev_idx]
            self._show_image()
        elif event.key in ("r", "R"):
            self.rotations[self.idx] = (self.rotations[self.idx] + 90) % 360
            self._show_image()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # 1. Load data
    X_vec, y_vec = load_training_data(DATA_DIR, SAMPLES_PER_IMAGE, MAX_FILES)

    if not X_vec:
        print("No training data found. "
              "Make sure images (*c.jpg) and masks (*m.jpg) exist in ./data/")
        return

    # 2. Train/test split (done once; same split for all depths)
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y_vec,
        test_size=TEST_SIZE,
        stratify=y_vec,
        random_state=1,
    )
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}\n")

    # 3. Sweep over depth values → ROC points, store all trained classifiers
    roc_points   = []
    classifiers  = {}   # depth -> trained DecisionTreeClassifier

    print(f"{'depth':>6}  {'detected ground':>16}  {'false alarms':>13}  {'max C comparisons':>18}")
    print("-" * 62)
    for depth in DEPTH_SWEEP:
        dt       = train_classifier(X_train, y_train, max_depth=depth)
        tpr, fpr = tpr_fpr(dt, X_test, y_test)
        roc_points.append((fpr, tpr, depth))
        classifiers[depth] = dt
        max_comparisons = 2 ** depth - 1
        print(f"{depth:>6}  {tpr:>15.1%}  {fpr:>12.1%}  {max_comparisons:>18}")

    print()

    if DISPLAY_DEPTH not in classifiers:
        raise ValueError(f"DISPLAY_DEPTH={DISPLAY_DEPTH} is not in DEPTH_SWEEP={DEPTH_SWEEP}")

    display_dt = classifiers[DISPLAY_DEPTH]

    # 4. Optionally print the tree for DISPLAY_DEPTH
    if PRINT_TREE:
        print(f"Decision tree for max_depth={DISPLAY_DEPTH}:")
        export_tree(display_dt)

    # 5. Build overlays using the DISPLAY_DEPTH classifier
    image_paths = sorted(glob.glob(f"{DATA_DIR}/*c.jpg"))
    if not image_paths:
        print("No images found for visualisation.")
        return

    print(f"Building overlays for {len(image_paths)} image(s) "
          f"(max_depth={DISPLAY_DEPTH}) …")
    overlays, names = build_overlays(display_dt, image_paths)

    # 6. Show ROC in its own figure, then open the scrollable image viewer
    show_roc(roc_points)
    ScrollableViewer(overlays, names, DISPLAY_DEPTH)
    plt.show()   # blocks until both windows are closed


if __name__ == "__main__":
    main()