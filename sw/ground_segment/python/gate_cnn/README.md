# Gate CNN – Training and Deployment

This folder contains the code and dataset structure used to train a Convolutional Neural Network (CNN) for gate detection using a single camera.
The CNN predicts whether a gate is visible in the image and, if so, returns the four corner points of that gate.

The output of the network is designed to be simple so it can later be integrated into the Paparazzi flight code.

---

# Project Goal

The CNN receives a single image as input and outputs:

* **best_gate** (0–1): confidence that a gate is visible
* **4 corner points** of the gate (normalized image coordinates)

Output format:

best_gate
(u1, v1)
(u2, v2)
(u3, v3)
(u4, v4)

The corner points represent the four corners of the detected gate.

If **best_gate ≈ 0**, the system assumes **no gate is visible**.

The flight controller will later use the predicted corner points to compute:

* gate position in the image
* gate orientation
* gate distance (using Perspective-n-Point)

---

# Folder Structure

```
sw/ground_segment/python/gate_cnn/

├─ data/
│  ├─ images/
│  │  ├─ train/
│  │  ├─ val/
│  │  └─ test/
│  │
│  └─ labels/
│     ├─ train.json
│     ├─ val.json
│     └─ test.json
│
├─ src/
│  ├─ dataset.py
│  ├─ model.py
│  ├─ train.py
│  ├─ infer.py
│  └─ utils/
│     └─ order_corners.py
│
├─ export/
│  ├─ export_onnx.py
│  └─ export_to_c.py
│
├─ requirements.txt
└─ README.md
```

---

# Dataset

All images are stored in:

```
data/images/
```

Split into:

* **train/** → training images
* **val/** → validation images
* **test/** → evaluation images

Labels are stored in:

```
data/labels/
```

Each JSON file contains labels for the images in the corresponding dataset split.

Example label entry:

```
{
  "image": "images/train/frame_000123.jpg",
  "has_gate": 1,
  "corners": [[0.21,0.31],[0.60,0.30],[0.63,0.78],[0.18,0.80]]
}
```

Fields:

* **image** → path to image
* **has_gate** → 1 if a gate is visible, 0 otherwise
* **corners** → normalized (u,v) coordinates of the four gate corners

If no gate is visible:

```
{
  "image": "images/train/frame_000124.jpg",
  "has_gate": 0
}
```

---

# Source Code

All Python training code is located in:

```
src/
```

## dataset.py

Responsible for loading the dataset.

This file:

* reads image paths from the JSON label files
* loads images from disk
* loads the corresponding labels
* converts them to tensors
* returns samples to the training pipeline

It essentially connects:

image files + label JSON → training samples

---

## model.py

Defines the CNN architecture.

The model receives an image as input and outputs:

* gate confidence
* four corner coordinates

Typical output vector size:

```
[ best_gate,
  u1, v1,
  u2, v2,
  u3, v3,
  u4, v4 ]
```

Total outputs: **9 values**

---

## train.py

Main training script.

Responsibilities:

* load the training and validation datasets
* initialize the CNN model
* define the loss functions
* run training for multiple epochs
* periodically evaluate on the validation set
* save trained models to disk

Trained models are stored as:

```
model.pth
```

or inside a checkpoints folder.

---

## infer.py

Runs the trained model on new images or video.

Responsibilities:

* load a trained model
* read images or frames
* run the CNN
* output predicted gate confidence and corner locations
* optionally visualize predictions

Useful for debugging the detector before deploying it on the drone.

---

## utils/order_corners.py

Ensures the four predicted points follow a consistent order.

The desired order is:

1. top-left
2. top-right
3. bottom-right
4. bottom-left

Consistent ordering makes it easier to compute:

* gate center
* Perspective-n-Point pose estimation
* flight control inputs

---

# Model Export

The export folder contains scripts that convert the trained model into formats suitable for deployment.

## export_onnx.py

Converts the trained PyTorch model into **ONNX format**.

```
model.pth → model.onnx
```

ONNX can be used in many inference engines including C++ based ones.

---

## export_to_c.py

Optional script that converts the network weights into **C arrays**.

Example output:

```
float conv1_weights[] = {...};
```

This allows the CNN to run directly inside a C program if needed.

---

# Installation

Install Python dependencies with:

```
pip install -r requirements.txt
```

---

# Training

Run the training script:

```
python src/train.py
```

This will train the CNN and store the model weights.

---

# Inference

Run inference on a test image:

```
python src/infer.py --image example.jpg
```

This will print:

* gate confidence
* predicted corner points

---

# Integration with the Flight System

The CNN will eventually provide the following data to the flight controller:

* **best_gate**
* **4 corner coordinates**

From these values the flight controller can compute:

* gate position in the image
* gate orientation
* gate distance
* navigation commands

The CNN is therefore only responsible for **visual gate detection**.

All navigation logic remains in the flight control system.
