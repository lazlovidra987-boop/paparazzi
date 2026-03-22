import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

# ---------------------------
# 1. Read Image
# ---------------------------
img = Image.open("229433483.jpg").convert("L")
I = np.array(img, dtype=float)

# ---------------------------
# 2. Rotate 90° CCW (vectorized)
# ---------------------------
I_rot = np.rot90(I)

# ---------------------------
# 3. Crop Bottom Half
# ---------------------------
H = I_rot.shape[0]
I_crop = I_rot[H//2:, :]

# ---------------------------
# 4. Gaussian Kernel (vectorized)
# ---------------------------
def gaussian_kernel(size=5, sigma=1):
    ax = np.arange(-(size//2), size//2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    G = np.exp(-(xx**2 + yy**2) / (2*sigma**2))
    return G / G.sum()

K = gaussian_kernel(5, 1)

# ---------------------------
# 5. Convolution (vectorized)
# ---------------------------
def convolve(I, K):
    k = K.shape[0] // 2
    padded = np.pad(I, k, mode='constant')
    
    windows = sliding_window_view(padded, K.shape)
    return np.sum(windows * K, axis=(2, 3))

I_blur = convolve(I_crop, K)

# ---------------------------
# 6. Otsu Threshold (vectorized)
# ---------------------------
def otsu_threshold(I):
    hist, _ = np.histogram(I.ravel(), bins=256, range=(0,256))
    
    prob = hist / hist.sum()
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    
    mu_t = mu[-1]
    
    sigma_b = (mu_t * omega - mu)**2 / (omega * (1 - omega) + 1e-10)
    
    return np.argmax(sigma_b)

T = otsu_threshold(I_blur)
I_bin = (I_blur > T).astype(np.uint8) * 255

# ---------------------------
# 7. Dilation (vectorized)
# ---------------------------
def dilation(I):
    padded = np.pad(I, 1, mode='constant')
    windows = sliding_window_view(padded, (3,3))
    return np.max(windows, axis=(2,3))

I_dil = dilation(I_bin)

# ---------------------------
# 8. Visualization
# ---------------------------
titles = [
    "Original",
    "Rotated 90° CCW",
    "Bottom Half",
    "Gaussian Blur",
    "Otsu Threshold",
    "Dilation"
]

images = [
    I,
    I_rot,
    I_crop,
    I_blur,
    I_bin,
    I_dil
]

plt.figure(figsize=(12,8))

for i in range(len(images)):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.savefig("plot.png")