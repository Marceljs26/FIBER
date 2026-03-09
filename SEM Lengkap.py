import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
from scipy import ndimage as ndi

# =========================
# LOAD IMAGE
# =========================
image_path = "S2 5K Measure_processed.jpg"   # ganti sesuai file anda
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# =========================
# PREPROCESSING
# =========================
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe_gray = clahe.apply(gray)
blur = cv2.GaussianBlur(clahe_gray, (5,5), 0)
th = threshold_otsu(blur)
binary = blur < th
# Morphological cleaning
kernel = np.ones((3,3), np.uint8)
binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_OPEN, kernel)
binary = binary.astype(bool)
 # Sharpening filter
# Sharpening filter
sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
sharpened = cv2.filter2D(clahe_gray, -1, sharpening_kernel)

# =========================
# SKELETONIZATION
# =========================
skeleton = skeletonize(binary)

# =========================
# LABELING
# =========================
labeled = label(binary)
regions = regionprops(labeled)


# =========================
# (Orientation map dihapus sesuai permintaan)


# =========================
# THICKNESS MAP
# =========================
distance = ndi.distance_transform_edt(binary)
thickness = distance * 2
fiber_diameter = thickness[skeleton]

# =========================
# EDGE DETECTION
# =========================
median_val = np.median(clahe_gray)
edge_thresh1 = int(max(0, 0.66 * median_val))
edge_thresh2 = int(min(255, 1.33 * median_val))
edges_canny = cv2.Canny(clahe_gray, edge_thresh1, edge_thresh2)
edges_canny_sharp = cv2.Canny(sharpened, edge_thresh1, edge_thresh2)
edges_sobel = cv2.Sobel(clahe_gray, cv2.CV_64F, 1, 1, ksize=3)
edges_sobel = np.uint8(np.absolute(edges_sobel))
edges_laplacian = cv2.Laplacian(clahe_gray, cv2.CV_64F)
edges_laplacian = np.uint8(np.absolute(edges_laplacian))
overlay = img.copy()
overlay[edges_canny != 0] = [255, 0, 0]
overlay_sharp = img.copy()
overlay_sharp[edges_canny_sharp != 0] = [0, 0, 255]

# =========================
# STATISTICS
# =========================

print("===== FIBER STATISTICS =====")
print("Mean diameter (pixel):", np.mean(fiber_diameter))
print("Std diameter (pixel):", np.std(fiber_diameter))

# =========================
# VISUALISASI TANPA ORIENTATION
# =========================
fig, axes = plt.subplots(2, 5, figsize=(18, 8))
axes = axes.ravel()
# 1. Original
axes[0].imshow(gray, cmap="gray")
axes[0].set_title("Original")
axes[0].axis('off')
# 2. Binary
axes[1].imshow(binary, cmap="gray")
axes[1].set_title("Binary")
axes[1].axis('off')
# 3. Skeleton
axes[2].imshow(skeleton, cmap="gray")
axes[2].set_title("Skeleton")
axes[2].axis('off')
# 4. Thickness Map
im4 = axes[3].imshow(thickness, cmap="jet")
axes[3].set_title("Thickness Map")
axes[3].axis('off')
cbar4 = fig.colorbar(im4, ax=axes[3], fraction=0.046, pad=0.04)
cbar4.set_label('Pixel')
# 5. Canny Edge Overlay
axes[4].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
axes[4].set_title("Canny Edge Overlay")
axes[4].axis('off')
# 6. Canny Edge Overlay (Sharpened)
axes[5].imshow(cv2.cvtColor(overlay_sharp, cv2.COLOR_BGR2RGB))
axes[5].set_title("Canny Edge Sharpened")
axes[5].axis('off')
# 7. Sobel Edge
axes[6].imshow(edges_sobel, cmap="gray")
axes[6].set_title("Sobel Edge")
axes[6].axis('off')
# 8. Laplacian Edge
axes[7].imshow(edges_laplacian, cmap="gray")
axes[7].set_title("Laplacian Edge")
axes[7].axis('off')
# 9. Contour Overlay
contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_overlay = img.copy()
cv2.drawContours(contour_overlay, contours, -1, (0,255,0), 1)
axes[8].imshow(cv2.cvtColor(contour_overlay, cv2.COLOR_BGR2RGB))
axes[8].set_title("Contour Overlay")
axes[8].axis('off')
plt.tight_layout()
plt.show()


# (Histogram orientation dihapus)
