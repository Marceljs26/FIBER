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
image_path = "S8 5K_processed.jpg"   # ganti sesuai file anda
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
# Perhalus sharpening dengan Gaussian blur sebelum Canny
sharpened_blur = cv2.GaussianBlur(sharpened, (3,3), 0)
edges_canny_sharp = cv2.Canny(sharpened_blur, edge_thresh1, edge_thresh2)
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
plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(overlay_sharp, cv2.COLOR_BGR2RGB))
plt.title("Canny Edge Sharpened")
plt.axis('off')
plt.show()


# (Histogram orientation dihapus)
