import torch
import numpy as np
import cv2
import torch.nn as nn
from hed_net import *
from torch.utils.data import Dataset
import os
import torch.optim as optim
import getopt
import numpy as np
import PIL
import PIL.Image
import sys
import matplotlib.pyplot as plt
import torchvision.transforms as T
from source.utils import *

torch.set_grad_enabled(False)
torch.backends.cudnn.enabled = True

img_in = './images/sample_3.png'
img_edges = './images/edges.png'
img_out = './images/img_out.png'

netNetwork = HedNetwork().eval()

def estimate(tenInput):
    global netNetwork

    if netNetwork is None:
        netNetwork = HedNetwork().eval()

    intWidth = tenInput.shape[2]
    intHeight = tenInput.shape[1]

    # assert(intWidth == 480) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
    # assert(intHeight == 320) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
    return netNetwork(tenInput.view(1, 3, intHeight, intWidth))[0, :, :, :].cpu()


tenInput = torch.FloatTensor(
    np.ascontiguousarray(
        np.array(PIL.Image.open(img_in).convert("RGB"))[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)
    )
)

tenOutput = estimate(tenInput)

PIL.Image.fromarray((tenOutput.clip(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, 0] * 255.0).astype(np.uint8)).save(img_edges)

tenOutput = (tenOutput.clip(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, 0] * 255.0).astype(np.uint8)
image_np = tenOutput

# Use contour detection to locate the central object without relying on segmentation
# Apply Gaussian Blur to smooth the image and reduce noise
blurred_image = cv2.GaussianBlur(image_np, (5, 5), 0)

# Detect edges using Canny Edge Detector
canny_edges = cv2.Canny(blurred_image, 50, 150)

# Find contours in the edge-detected image
contours, _ = cv2.findContours(canny_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Define center region for the cube (based on location in the initial image)
print(image_np.shape)
width = image_np.shape[1]
height = image_np.shape[0]
    
center_x, center_y = width // 2, height // 2
center_region_radius = min(width, height) // 5

# Filter contours based on proximity to center
filtered_contours = []
for cnt in contours:
    # Get the center of each contour
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        # Check if the contour center is within the central region
        if abs(cx - center_x) < center_region_radius and abs(cy - center_y) < center_region_radius:
            filtered_contours.append(cnt)

# Create an empty mask to draw filtered contours
contour_mask = np.zeros_like(image_np)
cv2.drawContours(contour_mask, filtered_contours, -1, (255), thickness=cv2.FILLED)

# Combine the contour mask with the original edges to isolate the cube
isolated_cube_edges = cv2.bitwise_and(image_np, contour_mask)

# Display the refined result
fig, ax = plt.subplots(1, 3, figsize=(12, 6))
ax[0].imshow(image_np, cmap='gray')
ax[0].set_title("Original Edge Detected Image")
ax[0].axis('off')

ax[1].imshow(isolated_cube_edges, cmap='gray')
ax[1].set_title("Contour-Based Isolation of Cube Edges")
ax[1].axis('off')

center_x, center_y, angle = calculate_pca_rotation_angle_from_edge_image(isolated_cube_edges)

# Draw the orientation arrow
arrow_length = 50
end_x = int(center_x - arrow_length * np.cos(np.deg2rad(angle)))
end_y = int(center_y - arrow_length * np.sin(np.deg2rad(angle)))

output_image = cv2.cvtColor(isolated_cube_edges, cv2.COLOR_GRAY2RGB)
cv2.arrowedLine(output_image, (int(center_y), int(center_x)), (end_y, end_x), (0, 255, 0), 2, tipLength=0.3)

# Display the final image with orientation arrow
plt.imshow(output_image, cmap='gray')
plt.title("Object Orientation")
plt.axis('off')

plt.savefig(img_out, bbox_inches='tight', dpi=300)

print("angle = ", angle)

