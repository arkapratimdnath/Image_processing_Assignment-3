Repository: arkapratimdnath/image_processing_assignment-3
Files analyzed: 2

Estimated tokens: 2.0k

Directory structure:
└── arkapratimdnath-image_processing_assignment-3/
    ├── requirements.txt
    └── solution.ipynb


================================================
FILE: requirements.txt
================================================
numpy
matplotlib
ipykernel
Pillow


================================================
FILE: solution.ipynb
================================================
# Jupyter notebook converted to Python script.

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

"""
# PART-I (1)
"""

def grayread(image_path):
    img=Image.open(image_path).convert("L")
    img=np.array(img,dtype=np.uint8)
    return img

def display_image(image):
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.show()

pic1=grayread("pic1.jpg")
display_image(pic1)
# Output:
#   <Figure size 640x480 with 1 Axes>

pic2=grayread("pic2.jpg")
display_image(pic2)
# Output:
#   <Figure size 640x480 with 1 Axes>

pic3=grayread("pic3.png")
display_image(pic3)
# Output:
#   <Figure size 640x480 with 1 Axes>

"""
# PART-I (2)
"""

def hist_equalization_scratch(image):
    hist, bins = np.histogram(image.flatten(), bins=256, range=[0,256])
    hist_normalised=hist/hist.sum()
    cdf = hist_normalised.cumsum()
    img_copy=image.copy()
    for i in range(img_copy.shape[0]):
        for j in range(img_copy.shape[1]):
            img_copy[i,j]=round(255*cdf[img_copy[i,j]])
    return img_copy


pic1_equalized=hist_equalization_scratch(pic1)
plt.title("Histogran Equalized Image")
plt.imshow(pic1_equalized, cmap='gray', vmin=0, vmax=255)
plt.savefig("pic1_equalised.jpg")
plt.show()
# Output:
#   <Figure size 640x480 with 1 Axes>

pic2_equalized=hist_equalization_scratch(pic2)
plt.title("Histogran Equalized Image")
plt.imshow(pic2_equalized, cmap='gray', vmin=0, vmax=255)
plt.savefig("pic2_equalised.jpg")
plt.show()
# Output:
#   <Figure size 640x480 with 1 Axes>

pic3_equalized=hist_equalization_scratch(pic3)

plt.title("Histogran Equalized Image")
plt.imshow(pic3_equalized, cmap='gray', vmin=0, vmax=255)
plt.savefig("pic3_equalised.jpg")
plt.show()
# Output:
#   <Figure size 640x480 with 1 Axes>

"""
# PART-I (3)
"""

"""
#### Erossion
"""

def erosion_scratch(img, kernel_size):
    pad = kernel_size // 2
    padded_img = np.pad(img, pad, mode='edge')
    eroded = np.zeros_like(img)
    structuring_element=np.ones((kernel_size, kernel_size), dtype=np.uint8)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded_img[i:i+kernel_size, j:j+kernel_size]-structuring_element
            eroded[i, j] = np.min(region)
    return eroded

"""
#### Dilation
"""

def dilation_scratch(img, kernel_size):
    pad = kernel_size // 2
    padded_img = np.pad(img, pad, mode='edge')
    dilated = np.zeros_like(img)
    structuring_element=np.ones((kernel_size, kernel_size), dtype=np.uint8)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded_img[i:i+kernel_size, j:j+kernel_size]+structuring_element
            dilated[i, j] = np.max(region)
    return dilated

pic1_erosion=erosion_scratch(pic1_equalized,3)
plt.title("Erosion Image")
plt.imshow(pic1_erosion, cmap='gray', vmin=0, vmax=255)
plt.savefig("pic1_erosion.jpg")
plt.show()
# Output:
#   <Figure size 640x480 with 1 Axes>

pic1_dilation=dilation_scratch(pic1_equalized,3)
plt.title("Dilation Image")
plt.imshow(pic1_dilation, cmap='gray', vmin=0, vmax=255)
plt.savefig("pic1_dilation.jpg")
plt.show()
# Output:
#   <Figure size 640x480 with 1 Axes>

pic1_grad=pic1_dilation - pic1_erosion
plt.title("Morphological Gradient Image")
plt.imshow(pic1_grad, cmap='gray', vmin=0, vmax=255)
plt.savefig("pic1_grad.jpg")
plt.show()
# Output:
#   <Figure size 640x480 with 1 Axes>

pic2_erosion=erosion_scratch(pic2_equalized,3)
plt.title("Erosion Image")
plt.imshow(pic2_erosion, cmap='gray', vmin=0, vmax=255)
plt.savefig("pic2_erosion.jpg")
plt.show()
# Output:
#   <Figure size 640x480 with 1 Axes>

pic2_dilation=dilation_scratch(pic2_equalized,3)
plt.title("Dilation Image")
plt.imshow(pic2_dilation, cmap='gray', vmin=0, vmax=255)
plt.savefig("pic2_dilation.jpg")
plt.show()
# Output:
#   <Figure size 640x480 with 1 Axes>

pic2_grad=pic2_dilation - pic2_erosion
plt.title("Morphological Gradient Image")
plt.imshow(pic2_grad, cmap='gray', vmin=0, vmax=255)
plt.savefig("pic2_grad.jpg")
plt.show()
# Output:
#   <Figure size 640x480 with 1 Axes>

pic3_erosion=erosion_scratch(pic3_equalized,3)
plt.title("Erosion Image")  
plt.imshow(pic3_erosion, cmap='gray', vmin=0, vmax=255)
plt.savefig("pic3_erosion.jpg")
plt.show()
# Output:
#   <Figure size 640x480 with 1 Axes>

pic3_dilation=dilation_scratch(pic3_equalized,3)
plt.title("Dilation Image")
plt.imshow(pic3_dilation, cmap='gray', vmin=0, vmax=255)
plt.savefig("pic3_dilation.jpg")
plt.show()
# Output:
#   <Figure size 640x480 with 1 Axes>

pic3_grad= pic3_dilation - pic3_erosion
plt.title("Morphological Gradient Image")
plt.imshow(pic3_grad, cmap='gray', vmin=0, vmax=255)
plt.savefig("pic3_grad.jpg")
plt.show()
# Output:
#   <Figure size 640x480 with 1 Axes>

"""
# PART-II (Group-C)
"""

def textural_segmentation(img, radius):
    results = []

    opened = dilation_scratch(erosion_scratch(img, radius), radius)
    closed = erosion_scratch(dilation_scratch(img, radius), radius)
    texture = closed - opened
    results.append(texture.astype(np.uint8))
    texture_map = np.max(np.stack(results, axis=-1), axis=-1)
    return texture_map

pic1_textural_segmented = textural_segmentation(pic1_grad, 3)
plt.title("Textural Segmented Image")
plt.imshow(pic1_textural_segmented, cmap='gray', vmin=0, vmax=255)
plt.savefig("pic1_textural_segmented.jpg")
plt.show()
# Output:
#   <Figure size 640x480 with 1 Axes>

pic1_boundary= dilation_scratch(pic1_textural_segmented,3) - erosion_scratch(pic1_textural_segmented,3)
plt.title("Boundary Image")
plt.imshow(pic1_boundary, cmap='gray', vmin=0, vmax=255)
plt.savefig("pic1_boundary.jpg")
plt.show()
# Output:
#   <Figure size 640x480 with 1 Axes>

pic2_textural_segmented = textural_segmentation(pic2_grad, 3)
plt.title("Textural Segmented Image")
plt.imshow(pic2_textural_segmented, cmap='gray', vmin=0, vmax=255)
plt.savefig("pic2_textural_segmented.jpg")
plt.show()
# Output:
#   <Figure size 640x480 with 1 Axes>

pic2_boundary= dilation_scratch(pic2_textural_segmented,3) - erosion_scratch(pic2_textural_segmented,3)
plt.title("Boundary Image")
plt.imshow(pic2_boundary, cmap='gray', vmin=0, vmax=255)
plt.savefig("pic2_boundary.jpg")
plt.show()
# Output:
#   <Figure size 640x480 with 1 Axes>

pic3_textural_segmented = textural_segmentation(pic3_grad, 3)
plt.title("Textural Segmented Image")
plt.imshow(pic3_textural_segmented, cmap='gray', vmin=0, vmax=255)
plt.savefig("pic3_textural_segmented.jpg")
plt.show()
# Output:
#   <Figure size 640x480 with 1 Axes>

pic3_boundary= dilation_scratch(pic3_textural_segmented,3) - erosion_scratch(pic3_textural_segmented,3)
plt.title("Boundary Image")
plt.imshow(pic3_boundary, cmap='gray', vmin=0, vmax=255)
plt.savefig("pic3_boundary.jpg")
plt.show()
# Output:
#   <Figure size 640x480 with 1 Axes>

