# Image Processing — Assignment 3

A compact notebook-based implementation of basic image processing tasks (grayscale read, histogram equalization, morphological operations and a texture-based segmentation pipeline).

## Features
- Read images in grayscale
- Histogram equalization (from scratch)
- Morphological erosion, dilation and morphological gradient
- Texture segmentation and boundary extraction
- Saves results as image files (JPEG/PNG)

## Prerequisites
Requires Python 3 and the packages in `requirements.txt`.

## Quick setup (Linux / Bash)
```bash
# create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt

# run the notebook (executes all cells and writes outputs)
pip install nbconvert
jupyter nbconvert --to notebook --execute solution.ipynb --output executed_solution.ipynb

# or launch interactive notebook
jupyter notebook solution.ipynb
```

## Where to find outputs
Running the notebook will generate and save processed images in the repository root (filenames like `pic1_equalised.jpg`, `pic1_grad.jpg`, `pic1_textural_segmented.jpg`, etc.).

## Notes
- Place input images (`pic1.jpg`, `pic2.jpg`, `pic3.png`) in the repository root before running.
- The implementation is educational and written for clarity rather than maximum performance.