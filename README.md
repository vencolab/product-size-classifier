#  Product Size Classifier

This project automatically classifies bed images into size categories (e.g. Single, Double, King) using OpenAI's [CLIP](https://openai.com/research/clip) model — **no custom training, no labelling, no API calls** required. Just drop your product folders in and let the model do the magic.

---

## What It Does

- Crawls through a nested directory of product folders
- Detects all images in each product folder
- Uses CLIP to compare the image(s) to natural language descriptions of bed sizes
- Averages the similarity scores across all images to make a final prediction
- Outputs results to a CSV file (for easy review or integration)

---

## Folder Structure

Place your products inside the `PRODUCTS_DIR` folder like so:
```
 Directory/
├── Product123/
│ ├── image1.jpg
│ └── image2.png
├── Product456/
│ └── another_image.jpeg
```

---

## Currently Supported Bed Sizes 

| Size Name           | Dimensions (cm)         |
|---------------------|-------------------------|
| Small Single (2'6") | 205 x 81                |
| Single (3')         | 205 x 97                |
| Small Double (4')   | 205 x 127               |
| Double (4'6")       | 205 x 143               |
| Kingsize (5')       | 205 x 158               |
| Super King (6')     | 205 x 188               |

---

## How to Run

You can use either conda or pip.

1. **Clone the repo and set up your  environment**

```bash

cd product-size-classifier
conda env create -f environment.yml
conda activate bed_classify

```
or 

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

2. **Update the config**

Open main.py and update the following line to point to your folder of product images:

python
Copied
Edit
PRODUCTS_DIR = "C:\\your\\path\\to\\products"

3. **Run it**


```
python main.py
```
---
##  Modifications & Customisation
Want to tweak it for your use case? Here’s how:

### Add More Bed Sizes
Open the labels dictionary in main.py and add more descriptions, e.g.:

```
"XL Super King (7')": "an extra large super king bed, 205cm long x 210cm wide"
```
--

## How It Works
Uses CLIP in zero-shot mode:

Convert text descriptions of bed sizes into embeddings

Convert product images into embeddings

Compare image ↔ text using cosine similarity

Select the bed size with the highest match score

---

## Accuracy Tips
Use multiple images per product for better predictions

Crop or curate out-of-context photos (e.g. with excessive background or clutter)

Add prompt variations, e.g.:

"a double bed, 205cm long x 143cm wide, viewed from front"

"a 143cm wide bed without a large headboard"

---

## No External APIs
This runs 100% locally using:

CLIP (ViT-B/32)

PIL

PyTorch

No cloud. No cost. No training required.