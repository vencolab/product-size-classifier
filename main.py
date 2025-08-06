import csv
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
import torch
import clip

# ---- CONFIG ----
PRODUCTS_DIR = Path(r"path\to\files")
OUTPUT_CSV = Path("bed_size_predictions.csv")
MODEL_VARIANT = "ViT-B/32"

# Bed size prompts with multiple descriptions per size, considering perspective/front view
labels = {
    "Small Single (2'6\")": [
        "a small single bed, 205cm long by 81cm wide, front view",
        "a small single bed, 205cm long by 81cm wide, perspective view",
        "a compact single bed, 2 feet 6 inches wide, front angle photo"
    ],
    "Single (3')": [
        "a single bed, 205cm long by 97cm wide, front view",
        "a classic single bed, 3 feet wide, perspective view",
        "a standard single bed measuring 205 by 97 centimeters, angled"
    ],
    "Small Double (4')": [
        "a small double bed, 205cm long by 127cm wide, front view",
        "a small double bed, 205cm long by 127cm wide, perspective photo",
        "a 4 feet wide double bed, front angle image"
    ],
    "Double (4'6\")": [
        "a double bed, 205cm long by 143cm wide, front view",
        "a double bed, 4 feet 6 inches wide, perspective view",
        "a standard double bed photo, angled front"
    ],
    "Kingsize (5')": [
        "a kingsize bed, 205cm long by 158cm wide, front view",
        "a king bed, 5 feet wide, perspective photo",
        "a large kingsize bed, front angle image"
    ],
    "Super King (6')": [
        "a super king bed, 205cm long by 188cm wide, front view",
        "a super king bed, 6 feet wide, perspective view",
        "a very large bed, super king size, front angle photo"
    ],
}

label_keys = list(labels.keys())

# ---- HELPER FUNCTIONS ----
def find_product_folders(base_folder: Path):
    """Find all folders that contain at least one image file."""
    product_folders = set()
    for file_path in base_folder.rglob("*"):
        if file_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            product_folders.add(file_path.parent)
    return list(product_folders)

def find_all_images(folder: Path):
    """Find all images in a folder and its subfolders."""
    return [f for f in folder.rglob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]

# ---- SETUP ----
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(MODEL_VARIANT, device=device)

# Flatten all prompts and map back to labels for prompt ensembling
all_prompts = []
prompt_to_label = []
for size, prompt_list in labels.items():
    for prompt in prompt_list:
        all_prompts.append(prompt)
        prompt_to_label.append(size)

text_inputs = clip.tokenize(all_prompts).to(device)

# Encode text prompts once (you can reuse this)
with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# ---- MAIN LOOP ----
results = []

print(f"🔍 Scanning product folders under: {PRODUCTS_DIR}")
product_folders = find_product_folders(PRODUCTS_DIR)
print(f"Found {len(product_folders)} product folders with images.\n")

for folder_path in product_folders:
    images = find_all_images(folder_path)
    if not images:
        print(f"⚠️ No images found in {folder_path.name}, skipping.")
        continue

    predictions = []

    for image_path in images:
        try:
            # Open and verify image
            with Image.open(image_path) as img:
                img.verify()

            # Preprocess & encode image
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                similarities = (image_features @ text_features.T).squeeze(0)

            # Aggregate similarity scores per label (prompt ensembling)
            label_scores = {}
            for idx, label in enumerate(prompt_to_label):
                label_scores.setdefault(label, []).append(similarities[idx].item())

            avg_scores = {label: np.mean(scores) for label, scores in label_scores.items()}
            best_label = max(avg_scores, key=avg_scores.get)
            confidence = avg_scores[best_label]

            predictions.append((best_label, confidence))

        except Exception as e:
            print(f"❌ Image error in {folder_path.name} - {image_path.name}: {e}")

    if predictions:
        # Aggregate predictions across images per folder
        conf_dict = {}
        for label, conf in predictions:
            conf_dict.setdefault(label, []).append(conf)
        avg_conf = {k: np.mean(v) for k, v in conf_dict.items()}
        final_label = max(avg_conf, key=avg_conf.get)
        final_conf = avg_conf[final_label]

        print(f"✅ {folder_path.name}: Predicted size: {final_label} (Avg confidence: {final_conf:.3f}, Images: {len(images)})")

        results.append({
            "product": folder_path.name,
            "predicted_size": final_label,
            "confidence": round(final_conf, 4),
            "num_images": len(images)
        })
    else:
        print(f"⚠️ No valid predictions for {folder_path.name}")

# ---- SAVE RESULTS ----
with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["product", "predicted_size", "confidence", "num_images"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"\n🎉 All done! Results saved to {OUTPUT_CSV}")
