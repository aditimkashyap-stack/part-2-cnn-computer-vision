# Part‑2: CNN Computer Vision Prototype — Manufacturing Defect Classification

Dataset type: https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs?usp=sharing

##  Problem Statement
We are given a synthetic manufacturing defect image dataset.  
The task is to build a CNN‑based prototype to classify product images into one of four classes: **normal, scratch, dent, stain**.

---

##  Task 1: Problem Identification
- **Problem type:** Image Classification  
- **Reason:** Each image belongs to exactly one defect class. The goal is to assign the correct label.

---

##  Task 2: Dataset Exploration
- **Classes:** 4 (normal, scratch, dent, stain)  
- **Images per class:** 120 each (balanced dataset)  
- **Total images:** 480  
- **Image dimensions:** ~128×128 pixels (synthetic dataset)  
- **Imbalance:** None — dataset is balanced.

---

##  Task 3: Image Preprocessing
- Resized all images to **128×128**.  
- Normalized pixel values to [0,1].  
- Train/test split: 80/20.  
- Augmentation: rotation, flips, zoom for robustness.

---

##  Task 4: CNN Model Creation
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(4, activation='softmax')
])
