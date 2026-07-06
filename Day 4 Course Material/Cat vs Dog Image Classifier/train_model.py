import os
import cv2
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression


# Dataset path
dataset_path = r"/content/sample_data/ImageDataset"

images = []
labels = []

classes = ["Cat", "Dog"]

IMG_SIZE = 64

for label, folder in enumerate(classes):

    folder_path = os.path.join(dataset_path, folder)

    for file in os.listdir(folder_path):

        img_path = os.path.join(folder_path, file)

        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        img = img.flatten()

        images.append(img)

        labels.append(label)


X = np.array(images)
y = np.array(labels)

print("Training Images", len(X))

model = LogisticRegression(max_iter=1000)

model.fit(X, y)

joblib.dump(model, "cat_dog_model.pkl")

print("Model Saved Successfully!")
