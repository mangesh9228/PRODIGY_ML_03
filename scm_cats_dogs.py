import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Dataset Path
dataset_path = "train"

images = []
labels = []

# Load Images
for file in os.listdir(dataset_path):

    if file.endswith(".jpg"):

        img_path = os.path.join(dataset_path, file)

        image = cv2.imread(img_path)
        image = cv2.resize(image, (64, 64))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        images.append(image.flatten())

        if file.startswith("cat"):
            labels.append(0)
        else:
            labels.append(1)

X = np.array(images)
y = np.array(labels)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train SVM Model
model = SVC(kernel="linear")
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Predict One Image
sample = cv2.imread(os.path.join(dataset_path, os.listdir(dataset_path)[0]))
sample = cv2.resize(sample, (64,64))
sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
sample = sample.flatten().reshape(1,-1)

prediction = model.predict(sample)

if prediction[0] == 0:
    print("\nPrediction: Cat")
else:
    print("\nPrediction: Dog")