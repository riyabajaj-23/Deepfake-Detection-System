import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

IMG_SIZE = 224
BATCH_SIZE = 32

model = load_model("deepfake_model_final.h5")

test_dir = "dataset/test"

datagen = ImageDataGenerator(rescale=1./255)

test_data = datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

probs = model.predict(test_data)

predictions = (probs > 0.5).astype(int)

print("\nAccuracy:", accuracy_score(test_data.classes, predictions))

print("\nClassification Report:")
print(classification_report(test_data.classes, predictions))

cm = confusion_matrix(test_data.classes, predictions)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Fake","Real"],
            yticklabels=["Fake","Real"])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()