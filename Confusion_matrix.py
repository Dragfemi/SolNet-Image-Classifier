import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf

# Load the SolNet model
model = tf.keras.models.load_model('models/solnet.hdf5')

# Load your test dataset (replace with the actual dataset paths)
# Assuming the test dataset is stored in a directory structure compatible with image_dataset_from_directory
test_data = tf.keras.preprocessing.image_dataset_from_directory(
    'path/to/test_dataset',  # Replace with the path to your test dataset
    image_size=(227, 227),  # Match the input size of SolNet
    batch_size=32,  # Adjust batch size as needed
    shuffle=False  # Ensure images are not shuffled for alignment with labels
)

# Extract images and labels from the dataset
test_images = []
test_labels = []

for images, labels in test_data:
    test_images.append(images.numpy())
    test_labels.append(labels.numpy())

test_images = np.concatenate(test_images, axis=0)
test_labels = np.concatenate(test_labels, axis=0)

# Generate predictions
predictions = model.predict(test_images)

# Convert probabilities to binary class predictions (threshold = 0.5)
predicted_classes = (predictions > 0.5).astype(int)

# Generate confusion matrix
cm = confusion_matrix(test_labels, predicted_classes)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Faulty'])
disp.plot()

# Show the plot
import matplotlib.pyplot as plt
plt.show()
