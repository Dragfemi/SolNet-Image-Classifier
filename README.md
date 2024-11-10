# SolNet-Image-Classifier
This project implements a neural network-based model called SolNet to classify images into binary categories (e.g., "clean" or "dirty"). The model architecture, training, evaluation, and history tracking are organized across multiple scripts and files as described below.

Folder Contents
1. evaluate.py
This script is used to evaluate the performance of the trained SolNet model. It loads the trained model and visualizes its accuracy and loss over the training epochs.

Key Components:
Model Loading: Uses tensorflow.keras.models.load_model to load the saved SolNet model (solnet.hdf5).
History Loading: Reads training history from history.json (a file containing metrics for each epoch, such as loss and accuracy).
Plotting: Utilizes matplotlib to plot the accuracy and loss values across epochs, allowing for visual inspection of model performance over time.
Usage:
Run evaluate.py as a standalone script to view the accuracy and loss plots:

2. model.py
This script defines the architecture of the SolNet model, a custom convolutional neural network (CNN) for binary image classification. The model uses several convolutional, pooling, normalization, and dense layers to achieve feature extraction and classification.

Key Components:
Model Architecture:
Convolutional Layers: Extract spatial features with filters of varying sizes.
Batch Normalization: Stabilizes and accelerates training.
Max Pooling: Reduces the spatial size of features, making computations more efficient.
Dense Layers: After flattening, fully connected layers help learn complex representations.
Dropout: Adds regularization to prevent overfitting.
Compilation: The model is compiled with the Adam optimizer and binary cross-entropy loss.
Model Saving: Saves the constructed model as solnet.hdf5.
Usage:
The model can be built and saved by running model.py as a standalone script:

3. train.py
This script handles training the SolNet model on a dataset of labeled images, including steps for loading data, training the model, and saving training metrics to a JSON file for later evaluation.

Key Components:
Data Loading: Uses image_dataset_from_directory to load and preprocess images, with label_mode='binary' for binary classification. It splits data into training and validation sets with an 80-20 ratio.
Model Training: Trains the SolNet model on the loaded dataset for a specified number of epochs.
History Saving: Saves training history (accuracy, loss for both training and validation) as history.json, allowing evaluate.py to access and plot these metrics later.
Model Saving: Saves the trained model weights and configuration to solnet.hdf5.
Usage:
Run train.py to train the model on your dataset:

4. history.json
This file stores the training history, including metrics for each epoch. It contains:

acc: List of training accuracy values per epoch.
loss: List of training loss values per epoch.
val_acc: Validation accuracy per epoch, used to evaluate generalization.
val_loss: Validation loss per epoch, tracking how well the model generalizes to unseen data.
The evaluate.py script reads this file to generate accuracy and loss plots, aiding in model evaluation.

