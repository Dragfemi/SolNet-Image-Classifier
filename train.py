import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import binary_crossentropy
from sklearn.model_selection import cross_val_score
from model import SolNet  # Assuming SolNet is a class or function that builds your model
import json

def train():
    batch_size = 32
    location = "\dataset"
    label_mode = 'binary'
    seed = 10  # Changed for each fold made manually
    epochs = 30
    class_names = ['clean', 'dirty']
    
    # Define in_size before using it
    in_size = [227, 227, 3]
    
    tr_dataset = image_dataset_from_directory(
        directory=location, label_mode=label_mode, class_names=class_names,
        seed=seed, labels='inferred', image_size=in_size[:-1], 
        subset='training', batch_size=batch_size, validation_split=0.2
    )

    val_dataset = image_dataset_from_directory(
        directory=location, label_mode=label_mode, class_names=class_names,
        seed=seed, labels='inferred', image_size=in_size[:-1],
        subset='validation', batch_size=batch_size, validation_split=0.2
    )

    # Instantiate the model using a different variable name
    model = SolNet(in_size)
    
    # Train the model
    history = model.fit(tr_dataset, validation_data=val_dataset, epochs=epochs, batch_size=batch_size)
    # Save the history to a JSON file for later use
    with open('history.json', 'w') as f:
        json.dump(history.history, f)
    
    # Save the trained model
    model.save('models/solnet.hdf5')
    return history

if __name__ == "__main__":
    train()
