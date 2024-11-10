import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
import json

def evaluate():
    # Load the trained model
    solnet = load_model('models/solnet.hdf5', compile=False)
    
    # Load the history from the JSON file
    with open('history.json', 'r') as f:
        history = json.load(f)
    
    # Plot accuracy and loss over epochs
    plt.plot(history['loss'])
    plt.plot(history['acc'])
    plt.title('Accuracy and Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend(['Loss', 'Accuracy'], loc='upper left')
    plt.show()

if __name__ == "__main__":
    evaluate()