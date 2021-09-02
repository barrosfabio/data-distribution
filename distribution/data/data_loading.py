import pandas as pd
import numpy as np

def load_csv_data(path):
    # Loading the CSV Data
    data_frame = pd.read_csv(path)
    classes = data_frame['class']

    # Gathering the unique classes
    unique_classes = np.unique(classes)

    return [data_frame, unique_classes]