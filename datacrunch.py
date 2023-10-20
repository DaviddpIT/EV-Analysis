# import packages
import pandas as pd
import numpy as np

# Read data from text file
dataset = pd.read_csv("1_h_01.txt", sep="\t", header=1)

# Convert string to floating point number
dataset['mm'] = dataset['mm'].str.replace(',', '.').astype(float)
print(dataset)

# Convert string to datetime object
dataset["Datum data"] = pd.to_datetime(dataset["Datum data"], dayfirst=True)