# import packages
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read data from text file
dataset = pd.read_csv("1_h_01.txt", sep="\t", header=1)
dataset['mm'] = dataset["mm"].str.replace(',', '.')
dataset['mm'] = dataset["mm"].astype(float)

# Plot the data
dataset.plot()
plt.show()

# Calculate sample mean and sample variance
sample_mean = dataset['mm'].mean()
sample_var = dataset['mm'].var()

print(f'the sample mean is {sample_mean:.2f}')
print(f'the sample variance is {sample_var:.2f}')

# Apply the Methods of Moments to calculare the two parameters of the Gumbel distribution
eulergamma = 0.57721566490
a = math.sqrt(6 * sample_var) / math.pi
u = sample_mean - eulergamma * a

print(f"parameter a: {a:.2f} , parameter u: {u:.2f}")