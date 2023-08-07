# import packages
import math
import pandas as pd
import numpy as np
# import matplotlib as mplt
import matplotlib.pyplot as plt
import scipy

# Read data from text file
dataset = pd.read_csv("1_h_01.txt", sep="\t", header=1)

# Convert string to floating point number
dataset['mm'] = dataset["mm"].str.replace(',', '.')
dataset['mm'] = dataset["mm"].astype(float)
print(dataset)

# Convert string to datetime object
dataset["Datum data"] = pd.to_datetime(dataset["Datum data"], dayfirst=True)

# Plot the data
fig, ax = plt.subplots()
ax.scatter(dataset["Datum data"], dataset["mm"])
ax.set_ylabel(r'[mm]')
locator = mplt.dates.AutoDateLocator()
formatter = mplt.dates.ConciseDateFormatter(locator)

# Plot a histogramm

# Plot the data bis
fig, ax = plt.subplots()
ax.scatter(dataset['Datum data'], dataset['mm'])
ax.set_xlabel('Datum')
ax.set_ylabel('mm')
ax.set_title('Cumulated 1h EV rainfall data')
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