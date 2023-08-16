# import packages
import math
import pandas as pd
import numpy as np
import matplotlib as mplt
import matplotlib.pyplot as plt
from scipy import stats

# Read data from text file
dataset = pd.read_csv("1_h_01.txt", sep="\t", header=1)

# Convert string to floating point number
dataset['mm'] = dataset['mm'].str.replace(',', '.').astype(float)
print(dataset)

# Convert string to datetime object
dataset["Datum data"] = pd.to_datetime(dataset["Datum data"], dayfirst=True)

# Plot the data
fig, ax = plt.subplots()
ax.scatter(dataset["Datum data"], dataset["mm"])
ax.set_ylabel(r'[mm]')
ax.set_title('Cumulated 1h EV rainfall data')
locator = mplt.dates.AutoDateLocator()
formatter = mplt.dates.ConciseDateFormatter(locator)
plt.show()

# Plot a histogramm
plt.hist(dataset['mm'], bins=20, edgecolor='black')
plt.xlabel('Cumulated rainfall data [mm]')
plt.ylabel('Frequency')
plt.title('Histogram of Cumulated Rainfall Data')
plt.show()

# Calculate the ECDF
res = stats.ecdf(dataset['mm'])
print(res)

# Plot the ECDF
fig, ax = plt.subplots()
ax.plot(res.x, res.y)
ax.set_xlabel('Cumulated rainfall data [mm]')
ax.set_ylabel('Empirical CDF')
plt.show()

# Calculate sample mean and sample variance
sample_mean = dataset['mm'].mean()
sample_var = dataset['mm'].var()

print(f'the sample mean is {sample_mean:.2f}')
print(f'the sample variance is {sample_var:.2f}')

# Apply the Methods of Moments to calculate Gumbel distribution parameters
eulergamma = 0.57721566490
scale_gumble = math.sqrt(6 * sample_var) / math.pi
loc_gumbel = sample_mean - eulergamma * scale_gumble

print(f"Gumbel scale parameter: {scale_gumble:.2f}",
      f"Gumbel location parameter: {loc_gumbel:.2f}")

# Generate data points for the x-axis.
# As I estimated a continuous distribution I can refer to random data points.
# For graphical comparison the range should be nevertheless similar
x = np.linspace(dataset['mm'].min(), dataset['mm'].max(), 100)

# Calculate the standardized variable
y = (x - loc_gumbel) / scale_gumble

# probability density function
gumbel_r_pdf = stats.gumbel_r.pdf(y)

# cumulative distribution function
gumbel_r_cdf = stats.gumbel_r.cdf(y)

# Plot the Gumbel distribution COMPARE WITH ECDF
ax = plt.subplot()
ax.set_xlabel('cumulative density function')
res.cdf.plot(ax)
ax.plot(x, gumbel_r_cdf, label='gumbel_r cdf', color='red')
ax.legend()
plt.show()

# Plot the Gumbel distribution COMPARE WITH HISTOGRAMM
fig, ax = plt.subplots()
ax.plot(res.x, res.y, label='ECDF')              
ax.plot(x, gumbel_r_cdf, label='Gumbel CDF', color='red')
ax.legend()
ax.set_xlabel('Cumulative density function')
plt.show()