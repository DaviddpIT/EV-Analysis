# import packages
import math
import pandas as pd
import numpy as np
import matplotlib as mplt
import matplotlib.pyplot as plt
from scipy import stats

# Read data from text file
df = pd.read_csv("all_data_02.csv",header=0, sep="\t")
print(df.head())

# Convert string to datetime object
df["date"] = pd.to_datetime(df["date"], dayfirst=True)
print(df.dtypes)

# Pivot the datatable 
df = df.pivot(index="date", columns="time", values="value")

# Rename columns
new_columns = {1:"1h", 2:"2h", 3:"3h", 6:"6h", 12:
               "12h", 24:"24h", 15:"15min", 30:"30min", 45:"45min"}
df.rename(columns=new_columns, inplace=True)


df.sort_index(ascending=True)

# NOT WORKING RESET INDEX
df.reset_index()
print(df)

# Plot the data
fig, ax = plt.subplots()
ax.scatter(dataset["Datum data"], dataset["mm"])
ax.set_ylabel(r'[mm]')
ax.set_title('Cumulated 1h EV rainfall data')
locator = mplt.dates.AutoDateLocator()
formatter = mplt.dates.ConciseDateFormatter(locator)
# plt.show()

# Plot a histogramm
plt.hist(dataset['mm'], bins=20, edgecolor='black')
plt.xlabel('Cumulated rainfall data [mm]')
plt.ylabel('Frequency')
plt.title('Histogram of Cumulated Rainfall Data')
# plt.show()

# Calculate the ECDF
res = stats.ecdf(dataset['mm'])
# print(res)

# Plot the ECDF
fig, ax = plt.subplots()
res.cdf.plot(ax)
ax.set_xlabel('Cumulated rainfall data [mm]')
ax.set_ylabel('Empirical CDF')
# plt.show()

# Calculate sample mean and sample variance
sample_mean = dataset['mm'].mean()
sample_var = dataset['mm'].var()

print(f'the sample mean is {sample_mean:.2f}')
print(f'the sample variance is {sample_var:.2f}')

# Apply the Methods of Moments to calculate Gumbel distribution parameters
eulergamma = 0.57721566490
scale_gumbel = math.sqrt(6 * sample_var) / math.pi
loc_gumbel = sample_mean - eulergamma * scale_gumbel

print(f"Gumbel scale parameter: {scale_gumbel:.2f}",
      f"Gumbel location parameter: {loc_gumbel:.2f}")

# Generate data points for the x-axis.
# As I estimated a continuous distribution I can refer to random data points.
# For graphical comparison the range should be nevertheless similar
x = np.linspace(dataset['mm'].min(), dataset['mm'].max(), 100)

# Calculate the standardized variable
y = (x - loc_gumbel) / scale_gumbel

# probability density function
gumbel_r_pdf = stats.gumbel_r.pdf(y) / scale_gumbel

# cumulative distribution function
gumbel_r_cdf = stats.gumbel_r.cdf(y)

# Plot the empirical cumulative density function and fitted gumbel distribution
ax = plt.subplot()
ax.set_xlabel('x [mm]')
ax.set_ylabel('probability of non-exceedance: P (X <= x)')
res.cdf.plot(ax, label='ECDF')
ax.plot(x, gumbel_r_cdf, label='fitted Gumbel distribution', color='red')
ax.legend()
# plt.show()

# Plot the probability density function and histogram
fig, ax = plt.subplots()
plt.hist(dataset['mm'], bins='auto', edgecolor='white', density='True', alpha=0.5, label='Histogram')
ax.plot(x, gumbel_r_pdf, label='fitted Gumbel distribution', color='red')
plt.xlabel('[mm]')
plt.ylabel('Density')            
ax.legend()
# plt.show()

# Calculate hydrological EVs for specified return periods
return_period = [5, 10, 30 , 50, 100]

# associated propability of non exceedence
for Tr in return_period:
      p = 1 - 1 / Tr

# Apply inverse transform sampling via the percent point function
x = np.zeros(len(return_period))
# for probability in p:
#       x[] # enumerate...
x = stats.gumbel_r.ppf(p, loc=loc_gumbel, scale=scale_gumbel)
print(x)