# import packages
import math
import pandas as pd
import numpy as np
import matplotlib as mplt
import matplotlib.pyplot as plt
from scipy import stats

# Read data from text file
df = pd.read_csv("all_data_02.csv", header=0, sep="\t")
print(df.head())
print()

# Convert string to datetime object
df["date"] = pd.to_datetime(df["date"], dayfirst=True)
print(df.dtypes)
print()

# Pivot the datatable 
df = df.pivot(index="date", columns="time", values="value")

# Rename columns
new_columns = {1:"1h", 3:"3h", 6:"6h", 12:
               "12h", 24:"24h", 15:"15min", 30:"30min", 45:"45min"}
df.rename(columns=new_columns, inplace=True)

# Sort the datatable and reset index
df.sort_index(ascending=True)
df = df.reset_index()
print(df)

# Plot the data
fig, ax = plt.subplots()
ax.scatter(df["date"], df["1h"])
ax.set_ylabel(r'[mm]')
ax.set_title('Cumulated 1h EV rainfall data')
locator = mplt.dates.AutoDateLocator()
formatter = mplt.dates.ConciseDateFormatter(locator)
# plt.show()

# Plot a histogramm
plt.hist(df["1h"], bins=20, edgecolor='black')
plt.xlabel('Cumulated rainfall data [mm]')
plt.ylabel('Frequency')
plt.title('Histogram of Cumulated Rainfall Data')
# plt.show()

# Calculate the ECDF
res = stats.ecdf(df["1h"][df['1h'].notna()])
print(res)
print()

# Plot the ECDF
fig, ax = plt.subplots()
res.cdf.plot(ax)
ax.set_xlabel('Cumulated rainfall data [mm]')
ax.set_ylabel('Empirical CDF')
# plt.show()

# Calculate sample mean and sample variance
sample_mean = df["1h"].mean()
sample_var = df["1h"].var()

# print(f'the sample mean is {sample_mean:.2f}')
# print(f'the sample variance is {sample_var:.2f}')

# ----------------------------------------------------------------------------
# Create dictionaries to store the Gumbel distribution parameters for different 
# time periods
# ----------------------------------------------------------------------------
# Get the keys
column_names = list(df.columns)
column_names = column_names[1:]

# Initialize dictionaries
loc_gumbel = {}
scale_gumbel = {}

# ----------------------------------------------------------------------------
# Apply the Methods of Moments to calculate Gumbel distribution parameters
# ----------------------------------------------------------------------------
eulergamma = 0.57721566490
for key in column_names:
      sample_mean = df[key].mean()
      sample_var = df[key].var()
      scale_gumbel[key] = math.sqrt(6 * sample_var) / math.pi
      loc_gumbel[key] = sample_mean - eulergamma * scale_gumbel[key]


for k, v in scale_gumbel.items():
      print(f"{k}: Gumbel scale parameter: {v:.2f}")
print()

for k, v in loc_gumbel.items():
      print(f"{k}: Gumbel location parameter: {v:.2f}")
print()

# ----------------------------------------------------------------------------
# Calculate datapoints for the fitted Gumbel distribution
# ----------------------------------------------------------------------------
# Generate random data points for the x-axis of the plot
x = {}
for key in column_names:
      # For graphical comparison the range should be similar
      x[key] = np.linspace(df[key].min(), df[key].max(), 100)

# Calculate the standardized variable
y = {}
for key in column_names:
      y[key] = (x[key] - loc_gumbel[key]) / scale_gumbel[key]

# Apply the probability density function
gumbel_r_pdf = {}
for key in column_names:
      gumbel_r_pdf[key] = stats.gumbel_r.pdf(y[key]) / scale_gumbel[key]

# Apply the cumulative distribution function
gumbel_r_cdf = {}
for key in column_names:
      gumbel_r_cdf[key] = stats.gumbel_r.cdf(y[key])

# Plot the empirical cumulative density function and fitted gumbel distribution
# ax = plt.subplot()
# ax.set_xlabel('x [mm]')
# ax.set_ylabel('probability of non-exceedance: P (X <= x)')
# res.cdf.plot(ax, label='ECDF')
# ax.plot(x, gumbel_r_cdf, label='fitted Gumbel distribution', color='red')
# ax.legend()
# plt.show()

# Plot the probability density function and histogram
# fig, ax = plt.subplots()
# plt.hist(df["1h"][df['1h'].notna()], bins='auto', edgecolor='white', density='True', alpha=0.5, label='Histogram')
# ax.plot(x, gumbel_r_pdf, label='fitted Gumbel distribution', color='red')
# plt.xlabel('[mm]')
# plt.ylabel('Density')            
# ax.legend()
# plt.show()

# ----------------------------------------------------------------------------
# Calculate IDF Curves
# ----------------------------------------------------------------------------
return_period = [5, 10, 30 , 50, 100]

# Propability of non exceedence
probability = []
for Tr in return_period:
      probability.append(1 - 1 / Tr)

# Calculate EVs
data = []
for p in probability:
      row = []
      for key in column_names:
            # Apply inverse transform sampling via the percent point function gumbel_r.ppf(). By Specifyng the loc and scale parameter, the function automatically handels back transformation from the standardized to the original variable
            ev = stats.gumbel_r.ppf(p, loc=loc_gumbel[key], scale=scale_gumbel[key])
            row.append(ev)
      data.append(row)

EVs = pd.DataFrame(data, index=return_period, columns=column_names)
print(EVs)

# Select only hourly EVs
EVs = EVs.drop(columns=['30min', '45min','15min'])

# Plot the EVs for a given return period on log log scale
fig, ax = plt.subplots()
ax.set_xlabel('duration')
ax.set_ylabel('h [mm]')
ax.set_title("Computed extreme values") 
# Define values for y-axis: EVs for different durations
h = EVs.loc[5].to_list()
# Define values for x-values: durations converted to integers
tp = [int(s[:-1]) for s in EVs.columns.to_list()]
ax.loglog(tp, h, label='Tr = 5 years', marker='o', linestyle='None')
ax.legend()
ax.grid(True)
plt.show()
