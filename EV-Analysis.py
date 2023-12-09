# import packages
import math
import pandas as pd
import numpy as np
import matplotlib as mplt
import matplotlib.pyplot as plt
import scipy
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
print()

# ----------------------------------------------------------------------------
# Define dictionary keys for data storage. Every key identifies a different
# time period
# ----------------------------------------------------------------------------

# Get the keys
column_names = list(df.columns)
column_names = column_names[1:]

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

# Calculate the ECDFs
ecdf = {}
for key in column_names:
      ecdf[key] = stats.ecdf(df[key][df[key].notna()])

# print(ecdf['1h'])
# print()

# Plot the ECDF
fig, ax = plt.subplots()
ecdf['1h'].cdf.plot(ax)
ax.set_xlabel('Cumulated rainfall data [mm]')
ax.set_ylabel('Empirical CDF')
# plt.show()

# Calculate sample mean and sample variance
sample_mean = df["1h"].mean()
sample_var = df["1h"].var()

# print(f'the sample mean is {sample_mean:.2f}')
# print(f'the sample variance is {sample_var:.2f}')

# ----------------------------------------------------------------------------
# Apply the Methods of Moments to calculate Gumbel distribution parameters
# ----------------------------------------------------------------------------

# Initialize dictionaries
loc_gumbel = {}
scale_gumbel = {}

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
# Apply fit function to get lognormal distribution parameters
# ----------------------------------------------------------------------------

# Initialize dictionaries
shape_lognorm = {}
loc_lognorm = {}
scale_lognorm = {}

# Check for negative values in the data. Lognormal is defined only for positive values
for key in column_names:
      if (df[key] < 0).any():
            print("Data contains negative values. Remove or handle them appropriately.")

# Fit the data to a lognormal distribution
for key in column_names:
     shape_lognorm[key], loc_lognorm[key], scale_lognorm[key] = stats.lognorm.fit(df[key][df[key].notna()])

# Print the parameters of the fitted distribution
for k, v in shape_lognorm.items():
      print(f"{k}: Lognorm shape parameter: {v:.2f}")
print()

for k, v in loc_lognorm.items():
      print(f"{k}: Lognorm location parameter: {v:.2f}")
print()

for k, v in scale_lognorm.items():
      print(f"{k}: Lognorm scale parameter: {v:.2f}")
print()

# ----------------------------------------------------------------------------
# Calculate datapoints from the fitted distributions
# ----------------------------------------------------------------------------
# Generate random data points for the x-axis of the plot
x = {}
for key in column_names:
      # For graphical comparison the range should be similar
      x[key] = np.linspace(df[key].min(), df[key].max(), 100)

# ----------------------------------------------------------------------------
# Gumbel

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
      gumbel_r_cdf[key] = stats.gumbel_r.cdf(x[key], loc_gumbel[key], scale_gumbel[key])

# ----------------------------------------------------------------------------
# Lognormal

# Apply the probability density function
lognorm_pdf = {}
for key in column_names:
      lognorm_pdf[key] = lognormal_pdf = stats.lognorm.pdf(x[key], shape_lognorm[key], loc_lognorm[key], scale_lognorm[key])

# Apply the cumulative distribution function
lognorm_cdf = {}
for key in column_names:
      lognorm_cdf[key] = lognormal_cdf = stats.lognorm.cdf(x[key], shape_lognorm[key], loc_lognorm[key], scale_lognorm[key])

# From here on the code does apply only for a given dictionary key (e.g. ’1h’)
# that for now has to be setted by hand

# Plot the empirical cumulative density function and fitted distributions
ax = plt.subplot()
ax.set_xlabel('x [mm]')
ax.set_ylabel('probability of non-exceedance: P (X <= x)')
ecdf['1h'].cdf.plot(ax, label='ECDF')
ax.plot(x['1h'], gumbel_r_cdf['1h'], label='fitted Gumbel distribution', color='red')
ax.plot(x['1h'], lognorm_cdf['1h'], label='fitted lognormal distribution', color='green')
ax.legend()
# plt.show()

# Plot the probability density function and histogram
fig, ax = plt.subplots()
plt.hist(df["1h"][df['1h'].notna()], bins='auto', edgecolor='white', density='True', alpha=0.5, label='Histogram')
ax.plot(x['1h'], gumbel_r_pdf['1h'], label='fitted Gumbel distribution', color='red')
ax.plot(x['1h'], lognorm_pdf['1h'], label='fitted Lognormal distribution', color='green')
plt.xlabel('[mm]')
plt.ylabel('Density')            
ax.legend()
plt.show()

# ----------------------------------------------------------------------------
# Calculate Extreme Values
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

# ----------------------------------------------------------------------------
# Calculate IDF Curves h(Tr) = a * tp ^ n
# ----------------------------------------------------------------------------

# Visual Check of the datapoints on normal scale
EVs.plot()

# Define power law
def power_law(x, a, b):
    return a*x**b

# Determine a and n parameters for different return periods
data = []
# Define x-data points: durations converted to integers
tp = [int(s[:-1]) for s in EVs.columns.to_list()]
for Tr in EVs.index.to_list():
      row = []
      # Define y-data points: EVs for different durations
      h = EVs.loc[Tr].to_list()
      # Curve fit the data: popt returns the two fitted parameters a and n in a list
      popt, pcov = scipy.optimize.curve_fit(power_law, tp, h)
      popt = list(popt)
      data.append(popt)

Parameters = pd.DataFrame(data, index=return_period, columns=['a','n'])
print(Parameters)