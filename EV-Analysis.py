def main():

      import math
      import pandas as pd
      import numpy as np
      import matplotlib as mplt
      import matplotlib.pyplot as plt
      import scipy
      from scipy import stats
      import sys

      # Read data from text file
      df = pd.read_csv("all_data_02.csv", header=0, sep="\t")
      # print(df.head())
      # print()

      # Convert string to datetime object
      df["date"] = pd.to_datetime(df["date"], dayfirst=True)
      # print(df.dtypes)
      # print()

      # Pivot the datatable 
      df = df.pivot(index="date", columns="time", values="value")

      # Rename columns
      new_columns = {1:"1h", 3:"3h", 6:"6h", 12:
                  "12h", 24:"24h", 15:"15min", 30:"30min", 45:"45min"}
      df.rename(columns=new_columns, inplace=True)

      # Sort the datatable and reset index
      df.sort_index(ascending=True)
      df = df.reset_index()
      print("Read data array")
      print(df)
      print()

      # ------------------------------------------------------------------------
      # Define dictionary keys for data storage. Every key identifies a
      # different time period
      # ------------------------------------------------------------------------

      # Get the keys
      column_names = list(df.columns)
      column_names = column_names[1:]

      # ------------------------------------------------------------------------
      # It's good practice to check the data visually beforehand. The following lines of codes can be used to plot the data
      # ------------------------------------------------------------------------
      
      # Plot the data
      # fig, ax = plt.subplots()
      # ax.scatter(df["date"], df["1h"])
      # ax.set_ylabel(r'[mm]')
      # ax.set_title('Cumulated 1h EV rainfall data')
      # locator = mplt.dates.AutoDateLocator()
      # formatter = mplt.dates.ConciseDateFormatter(locator)
      # plt.show()

      # Plot a histogramm
      # plt.hist(df["1h"], bins=20, edgecolor='black')
      # plt.xlabel('Cumulated rainfall data [mm]')
      # plt.ylabel('Frequency')
      # plt.title('Histogram of Cumulated Rainfall Data')
      # plt.show()

      # ------------------------------------------------------------------------
      # Calculate data statistics
      # ------------------------------------------------------------------------
      
      # Calculate the ECDFs
      ecdf = {}
      for key in column_names:
            ecdf[key] = stats.ecdf(df[key][df[key].notna()])

      # print ECDF results
      # print(ecdf['1h'])
      # print()

      # Calculate sample mean and sample variance
      sample_mean = df["1h"].mean()
      sample_var = df["1h"].var()

      # print(f'the sample mean is {sample_mean:.2f}')
      # print(f'the sample variance is {sample_var:.2f}')

      # ------------------------------------------------------------------------
      # Apply the Methods of Moments to calculate Gumbel distribution parameters
      # ------------------------------------------------------------------------

      # Initialize dictionaries to store the data
      gumbel_dist = {}
      loc_gumbel = {}
      scale_gumbel = {}

      eulergamma = 0.57721566490
      for key in column_names:
            sample_mean = df[key].mean()
            sample_var = df[key].var()
            scale_gumbel[key] = math.sqrt(6 * sample_var) / math.pi
            loc_gumbel[key] = sample_mean - eulergamma * scale_gumbel[key]

      # Print the parameters of the fitted distribution
      # for k, v in scale_gumbel.items():
      #       print(f"{k}: Gumbel scale parameter: {v:.2f}")
      # print()

      # for k, v in loc_gumbel.items():
      #       print(f"{k}: Gumbel location parameter: {v:.2f}")
      # print()

      # Define a frozen gumbel distribution object from the estimated parameters
      for key in column_names:
            gumbel_dist[key] = stats.gumbel_r(loc_gumbel[key], scale_gumbel[key])

      # ------------------------------------------------------------------------
      # Apply fit function to get lognormal distribution parameters
      # ------------------------------------------------------------------------

      # Initialize dictionaries
      lognorm_dist = {}
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

            # Define a frozen lognorm distribution object from the estimated parameters
            lognorm_dist[key] = stats.lognorm(shape_lognorm[key], loc_lognorm[key], scale_lognorm[key])

      # Print the parameters of the fitted distribution
      # for k, v in shape_lognorm.items():
      #       print(f"{k}: Lognorm shape parameter: {v:.2f}")
      # print()

      # for k, v in loc_lognorm.items():
      #       print(f"{k}: Lognorm location parameter: {v:.2f}")
      # print()

      # for k, v in scale_lognorm.items():
      #       print(f"{k}: Lognorm scale parameter: {v:.2f}")
      # print()

      # ------------------------------------------------------------------------
      # Calculate datapoints from the fitted distributions
      # ------------------------------------------------------------------------

      # Generate random data points for the x-axis of the plot
      x = {}
      for key in column_names:
            # For graphical comparison the range should be similar
            x[key] = np.linspace(df[key].min(), df[key].max(), 100)

      # ------------------------------------------------------------------------
      # Gumbel

      # Calculate the standardized variable

      # Apply the probability density function
      gumbel_r_pdf = {}
      for key in column_names:
            gumbel_r_pdf[key] = gumbel_dist[key].pdf(x[key])

      # Apply the cumulative distribution function
      gumbel_r_cdf = {}
      for key in column_names:
            gumbel_r_cdf[key] = gumbel_dist[key].cdf(x[key])

      # ------------------------------------------------------------------------
      # Lognormal

      # Apply the probability density function
      lognorm_pdf = {}
      for key in column_names:
            lognorm_pdf[key] = lognorm_dist[key].pdf(x[key])

      # Apply the cumulative distribution function
      lognorm_cdf = {}
      for key in column_names:
            lognorm_cdf[key] = lognorm_dist[key].cdf(x[key])

      # ------------------------------------------------------------------------
      # Plot the probability density function and the Cumulative distribution function for the fitted distributions for a given time period defined by the user
      # ------------------------------------------------------------------------

      # get time period from user input
      print()
      udf_key = input(f"Choose time period "
                                    f"from {column_names}: ")
      if udf_key not in column_names:
            print(f"user input not corresponding"
                  f"to possible choices: {column_names}")
            sys.exit(1)  # Exit with error code 1

      # Plot the data
      fig, ax = plt.subplots()
      ax.scatter(df["date"], df[udf_key])
      ax.set_ylabel(r'[mm]')
      ax.set_title('Cumulated EV rainfall data')
      plt.show()

      # Plot a histogramm
      plt.hist(df[udf_key], bins=20, edgecolor='black')
      plt.xlabel('Cumulated rainfall data [mm]')
      plt.ylabel('Frequency')
      plt.title('Histogram of Cumulated Rainfall Data')
      plt.show()

      # Plot the empirical cumulative density function and fitted distributions
      fig, ax = plt.subplots()
      ax.set_xlabel('Cumulated rainfall data [mm]')
      ax.set_ylabel('probability of non-exceedance: P (X <= x)')
      ecdf[udf_key].cdf.plot(ax, label='ECDF')
      ax.plot(x[udf_key], gumbel_r_cdf[udf_key], label=f'fitted Gumbel distribution for {udf_key}', color='red')
      ax.plot(x[udf_key], lognorm_cdf[udf_key], label=f'fitted lognormal distribution for {udf_key}', color='green')
      ax.legend()
      plt.show()

      # Plot the probability density function and histogram
      fig, ax = plt.subplots()
      plt.hist(df[udf_key][df[udf_key].notna()], bins='auto', edgecolor='white', density='True', alpha=0.5, label='Histogram')
      ax.plot(x[udf_key], gumbel_r_pdf[udf_key], label=f'fitted Gumbel distribution for {udf_key}', color='red')
      ax.plot(x[udf_key], lognorm_pdf[udf_key], label=f'fitted lognormal distribution for {udf_key}', color='green')
      plt.xlabel('[mm]')
      plt.ylabel('Density')            
      ax.legend()
      plt.show()

      # ------------------------------------------------------------------------
      # Apply the Kolmogorov-Smirnof Test to evaluate which distrubtions fits
      # the data better
      # ------------------------------------------------------------------------

      # H0-hypothesis: the rainfall values for a given timeperiod follow the choosen distribution (e.g. Gumbel, lognorm, etc...)
      # Chosen alpha-level = 0.05

      # By comparing the p-value we can decide which disrtribution fits best. the higher the p-value (always in case of p-avlues above alpha-level) the better a distributions fits the data

      lognorm_test_statistic, lognorm_p_value = stats.ks_1samp(df[udf_key][df[udf_key].notna()], lognorm_dist[udf_key].cdf)

      gumbel_test_statistic, gumbel_p_value = stats.ks_1samp(df[udf_key][df[udf_key].notna()], gumbel_dist[udf_key].cdf)

      # Print the test statistic and p-value
      print()
      print("KS gumbel Test Statistic:", gumbel_test_statistic)
      print("gumbel p-value:", gumbel_p_value)
      print()
      print("KS lognorm Test Statistic:", lognorm_test_statistic)
      print("lognorm p-value:", lognorm_p_value)

      if lognorm_p_value < gumbel_p_value:
            print()
            print("Considering the given p-values, the gumbel distributions fits the data better\n")
            best_dist = gumbel_dist
      else:
            print()
            print("Considering the given p-values, the Lognorm distributions fits the data better \n")
            best_dist = lognorm_dist

      # ------------------------------------------------------------------------
      # Calculate Extreme Values
      # ------------------------------------------------------------------------
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
                  ev = best_dist[key].ppf(p)
                  row.append(ev)
            data.append(row)

      EVs = pd.DataFrame(data, index=return_period, columns=column_names)
      print(f"Calculated EV for different return periods {return_period} and "
            f"time periods {column_names} \n")
      print(EVs)

      # ------------------------------------------------------------------------
      # Plot Extreme Values
      # ------------------------------------------------------------------------

      # get return period from user input
      print()
      udf_return_period = int(input(f"Choose return "
                                    f"period from {return_period} for plotting: "))
      if udf_return_period not in return_period:
            print(f"user input not corresponding "
                  f"to possible choices: {return_period}")
            sys.exit(2)  # Exit with error code 2


      # Define values for y-axis: EVs for different durations
      h = EVs.loc[udf_return_period].to_list()

      # Define values for x-values: durations converted to integers
      tp = []
      for s in EVs.columns.to_list():
            if len(s) > 3:
                  # Convert minutes to hours
                  tp.append(int(s[:-3]) * 1 / 60)
            else:
                  tp.append(int(s[:-1]))

      fig, ax = plt.subplots()
      ax.set_xlabel('duration')
      ax.set_ylabel('h [mm]')
      ax.set_title("Computed extreme values") 
      ax.loglog(tp, h, label=f'Tr = {udf_return_period} years', marker='o', linestyle='None')
      ax.legend()
      ax.grid(True)
      plt.show()

      # ------------------------------------------------------------------------
      # Calculate IDF Curves h(Tr) = a * tp ^ n
      # ------------------------------------------------------------------------

      # Visual Check of the datapoints on normal scale
      # EVs.plot()
      # plt.show()

      # Define power law
      def power_law(x, a, b):
            return a*x**b

      # Determine a and n parameters for different return periods
      data = []
      # x-data points are given by tp previously computed
      for Tr in return_period:
            row = []
            # Define y-data points: EVs for different durations
            h = EVs.loc[Tr].to_list()
            # Curve fit the data: popt returns the two fitted parameters a and n in a list
            popt, pcov = scipy.optimize.curve_fit(power_law, tp, h)
            popt = list(popt)
            data.append(popt)

      Parameters = pd.DataFrame(data, index=return_period, columns=['a','n'])
      print(f"Calculated a and n parameters for "
            f"the return periods {return_period}")
      print(Parameters)


main()