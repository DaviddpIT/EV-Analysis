# HYDROLOGICAL EXTREME VALUE ANALYSIS

#### Video Demo:  <https://youtu.be/vLusma25RbY>

### Description:

The Hydrological Extreme Value Analysis project is a Python-based software solution for analyzing and visualizing extreme hydrological data. The code includes functionalities for processing and exploring hydrological datasets, fitting statistical distributions (so far two of them: Gumbel and Lognormal), and generating Intensity-Duration-Frequency (IDF) curves. IDF – curves are of fundamental importance when addressing problems in the field of hydraulics (e.g. storm water management, river engineering, sewer design, …) as they correlate the cumulated rainfall on a given site to the duration of the event and a respective return period

### Features and Code Sections

- Section 1 **Data Crunching**\
  In this section the necessary packages and dependencies are loaded. Make sure to have the following dependencies installed
    - math
    - pandas
    - numpy
    - matplotlib
    - scipy
    - sys

    The input file *all_data_02.csv* is loaded into a pandas dataframe, given the intrinsic type of hydrological data (a mix of dates and numeric values).

- Section 2 **Defining data types**\
The processed and / or calculated data will be stored in dictionaries (e.g. loc_gumbel, scale_gumbel, etc.). The dictionary keys are given by the different time periods, e.g. ‘1h’. This allows storing all the different location parameters for the e.g. the gumbel distribution for all time periods in one single dictionary.

- Section 3 **Calculate data statistics**\
The scipy.stats module is used to calculate the empirical distribution frequency function (ECDF), whereas the sample mean and the sample variance are calculated using dataframe related methods.
The calculated data statistics is necessary for visual comparison and the calculation of the statistical distribution parameters (e.g. scale, location, shape, …) (in the case of e.g. the Method of Moments)

- Section 4 **Determine Gumbel Distribution**\
The Method of Moments is apllied to determine Gumbel distribution’s identifying parameters: location and scale based on the underlying sample. The calculation is repeated for every time period.

- Section 5 **Determine Lognormal Distribution**\
The build in fit method that applies to objects of the stats module is used to determine the parameters for the lognormal distribution. Again the calculation is repeated for every time period.

- Section 6 **Determine data points**\
to plot the fitted distributions random datapoints are calculated for every time period. this datapoints are evenly distributed between the minimum and maximum of the cumulated rainfall for every time period. Based on this datapoints the corresponding probability is calculated by determing the cumulative distribution function (cdf) and the probability density function (pdf) for each time period and each statistical distribution.

- Section 7 **Plotting**\
This interim results are plotted for visual comparison. The plots are generated for a specific duration, specified by user input, e.g. ‘1h’. The histogram is compared to the pdf and the ECDF to the cdf.

- Section 8 **Statistical Kolmogorov Smirnof Test**\
The Kolmogorov Smirnof (KS) statistical test is used to identify the best fitting distribution:
    - H0-hypothesis: the rainfall values for a given time period follow the choosen distribution (e.g. Gumbel, lognorm, etc...)
    - Chosen alpha-level = 0.05

    By comparing the p-value we can decide which disrtribution fits best. the higher the p-value (always in case of p-avlues above alpha-level) the better a distributions fits the data. Again, the calculation is repeated for every duration.

- Section 9 **Determine extreme values**\
After the best fitting distribution has been identified, the extreme values for fixed return periods are calculated using the build in percent point function (inverse of the pdf) that applies to objects of the stats module (from the probability value I derive the corresponding rainfall). The calculated extreme values are stored in a pandas dataframe.

- Sectino 10 **Plot extreme values**\
The calculated extreme values are plotted for a user specified return period.

- Section 11 **Determine IDF curves**\
A power law function is defined $$h(T_r) = a \cdot t_p ^ n$$. The *scipy.optimize.curve_fit* method is used to fit the power law to the datapoints, determing the parameters a and n. This calculation is repeated for every time period.
Finally the parameters are stored in a pandas dataframe.

### Usage

1. Clone the repository
<https://github.com/DaviddpIT/EV-Analysis.git>
2. Prepare your own site specific rainfall data input file. Adhere strictly to the formatting of the example input file *all_data_02.csv* (or adapt the first code section where the input data is handled)
3. Run the main script *EV-Analysis.py*
4. Follow the prompts to select a time period and return period for plotting

### Design choices

Processed tabular data will be stored in pandas dataframes. Intermediate data is stored in dictionaries, where the event's durations (e.g. '1h', '3h', etc...) set the keys

### Bibliography and addtional readings

I higly recommend <https://abouthydrology.blogspot.com/> for a comprehensive in depth introduction to hydrological Extreme Value Analysis

Rainfall data for the Province of Bolzano - Italy can be downloaded from here <https://weather.provinz.bz.it/download-data.asp
>