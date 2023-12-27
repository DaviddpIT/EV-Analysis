# HYDROLOGICAL EXTREME VALUE ANALYSIS

### Description:
The Hydrological Extreme Value Analysis project is a Python-based software solution for analyzing and visualizing extreme hydrological data. The code includes functionalities for processing and exploring hydrological datasets, fitting statistical distributions (so far two of them: Gumbel and Lognormal), and generating Intensity-Duration-Frequency (IDF) curves. 

### Features

- Data reading and preprocessing from input csv file
- Statistical sample analysis including ECDF calculation and data visualization
- Gumbel and Lognormal distribution parameter estimation
- Visualization of fitted distributions and comparison (plotting the histogram vs the probability density function and the ECDF vs the cumulative distribution function)
- Statistical Testing (Kolmogorov Smirnofone sample) in order to define the best fitting distribution for a given time period
- Extreme value calculation for user-defined return periods
- Plotting of Extreme Values and IDF curves

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
