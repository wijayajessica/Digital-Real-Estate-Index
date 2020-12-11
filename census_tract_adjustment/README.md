Instructions:

The first step is to collect all of the census tract level data that isn't provided by NeighborhoodScout:

1. Download the following tables for the most recently available year (unless otherwise noted) from https://data.census.gov/cedsci/table?t=Housing&g=0400000US08.140000&tid=ACSDT5Y2018.B25002&hidePreview=false, place them into the following folder: `index-team-data/[insert market area name here]/census_tract_datasets` and rename them as follows:

  - AGE AND SEX -> age_demographics.csv

  - ACS demographic and housing estimates -> racial_demographics.csv

  - Fertility -> fertility.csv

  - MEANS OF TRANSPORTATION TO WORK BY TENURE or MEANS OF TRANSPORTATION TO WORK BY AGE -> commuting_mode.csv

  - INCOME IN THE PAST 12 MONTHS -> income.csv

  - FOOD STAMPS/SUPPLEMENTAL NUTRITION ASSISTANCE PROGRAM (SNAP) -> food_stamps.csv

  - SELECTED ECONOMIC CHARACTERISTICS -> economic_characteristics.csv

  - HOUSEHOLDS AND FAMILIES -> household_size.csv

  - VALUE -> house_value.csv

  - MARITAL STATUS -> marital_status.csv

  - MORTGAGE STATUS -> mortgage_status.csv

  - FINANCIAL CHARACTERISTICS FOR HOUSING UNITS WITH A MORTGAGE -> mortgage_amount.csv

  - YEAR STRUCTURE BUILT -> year_built.csv

  - AGE AND SEX (all years since 2010) -> folder called age_demographics_2010_2018


2. create a new notebook called `[insert market area name here]_census_tract_features.ipynb` by simply copying the corresponding notebook associated with a market area that has already been processed. The purpose of this notebook is to preprocess the census tract features from NeighborhoodScout and the Census Bureau. Because the census tracts features of each market area have different issues (.e.g. missing data, wrong data type), this is not an automatic process. Instead, you need to run the notebook cell by cell to check that the notebook handles all the issues and, if not, you need to write code to resolve the remaining issues. 

3. See `ct_model_evaluation.ipynb` if you'd like to compare the performance of the census tract adjustment model to the baseline model that makes the same prediction for each census tract.

4. See Jessica's notebooks...

5. See `census_tract_explainer_linear_regression.ipynb` if you'd like to see the subpar results from using a linear regression model to explain how each of the census tract features contributes to the average ratios of the census tracts. 

6. See `population_growth_explainer.iypnb` if you'd like to see the linear regression model that tries to predict each census tract's average population growth rate since 2010 from the other census tract features. 


