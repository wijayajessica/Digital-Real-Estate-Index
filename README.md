# Digital Real Estate Index

For the most updated csv data and technical reports/presentation slides, go to this [goggle drive shared folder](https://drive.google.com/drive/u/2/folders/0ADNiRHNQgWGcUk9PVA)

Instructions:

In order to produce a forecast for a given market area please, please follow the steps below:

1. Retrieve the data on each listing in the market area by specifying `market_area_name` in `query_listing_dates.ipynb` and then running the notebook. This will output a text file that contains the zipcodes in the given market area as well a CSV file that contains the relevant information for each listing.

2. Specify `market_area` in `process_listing_dates.ipynb` and then run the notebook. This will ensure that any listing that is on the market multiple times within a 365 day period is counted only once.

3. Download the following tables for the most recently available year (unless otherwise noted) from https://data.census.gov/cedsci/table?t=Housing&g=0400000US08.140000&tid=ACSDT5Y2018.B25002&hidePreview=false, place them into the following folder: `[insert market area name here]_census_tract_data` and rename them as follows:

AGE AND SEX -> age_demographics.csv

ACS demographic and housing estimates -> racial_demographics.csv

Fertility -> fertility.csv

MEANS OF TRANSPORTATION TO WORK BY TENURE or MEANS OF TRANSPORTATION TO WORK BY AGE -> commuting_mode.csv

INCOME IN THE PAST 12 MONTHS -> income.csv

FOOD STAMPS/SUPPLEMENTAL NUTRITION ASSISTANCE PROGRAM (SNAP) -> food_stamps.csv

SELECTED ECONOMIC CHARACTERISTICS -> economic_characteristics.csv

HOUSEHOLDS AND FAMILIES -> household_size.csv

VALUE -> house_value.csv

MARITAL STATUS -> marital_status.csv

MORTGAGE STATUS -> mortgage_status.csv

FINANCIAL CHARACTERISTICS FOR HOUSING UNITS WITH A MORTGAGE -> mortgage_amount.csv

YEAR STRUCTURE BUILT -> year_built.csv

AGE AND SEX (all years since 2010) -> folder called age_demographics_2010_2018


4. create new notebook called `[insert market area name here]_census_tract_features.ipynb` to preprocess the census tract features from NeighborhoodScout and the Census Bureau. Because the census tracts features of each market area have different issues, this is not an automatic process. Instead, you need to run the notebook cell by cell to ensure that the notebook catches all the typical issues (missing data, etc.) and need to write code to resolve any other issues that come up. 
