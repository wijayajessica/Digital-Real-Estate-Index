# Digital Real Estate Index

Instructions:

This project is organized into 4 sections:

1. Query and process MLS listing data from March 2016 through the latest month that was fully observed (see `listing_data` directory)

2. Build forecasting model for entire market (see `market_level_forecasting` directory)

3. Process and interpret impact of census tract features (see `census_tract_adjustment` directory)

4. Create forecast for each census tract (see `census_tract_forecasting` directory)

Each of these directory has its own README.md file that explains how to navigate the directory. 

If you just cloned this repo, please run `download_s3_folder.py`, which will download all the raw data needed to make the forecasts from the `rex-harvard-iacs` S3 bucket. 
