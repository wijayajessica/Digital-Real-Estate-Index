from copy import deepcopy
import datetime
import json
import pandas as pd


def convert_to_zipcode(zip_int):
    zipcode = str(zip_int)
    if len(zipcode) == 4:
        return '0' + zipcode
    return zipcode


df = pd.read_csv('boston_listing_dates.csv')
df['zipcode'] = df['zipcode'].apply(convert_to_zipcode)

start_date = df.list_date[0]
date_today = datetime.date.today()
date_today_str = str(date_today)
dates = [str(date.date()) for date in pd.date_range(start_date, date_today, freq='d')]
zipcodes = set(df.zipcode)

category_to_cnt = {cat: 0 for cat in ['new_listings', 'sales']}
date_to_category_to_cnt = {date: deepcopy(category_to_cnt) for date in dates}
zipcode_to_date_to_category_to_cnt = {zipcode: deepcopy(date_to_category_to_cnt)
                                      for zipcode in zipcodes}
date_to_category_to_market_area_cnt = deepcopy(date_to_category_to_cnt)

for idx, row in df.iterrows():
    zipcode = row['zipcode']
    list_date = row['list_date']
    sale_date = row['sale_date']
    if list_date <= date_today_str:
        zipcode_to_date_to_category_to_cnt[zipcode][list_date]['new_listings'] += 1
        date_to_category_to_market_area_cnt[list_date]['new_listings'] += 1
    if pd.notnull(sale_date) and sale_date <= date_today_str:
        zipcode_to_date_to_category_to_cnt[zipcode][sale_date]['sales'] += 1
        date_to_category_to_market_area_cnt[sale_date]['sales'] += 1

with open('zipcode_to_date_to_category_to_cnt.txt', 'w') as f:
    f.write(json.dumps(zipcode_to_date_to_category_to_cnt))
with open('date_to_category_to_market_area_cnt.txt', 'w') as f:
    f.write(json.dumps(date_to_category_to_market_area_cnt))
