import wrds
import pandas as pd
import time

start_time = time.time()

# Credentials to log into WRDS
db = wrds.Connection(wrds_username='s2029442')

# Get ROA from WRDS Financial Ratios Firm Level
'''
SELECT      DISTINCT  
            fr.ticker
            , fr.qdate
            , fr.roa
FROM        wrdsapps_finratio.firm_ratio fr;
'''

roa_010 = db.raw_sql("SELECT DISTINCT fr.ticker, fr.qdate, fr.roa FROM wrdsapps_finratio.firm_ratio fr;")
roa_010.to_csv('data_pull/db/roa_010.csv', index=False, header=True)

# Convert quarter date to datetime
roa_010['qdate'] = pd.to_datetime(roa_010['qdate'])

# Sort the DataFrame by 'ticker' and 'qdate' in ascending order
roa_v0 = roa_010.sort_values(by=['ticker', 'qdate'])

# Group by 'ticker' and use shift to get the ROA value from 4 quarters before
roa_v0['roa_prev_year'] = roa_v0.groupby('ticker')['roa'].shift(4)

# Calculate the change in ROA
roa_v0['roachange'] = roa_v0['roa'] - roa_v0['roa_prev_year']

# Drop rows where 'roa_prev_year' (and thus 'roachange') is NA (i.e., the first 4 quarters for each ticker)
roa_v0.dropna(subset=['roa_prev_year'], inplace=True)

# Now, keep only the latest entry for each ticker
roa_v0 = roa_v0.groupby('ticker').last().reset_index()

# Select only the required columns
roa_v0 = roa_v0[['ticker', 'roachange']]

# Save the DataFrame to a CSV file
roa_v0.to_csv('data_pull/db/roa_v0.csv', index=False, header=True)

# Convert YoY ROA Change to a categorical variable
# Define bins and labels for the categories
bins = [-float('inf'), -0.10, -0.03, 0, 0.01, 0.02, 0.10, float('inf')]
labels = ['Significant Decrease'
        , 'Moderate Decrease'
        , 'Slight Decrease'
        , 'Stable'
        , 'Slight Increase'
        , 'Moderate Increase'
        , 'Significant Increase']

roa_v0['roachange'] = pd.cut(roa_v0['roachange'], bins=bins, labels=labels)

# Save the DataFrame to a CSV file
roa_v0.to_csv('data_pull/db/roa_v0.csv', index=False, header=True)

end_time = time.time()
runtime = end_time - start_time
print('Runtime:', runtime)
