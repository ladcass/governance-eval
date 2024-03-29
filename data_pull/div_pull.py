import wrds
import numpy as np
import pandas as pd
import time

start_time = time.time()

# Credentials to log into WRDS
db = wrds.Connection(wrds_username='s2029442')

# Get company list
companies = pd.read_csv('data_pull/db/companies.csv')
companies.dropna()


'''''
WITH        dir AS (SELECT dir.ticker, dir.director_detail_id, dir.female, dir.ethnicity
                    FROM risk.rmdirectors dir
                    WHERE dir.year = 2023.0)
SELECT      dir.ticker, dir.director_detail_id, dir.female, dir.ethnicity, comm.committeename
FROM        dir
            LEFT JOIN  boardex.na_wrds_org_summary comp ON comp.ticker = dir.ticker
            LEFT JOIN boardex.na_board_dir_committees comm ON comm.boardid = comp.boardid
GROUP BY    dir.ticker, dir.director_detail_id, dir.female, dir.ethnicity, comm.committeename;
'''

# Get company info from BoardEx and ISS ESG
div_010 = db.raw_sql('WITH dir AS (SELECT dir.ticker, dir.director_detail_id, dir.female, dir.ethnicity FROM risk.rmdirectors dir WHERE dir.year = 2023.0) SELECT dir.ticker, dir.director_detail_id, dir.female, dir.ethnicity, comm.committeename FROM dir LEFT JOIN boardex.na_wrds_org_summary comp ON comp.ticker = dir.ticker LEFT JOIN boardex.na_board_dir_committees comm ON comm.boardid = comp.boardid GROUP BY dir.ticker, dir.director_detail_id, dir.female, dir.ethnicity, comm.committeename;')

# Save data to a CSV to not have to pull data over and over again
div_010.to_csv('data_pull/db/div_010.csv', index=False, header=True)

# Convert 'committeename' and 'female' columns to strings, replacing NaN with an empty string
div_010['female'] = div_010['female'].apply(lambda x: 'No' if x != 'Yes' else 'Yes')

# Create the 'women_pct' column
female_count = div_010[div_010['female'] == 'Yes'].groupby('ticker')['female'].count().reset_index()
total_count = div_010.groupby('ticker')['female'].count().reset_index()
female = pd.merge(female_count, total_count, on='ticker')
female['female_pct'] = female['female_x'] / female['female_y']
female = female[['ticker', 'female_pct']]

# Create the 'diversity_board' column
diversity_keywords = r'divers|incl|sustai'
div_010['diversity_board'] = div_010['committeename'].str.contains(diversity_keywords, case=False, regex=True)
diversity = div_010.groupby('ticker')['diversity_board'].any().reset_index()

# Merge the two dataframes female and diversity
div_020 = pd.merge(female, diversity, on='ticker')

# Store the result in a CSV
div_020.to_csv('data_pull/db/div_020.csv', index=False, header=True)

# Now we calculate the Shannon index:
# Step 1 & 2: Group by ticker and ethnicity to count and calculate proportions
ethnicity_counts = div_010.groupby(['ticker', 'ethnicity']).size().reset_index(name='count')
total_counts = div_010.groupby('ticker').size().reset_index(name='total')
ethnicity_proportions = pd.merge(ethnicity_counts, total_counts, on='ticker')
ethnicity_proportions['p_i'] = ethnicity_proportions['count'] / ethnicity_proportions['total']

# Step 3: Calculate Shannon Index for each ticker
ethnicity_proportions['p_i_log_p_i'] = ethnicity_proportions['p_i'] * np.log(ethnicity_proportions['p_i'])
shannon = ethnicity_proportions.groupby('ticker')['p_i_log_p_i'].sum().reset_index()
shannon['shannon'] = abs(shannon['p_i_log_p_i'])
shannon = shannon.drop(columns=['p_i_log_p_i'])

# Merge the two dataframes div_020 and shannon
div_v0 = pd.merge(div_020, shannon, on='ticker')

# Left join to keep only the rows in div_020 with a corresponding ticker in index chosen
div_v0 = pd.merge(companies, div_v0, how='left', on='ticker')

# Store the result in a CSV
div_v0.to_csv('data_pull/db/div_v0.csv', index=False, header=True)

end_time = time.time()
runtime = end_time - start_time
print('Runtime:', runtime)
