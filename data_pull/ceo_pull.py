import wrds
import pandas as pd
from datetime import datetime
from thefuzz import process
import time

start_time = time.time()
# Credentials to log into WRDS
db = wrds.Connection(wrds_username='s2029442')

# Get company list
companies = pd.read_csv('data_pull/db/companies.csv')
companies.dropna()

'''''
SELECT      comp.ticker, emp.directorid, emp.directorname, emp.rolename, emp.datestartrole, emp.dateendrole, edu.companyname as university, edu.qualification
FROM        boardex.na_wrds_dir_profile_emp emp 
            INNER JOIN boardex.na_wrds_org_summary comp ON comp.boardid = emp.companyid 
            LEFT JOIN boardex.na_dir_profile_education edu ON edu.directorid = emp.directorid
GROUP BY    comp.ticker, emp.directorid, emp.directorname, emp.rolename, emp.datestartrole, emp.dateendrole, university, edu.qualification;
'''

# Get company info from BoardEx
ceo_010 = db.raw_sql( 'SELECT comp.ticker, emp.directorid, emp.directorname, emp.rolename, emp.datestartrole, emp.dateendrole, edu.companyname as university, edu.qualification FROM boardex.na_wrds_dir_profile_emp emp INNER JOIN boardex.na_wrds_org_summary comp ON comp.boardid = emp.companyid LEFT JOIN boardex.na_dir_profile_education edu ON edu.directorid = emp.directorid GROUP BY comp.ticker, emp.directorid, emp.directorname, emp.rolename, emp.datestartrole, emp.dateendrole, university, edu.qualification;')

# Save data to a CSV to not have to pull data over and over again
ceo_010.to_csv('data_pull/db/ceo_010.csv', index=False, header=True)

# Perform the left join
ceo_010 = pd.merge(companies, ceo_010, how='left', on='ticker')

# Filter rows based on conditions
# 1. ceo_010.ticker is present in companies.ticker for the same year
# 2. rolename contains 'CEO'
# 3. ticker is not None
ceo_020 = ceo_010[
    (ceo_010['rolename'].str.contains('CEO', case=False)) &
    (ceo_010['rolename'].str.contains('regional', case=False) == False) &
    (ceo_010['rolename'].str.contains('division', case=False) == False) &
    (ceo_010['ticker'].notna())]

ceo_020.to_csv('data_pull/db/ceo_020.csv', index=False, header=True)

# Get university ranking info to calculate the university score
# Define the range of years for your CSV files
years = range(2011, 2025)

# Initialize an empty list to store each DataFrame
dfs = []

# Loop through each year, construct the file path, read the CSV, and add the 'year' column
for year in years:
    file_path = f'data_pull/raw/{year}_rankings.csv'
    df = pd.read_csv(file_path)

    # Remove columns we don't want
    df = df[['rank', 'name', 'scores_overall']]

    # Remove rows with "Reporter" as rank
    df = df[df['rank'] != 'Reporter']
    df = df[df['scores_overall'] != '—']
    df = df[df['scores_overall'] != '-']

    # Handle score ranges by calculating the average if necessary
    def handle_score_range(score):
        if '—' in str(score):
            low, high = map(float, score.split('—'))
            return (low + high) / 2
        elif '-' in str(score):
            low, high = map(float, score.split('-'))
            return (low + high) / 2
        elif '–' in str(score):
            low, high = map(float, score.split('–'))
            return (low + high) / 2
        else:
            return float(score)


    df['scores_overall'] = df['scores_overall'].apply(handle_score_range)

    # Append the DataFrame to the list
    dfs.append(df)

# Concatenate all DataFrames in the list into one
rankings = pd.concat(dfs, ignore_index=True)

# Calculate the average score for each university
universities = rankings.groupby('name')['scores_overall'].mean().reset_index()

# Rename the 'scores_overall' column to 'uniscoreavg'
universities.rename(columns={'scores_overall': 'uniscoreavg'}, inplace=True)

# Store the result in a CSV
universities.to_csv('data_pull/db/universities.csv', index=False, header=True)


# Given that the universities in ceo_020 and in universities were given by different sources,
# the same university is mentioned differently in each, therefore, we will match names and return the best match
def get_closest_match(x, choices, cutoff=80):
    match = process.extractOne(x, choices, score_cutoff=cutoff)
    return match[0] if match else None


# Assuming 'name' in universities and 'university' in ceo_020 need matching
choices = universities['name']
ceo_020['matched_university'] = ceo_020['university'].apply(lambda x: get_closest_match(str(x), choices))

# Now you can join on 'name' and 'matched_university'
ceo_030 = pd.merge(ceo_020, universities, left_on='matched_university', right_on='name', how='left')

# Clean the columns. Remove unneeded, and rename name to university
ceo_030 = ceo_030.drop(columns=['university', 'matched_university'])
ceo_030.rename(columns={'name': 'university'})

# Categorize distinct values of qualifications into appropriate degree levels
undergrad = ['BS', 'BA', 'BSc', 'BCom', 'BA (magna cum laude)', 'BFA (Bachelors of Fine Arts Program)',
             'BSE (summa cum laude)', 'BASc', 'BTech', 'AB', 'BBA', 'BCom', 'BA (Hons)', 'BSEE', 'BA (summa cum laude)',
             'BBA (Hons)', "Bachelor's Degree (Hons)", 'BComm', 'BS (cum laude)', 'BCom (Hons)',
             'BAppSc (Bachelor of Applied Science)', 'BBA (summa cum laude)', "Bachelor's Degree", 'BE', 'BSBA',
             'BSc (Hons)', 'BSE', 'AA', 'Graduated', 'Attended', 'Studied', 'Completed']
postgraduate = ['MA', 'MSc', 'MEng', 'MSEE', 'MPhil', 'MPA', 'MS', 'MSME', 'MSc (Hons)', 'MS (Hons)',
                'Post Graduate Diploma', 'Graduate Diploma', 'Diploma', 'Certificate', 'Certified', 'Qualified',
                'Chartered', 'Licensed', 'Registered', 'Certification', 'Advanced Management Program',
                'Management Development Program', 'Executive Course', 'Strategic Leadership Program',
                'Leadership and Management Program', 'General Management Program', 'Higher Diploma']
mba = ['MBA', 'Executive MBA', 'MBA (Distinction)', 'MBA (High Distinction)', 'MBA (summa cum laude)']
phd = ['PhD', 'Doctorate', 'ScD', 'JD', 'JD (magna Cum Laude)', 'JD (Cum Laude)', 'JD (Hons)', 'LLM', 'LLB',
       'Doctor of Veterinary Medicine (DVM)']


# Function to categorize qualifications
def categorize_qualification(row):
    qualification = str(row['qualification']).lower()  # Convert to string and lowercase for comparison
    row['bachelors'] = any(qual.lower() in qualification for qual in undergrad)
    row['masters'] = any(qual.lower() in qualification for qual in postgraduate) and not row[
        'bachelors']  # Avoids double counting if already counted as undergrad
    row['mba'] = any(qual.lower() in qualification for qual in mba)
    row['phd'] = any(qual.lower() in qualification for qual in phd)
    return row


# Apply the function across the DataFrame to have one column per degree level
ceo_030 = ceo_030.apply(categorize_qualification, axis=1)

# Drop the university column
ceo_030 = ceo_030.drop(columns=['university', 'qualification'])

# Convert date columns to datetime
ceo_030['datestartrole'] = pd.to_datetime(ceo_030['datestartrole'])
ceo_030['dateendrole'] = pd.to_datetime(ceo_030['dateendrole'], errors='coerce')

# For N/A values in dateendrole (still in position), we will replace them with the current date
ceo_030.fillna({['dateendrole']: pd.Timestamp(datetime.now())}, inplace=True)

# Get average of university grade, if available, and remove duplicate rows
aggregations = {
    'uniscoreavg': 'mean',          # Average of uniscoreavg
    'bachelors': 'max',             # OR operation can be achieved by taking the max, since True > False
    'masters': 'max',               # Same logic for other boolean columns
    'mba': 'max',
    'phd': 'max',
    'datestartrole': 'min',         # Minimum validstart
    'dateendrole': 'max',           # Maximum validend
}

# Group by year, ticker, and directorid then apply the aggregation
ceo_030 = ceo_030.groupby(['ticker', 'directorid', 'directorname', 'rolename']).agg(aggregations).reset_index()

# Store the result in a CSV
ceo_030.to_csv('data_pull/db/ceo_030.csv', index=False, header=True)

# Add column internal promotion to see if the CEO was promoted from within the company:
# Get data from BoardEx
'''''
WITH temp AS (  SELECT  comp.ticker
                        , emp.directorid
                        , emp.rolename
                        , emp.datestartrole 
                FROM    boardex.na_wrds_dir_profile_emp emp 
                        INNER JOIN boardex.na_wrds_org_summary comp ON comp.boardid = emp.companyid 
                GROUP BY comp.ticker
                        , emp.directorid
                        , emp.rolename
                        , emp.datestartrole)
SELECT      ceo.ticker
            , ceo.directorid
            , ceo.rolename
            , non_ceo.rolename AS nonrole
            , ceo.datestartrole
            , non_ceo.datestartrole AS nonstartrole 
FROM        temp ceo 
            LEFT JOIN temp AS non_ceo ON non_ceo.directorid = ceo.directorid 
                                        AND ceo.ticker = non_ceo.ticker 
GROUP BY    ceo.ticker
            , ceo.directorid
            , ceo.rolename
            , non_role
            , ceo.datestartrole
            , non_startrole;
'''

promotion_010 = db.raw_sql("with temp as (SELECT comp.ticker, emp.directorid, emp.rolename, emp.datestartrole FROM boardex.na_wrds_dir_profile_emp emp INNER JOIN boardex.na_wrds_org_summary comp ON comp.boardid = emp.companyid GROUP BY comp.ticker, emp.directorid, emp.rolename, datestartrole) SELECT ceo.ticker, ceo.directorid, ceo.rolename, non_ceo.rolename as nonrole, ceo.datestartrole, non_ceo.datestartrole as nonstartrole FROM temp ceo LEFT JOIN temp AS non_ceo ON non_ceo.directorid = ceo.directorid AND ceo.ticker = non_ceo.ticker GROUP BY ceo.ticker, ceo.directorid, ceo.rolename, nonrole, ceo.datestartrole, nonstartrole;")
promotion_010.to_csv('data_pull/db/promotion_010.csv', index=False, header=True)
promotion_010 = pd.read_csv('data_pull/db/promotion_010.csv')

promotion_020 = promotion_010[
    (promotion_010['rolename'].str.contains('CEO', case=False)) &
    (promotion_010['nonrole'].str.contains('CEO', case=False) == False) &
    (promotion_010['datestartrole'] >= promotion_010['nonstartrole']) &
    (promotion_010['directorid'].notna())]

promotion_020 = promotion_020[['ticker', 'directorid', 'rolename']]

# Remove duplicates
promotion_020 = promotion_020.drop_duplicates()
promotion_020.to_csv('data_pull/db/promotion_020.csv', index=False, header=True)

# Perform a left join between ceo_040 and promotion_020
# Use 'indicator' argument to add a temporary '_merge' column indicating whether the merge operation found a match or not
ceo_040 = pd.merge(ceo_030, promotion_020, on=['ticker', 'directorid', 'rolename'], how='left', indicator=True)

# # Create 'internal_promotion' column based on whether a match was found
ceo_040['internal_promotion'] = ceo_040['_merge'] == 'both'

# # Drop the temporary '_merge' column as it's no longer needed
ceo_040.drop(columns=['_merge'], inplace=True)

# Store the result in a CSV
ceo_040.to_csv('data_pull/db/ceo_040.csv', index=False, header=True)

# Create columns to know what years the rows are valid for
ceo_040['validstart'] = ceo_040['datestartrole'].dt.year
ceo_040['validend'] = ceo_040['dateendrole'].dt.year

# Calculate tenure
ceo_040['tenure'] = ceo_040['validend'] - ceo_040['validstart']
# Identify founders
ceo_040['founder'] = ceo_040['rolename'].str.contains('founder', case=False, na=False)

# Deal with past CEOs, co-CEOs, and Title Changes
# 1- Identify Co-CEOs and assign a temporary marker for grouping
ceo_040['is_co_ceo'] = ceo_040['rolename'].str.contains('co-ceo', case=False, na=False)

# 2- Case 3 Handling: Preprocess for Co-CEOs
#    Set directorname to 'Co-CEO', directorid to 12345, and rolename to 'CO-CEO' for Co-CEOs
ceo_040.loc[ceo_040['is_co_ceo'], 'directorname'] = 'Co-CEO'
ceo_040.loc[ceo_040['is_co_ceo'], 'directorid'] = 12345
ceo_040.loc[ceo_040['is_co_ceo'], 'rolename'] = 'CO-CEO'
ceo_040.loc[ceo_040['is_co_ceo'], 'validstart'] = float('2030')

# 3- Identify CEO with the latest start date:
ceo_temp = ceo_040.sort_values(by=['ticker', 'validstart'], ascending=[True, False])
ceo_temp = ceo_temp.drop_duplicates(subset='ticker', keep='first')

ceo_v0 = ceo_040.drop(columns=['datestartrole', 'dateendrole', 'validstart', 'validend'])

# 4- Aggregate functions for all cases
aggregations = {
    'directorid': 'first',          # For Co-CEOs, it's already set to 12345 above
    'directorname': 'first',        # Already set for Co-CEOs, others remain unchanged
    'rolename': 'first',            # Keep one title for case 2, set for Co-CEOs
    'uniscoreavg': 'mean',          # Average for all cases
    'bachelors': 'max',             # OR operation for boolean columns
    'masters': 'max',
    'mba': 'max',
    'phd': 'max',
    'tenure': 'sum',
    'founder': 'max',               # OR operation for founder
    'internal_promotion': 'max',    # OR operation for internal_promotion
    'is_co_ceo': 'max'              # This will help to distinguish Co-CEO rows
}

# Remove past CEOs (keep only ones that match the ticker,directorid combination in ceo_temp)
ceo_v0 = pd.merge(ceo_v0, ceo_temp[['ticker', 'directorid']], on=['ticker', 'directorid'], how='inner')

# Dropping duplicates keeping the first occurrence, which due to sorting will be the latest
ceo_v0 = ceo_v0.drop_duplicates(subset=['ticker'], keep='first')

ceo_v0 = ceo_v0.drop(columns=['directorid', 'directorname', 'rolename'])

ceo_v0 = ceo_v0.sort_values(by=['ticker'], ascending=True)

ceo_v0.to_csv('data_pull/db/ceo_v0.csv', index=False, header=True)

end_time = time.time()
runtime = end_time - start_time
print('Runtime:', runtime)
