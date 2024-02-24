import wrds
import numpy as np
import pandas as pd
from datetime import datetime
from thefuzz import process

# Credentials to log into WRDS
db = wrds.Connection(wrds_username='s2029442')

##############################################################################################################################
######################################## SCRAPE S&P 500 COMPANIES FROM WIKIPEDIA: ############################################ 
##############################################################################################################################

link = ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies#S&P_500_component_stocks")
sp500 = pd.read_html(link, header=0)[0]

sp500 = sp500.rename(columns={'Security':'company_name', 'Symbol':'ticker'})[['ticker', 'company_name']]
sp500.to_csv('data_pull/db/s&p500.csv', index=False, header=True)
sp500 = pd.read_csv('data_pull/db/s&p500.csv')

##############################################################################################################################
###################################### CREATE CEO_V0 THROUGH THE FOLLOWING STEPS: ############################################ 
##############################################################################################################################
'''
SELECT      comp.ticker, emp.directorid, emp.directorname, emp.rolename, emp.datestartrole, emp.dateendrole, edu.companyname as university, edu.qualification
FROM        boardex.na_wrds_dir_profile_emp emp 
            INNER JOIN boardex.na_wrds_org_summary comp ON comp.boardid = emp.companyid 
            LEFT JOIN boardex.na_dir_profile_education edu ON edu.directorid = emp.directorid
GROUP BY    comp.ticker, emp.directorid, emp.directorname, emp.rolename, emp.datestartrole, emp.dateendrole, university, edu.qualification;
'''

# Get company info from BoardEx
# ceo_010 = db.raw_sql('SELECT comp.ticker, emp.directorid, emp.directorname, emp.rolename, emp.datestartrole, emp.dateendrole, edu.companyname as university, edu.qualification FROM boardex.na_wrds_dir_profile_emp emp INNER JOIN boardex.na_wrds_org_summary comp ON comp.boardid = emp.companyid LEFT JOIN boardex.na_dir_profile_education edu ON edu.directorid = emp.directorid GROUP BY comp.ticker, emp.directorid, emp.directorname, emp.rolename, emp.datestartrole, emp.dateendrole, university, edu.qualification;')

# Save data to a CSV to not have to pull data over and over again
# ceo_010.to_csv('data_pull/db/ceo_010.csv', index=False, header=True)
ceo_010 = pd.read_csv('data_pull/db/ceo_010.csv')

# Convert date columns to datetime
ceo_010['datestartrole'] = pd.to_datetime(ceo_010['datestartrole'])
ceo_010['dateendrole'] = pd.to_datetime(ceo_010['dateendrole'], errors='coerce')

# For N/A values in dateendrole (still in position), we will replace them with the current date
ceo_010['dateendrole'].fillna(pd.Timestamp(datetime.now()), inplace=True)

# Create columns to know what years the rows are valid for
ceo_010['valid_start'] = ceo_010['datestartrole'].dt.year
ceo_010['valid_end'] = ceo_010['dateendrole'].dt.year

# Perform the left join
ceo_010 = pd.merge(sp500, ceo_010, how='left', on='ticker')

# Filter rows based on conditions
# 1. ceo_010.ticker is present in sp500.ticker for the same year
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
years = range(2011, 2024)

# Initialize an empty list to store each DataFrame
dfs = []

# Loop through each year, construct the file path, read the CSV, and add the 'year' column
for year in years:
    file_path = f'data_pull/raw/uni_rankings/{year}_rankings.csv'
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

# Stror the result in a CSV
universities.to_csv('data_pull/db/universities.csv', index=False, header=True)

# Given that the universities in ceo_020 and in univeristies were given by different sources, 
# the same univeristy is mentionend differently in each, therefore, we will match names and return the best match
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
ceo_030.rename(columns={'name': 'university'}, inplace=True)

# Strore the result in a CSV
ceo_030.to_csv('data_pull/db/ceo_030.csv', index=False, header=True)

# Categorize distinct values of qualifications into appropriate degree levels
undergrad = [ 'BS', 'BA', 'BSc', 'BCom', 'BA (magna cum laude)', 'BFA (Bachelors of Fine Arts Program)', 'BSE (summa cum laude)', 'BASc', 'BTech', 'AB', 'BBA', 'BCom', 'BA (Hons)', 'BSEE', 'BA (summa cum laude)', 'BBA (Hons)', "Bachelor's Degree (Hons)", 'BComm', 'BS (cum laude)', 'BCom (Hons)', 'BAppSc (Bachelor of Applied Science)', 'BBA (summa cum laude)', "Bachelor's Degree", 'BE', 'BSBA', 'BSc (Hons)', 'BSE', 'AA', 'Graduated', 'Attended', 'Studied', 'Completed' ] 
postgraduate = [ 'MA', 'MSc', 'MEng', 'MSEE', 'MPhil', 'MPA', 'MS', 'MSME', 'MSc (Hons)', 'MS (Hons)', 'Post Graduate Diploma', 'Graduate Diploma', 'Diploma', 'Certificate', 'Certified', 'Qualified', 'Chartered', 'Licensed', 'Registered', 'Certification', 'Advanced Management Program', 'Management Development Program', 'Executive Course', 'Strategic Leadership Program', 'Leadership and Management Program', 'General Management Program', 'Higher Diploma' ] 
mba = [ 'MBA', 'Executive MBA', 'MBA (Distinction)', 'MBA (High Distinction)', 'MBA (summa cum laude)' ] 
phd = [ 'PhD', 'Doctorate', 'ScD', 'JD', 'JD (magna Cum Laude)', 'JD (Cum Laude)', 'JD (Hons)', 'LLM', 'LLB', 'Doctor of Veterinary Medicine (DVM)' ]

# Function to categorize qualifications
def categorize_qualification(row):
    qualification = str(row['qualification']).lower() # Convert to string and lowercase for comparison
    row['bachelors'] = any(qual.lower() in qualification for qual in undergrad)
    row['masters'] = any(qual.lower() in qualification for qual in postgraduate) and not row['bachelors'] # Avoids double counting if already counted as undergrad
    row['mba'] = any(qual.lower() in qualification for qual in mba)
    row['phd'] = any(qual.lower() in qualification for qual in phd)
    return row

# Apply the function across the DataFrame to have one column per degree level
ceo_030 = ceo_030.apply(categorize_qualification, axis=1)

# Drop the univeristy column
ceo_030 = ceo_030.drop(columns=['university','qualification','datestartrole','dateendrole'])

# Get average of university grade, if available, and remove duplicate rows
aggregations = {
    'uniscoreavg': 'mean',  # Average of uniscoreavg
    'bachelors': 'max',  # OR operation can be achieved by taking the max, since True > False
    'masters': 'max',    # Same logic for other boolean columns
    'mba': 'max',
    'phd': 'max',
    'valid_start': 'min',  # Minimum valid_start
    'valid_end': 'max',    # Maximum valid_end
}

# Group by year, ticker, and directorid then apply the aggregation
ceo_040 = ceo_030.groupby(['ticker', 'company_name', 'directorid', 'directorname', 'rolename']).agg(aggregations).reset_index()

# Identify founders
ceo_040['founder'] = ceo_040['rolename'].str.contains('founder', case=False, na=False)

# Add column internal promotion to see if the CEO was promoted from within the company: 
# Get data from BoardEx
'''
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
            , non_ceo.rolename AS non_role
            , ceo.datestartrole
            , non_ceo.datestartrole AS non_startrole 
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

# promotion = db.raw_sql("with temp as (SELECT comp.ticker, emp.directorid, emp.rolename, emp.datestartrole FROM boardex.na_wrds_dir_profile_emp emp INNER JOIN boardex.na_wrds_org_summary comp ON comp.boardid = emp.companyid GROUP BY comp.ticker, emp.directorid, emp.rolename, datestartrole) SELECT ceo.ticker, ceo.directorid, ceo.rolename, non_ceo.rolename as non_role, ceo.datestartrole, non_ceo.datestartrole as non_startrole FROM temp ceo LEFT JOIN temp AS non_ceo ON non_ceo.directorid = ceo.directorid AND ceo.ticker = non_ceo.ticker GROUP BY ceo.ticker, ceo.directorid, ceo.rolename, non_role, ceo.datestartrole, non_startrole;")
# promotion.to_csv('data_pull/db/promotion_010.csv', index=False, header=True)
promotion = pd.read_csv('data_pull/db/promotion_010.csv')

promotion = promotion[
    (promotion['rolename'].str.contains('CEO', case=False)) &
    (promotion['non_role'].str.contains('CEO', case=False) == False) &
    (promotion['datestartrole'] >= promotion['non_startrole']) &
    (promotion['directorid'].notna())]

promotion = promotion[['ticker', 'directorid', 'rolename']]

# Remove duplicates
promotion = promotion.drop_duplicates()
promotion.to_csv('data_pull/db/promotion_020.csv', index=False, header=True)

# Combine to add column internal_promotion to ceo_040
promotion_020 = pd.read_csv('data_pull/db/promotion_020.csv')

# Perform a left join between ceo_040 and promotion_020
# Use 'indicator' argument to add a temporary '_merge' column indicating whether the merge operation found a match or not
ceo_040 = pd.merge(ceo_040, promotion_020, on=['ticker', 'directorid', 'rolename'], how='left', indicator=True)

# # Create 'internal_promotion' column based on whether a match was found
ceo_040['internal_promotion'] = ceo_040['_merge'] == 'both'

# # Drop the temporary '_merge' column as it's no longer needed
ceo_040.drop(columns=['_merge'], inplace=True)

# Strore the result in a CSV
ceo_040.to_csv('data_pull/db/ceo_040.csv', index=False, header=True)

#################################### Deal with past CEOs, co-CEOs, and Title Changes #########################################

# Identify Co-CEOs and assign a temporary marker for grouping
ceo_040['is_co_ceo'] = ceo_040['rolename'].str.contains('co-ceo', case=False, na=False)

# Case 3 Handling: Preprocess for Co-CEOs
# Set directorname to 'Co-CEO', directorid to 12345, and rolename to 'CO-CEO' for Co-CEOs
ceo_040.loc[ceo_040['is_co_ceo'], 'directorname'] = 'Co-CEO'
ceo_040.loc[ceo_040['is_co_ceo'], 'directorid'] = 12345
ceo_040.loc[ceo_040['is_co_ceo'], 'rolename'] = 'CO-CEO'

# Identify CEO with latest start date:
ceo_temp = ceo_040.sort_values(by=['ticker', 'valid_start'], ascending=[True, False])
ceo_temp = ceo_temp.drop_duplicates(subset='ticker', keep='first')

# Aggregate functions for all cases
aggregations = {
    'company_name': 'first',  # Keep the company name as is
    'directorid': 'first',  # For Co-CEOs, it's already set to 12345 above
    'directorname': 'first',  # Already set for Co-CEOs, others remain unchanged
    'rolename': 'first',  # Keep one title for case 2, set for Co-CEOs
    'uniscoreavg': 'mean',  # Average for all cases
    'bachelors': 'max',  # OR operation for boolean columns
    'masters': 'max',
    'mba': 'max',
    'phd': 'max',
    'valid_start': 'min',  # Minimum valid_start
    'valid_end': 'max',  # Maximum valid_end
    'founder': 'max',  # OR operation for founder
    'internal_promotion': 'max',  # OR operation for internal_promotion
    'is_co_ceo': 'max'  # This will help to distinguish Co-CEO rows
}

# Remove past CEOs (keep only ones that match the ticker,directorid combination in ceo_temp)
ceo_050 = pd.merge(ceo_040, ceo_temp[['ticker', 'directorid']], on=['ticker', 'directorid'], how='inner')

# Dropping duplicates keeping the first occurrence, which due to sorting will be the latest
ceo_050 = ceo_050.drop_duplicates(subset=['ticker'], keep='first')

ceo_050 = ceo_050.sort_values(by=['ticker'], ascending=True)

# Calculate tenure
ceo_050['tenure'] = ceo_040['valid_end'] - ceo_040['valid_start']
                                                   
ceo_050.to_csv('data_pull/db/ceo_050.csv', index=False, header=True)

##############################################################################################################################
###################################### CREATE BOARD_V0 THROUGH THE FOLLOWING STEPS: ########################################## 
##############################################################################################################################

# Get number of committes from BoardEx
'''
SELECT      comp.ticker, count(distinct(comm.committeename)) as num_committees, count(distinct(comm.directorid)) as dirs
FROM        boardex.na_board_dir_committees comm 
            LEFT JOIN boardex.na_wrds_org_summary comp ON comm.boardid = comp.boardid            
GROUP BY    comp.ticker;
'''
comms = db.raw_sql('SELECT comp.ticker, COUNT(DISTINCT comm.committeename) AS num_committees, COUNT(DISTINCT comm.directorid) AS dirs FROM boardex.na_board_dir_committees comm LEFT JOIN boardex.na_wrds_org_summary comp ON comm.boardid = comp.boardid GROUP BY comp.ticker;')

# Get info of directors on boards from ISS ESG Directors
'''
WITH temp AS (SELECT dir.ticker, case when dir.attend_less75_pct='Yes' then 0 else 1 end as attend, dir.meetingdate FROM risk.rmdirectors dir WHERE dir.year = 2023.0)
SELECT      t.ticker, sum(t.attend)/count(t.attend) as over75_pct, count(t.meetingdate) as num_meetings
FROM        temp t
GROUP BY    t.ticker;
'''
dirs = db.raw_sql('WITH temp AS (SELECT dir.ticker, case when dir.attend_less75_pct=\'Yes\' then 0 else 1 end as attend, dir.meetingdate FROM risk.rmdirectors dir WHERE dir.year = 2023.0) SELECT t.ticker, sum(t.attend)/count(t.attend) as over75_pct, count(t.meetingdate) as num_meetings FROM temp t GROUP BY t.ticker;')

# Join tables to only keep sp500 companies
board_020 = pd.merge(sp500, comms, how='left', on='ticker')
board_020 = pd.merge(board_020, dirs, how='left', on='ticker')

board_020['avg_meetings']=board_020['num_meetings']/board_020['num_committees']

# Save data to a CSV to not have to pull data over and over again
board_020.to_csv('data_pull/db/board_020.csv', index=False, header=True)

##############################################################################################################################
###################################### CREATE DIVERSITY THROUGH THE FOLLOWING STEPS: ######################################### 
##############################################################################################################################

'''
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
# div_010 = db.raw_sql('WITH dir AS (SELECT dir.ticker, dir.director_detail_id, dir.female, dir.ethnicity FROM risk.rmdirectors dir WHERE dir.year = 2023.0) SELECT dir.ticker, dir.director_detail_id, dir.female, dir.ethnicity, comm.committeename FROM dir LEFT JOIN boardex.na_wrds_org_summary comp ON comp.ticker = dir.ticker LEFT JOIN boardex.na_board_dir_committees comm ON comm.boardid = comp.boardid GROUP BY dir.ticker, dir.director_detail_id, dir.female, dir.ethnicity, comm.committeename;')

# Save data to a CSV to not have to pull data over and over again
# div_010.to_csv('data_pull/db/div_010.csv', index=False, header=True)
div_010 = pd.read_csv('data_pull/db/div_010.csv')

# Convert 'committeename' and 'female' columns to strings, replacing NaN with an empty string
# div_010['committeename'] = div_010['committeename'].fillna('').astype(str)
div_010['female'] = div_010['female'].apply(lambda x: 'No' if x != 'Yes' else 'Yes')

# Create the 'women_pct' column
female_yes_count = div_010[div_010['female'] == 'Yes'].groupby('ticker')['female'].count().reset_index(name='female')
total_count = div_010.groupby('ticker')['female'].count().reset_index(name='total')
female = pd.merge(female_yes_count, total_count, on='ticker')
female['female_pct'] = female['female'] / female['total']
female = female[['ticker', 'female_pct']]

# Create the 'diversity_board' column
diversity_keywords = r'divers|incl|sustai'
div_010['diversity_board'] = div_010['committeename'].str.contains(diversity_keywords, case=False, regex=True)
diversity = div_010.groupby('ticker')['diversity_board'].any().reset_index()

# Merge the two dataframes female and diversity
div_020 = pd.merge(female, diversity, on='ticker')

# Strore the result in a CSV
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
div_030 = pd.merge(div_020, shannon, on='ticker')

# Left join to keep only the rows in div_020 with a corresponding ticker in sp500
div_030 = pd.merge(sp500, div_030, how='left', on='ticker')

# Strore the result in a CSV
div_030.to_csv('data_pull/db/div_030.csv', index=False, header=True)