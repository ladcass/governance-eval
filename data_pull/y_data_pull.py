import wrds
import numpy as np
import pandas as pd
from datetime import datetime
from thefuzz import process

# Credentials to log into WRDS
db = wrds.Connection(wrds_username='s2029442')

##############################################################################################################################
################################################## GET Y VALUES FOR DATASETS: ################################################ 
##############################################################################################################################

######################################################## ROA & ROE ###########################################################

# Get ROA from WRDS Financial Ratios Firm Level
'''
SELECT      DISTINCT  
            fr.ticker
            , fr.adate
            , fr.roa
FROM        wrdsapps_finratio.firm_ratio fr;
'''

return_010 = db.raw_sql("SELECT DISTINCT fr.ticker, fr.adate, fr.roa, fr.roe FROM wrdsapps_finratio.firm_ratio fr;")
return_010.to_csv('data_pull/db/roa.csv', index=False, header=True)
# return_010 = pd.read_csv('data_pull/db/return_010.csv')

return_010['adate'] = pd.to_datetime(return_010['adate'])

# Sort the DataFrame based on 'ticker' and 'adate' (descending)
return_020 = return_010.sort_values(by=['ticker', 'adate'], ascending=[True, False])

# Remove blank 'roa' entries
return_020 = return_020.dropna(subset=['roa'])

# Drop duplicate 'ticker' entries, keeping the first entry (which has the max 'adate' due to sorting)
return_020 = return_020.drop_duplicates(subset=['ticker'], keep='first')
return_020 = return_020.drop(columns=['adate'])

# Save the DataFrame to a CSV file
return_020.to_csv('data_pull/db/return_020.csv', index=False, header=True)

######################################################### TOBIN-Q ############################################################

# Get Tobin Q from Compustat Daily All - Fundamentals Annual
'''
SELECT      f.tic
            , f.at/f.mkvalt   as tobin_q
FROM        comp_na_daily_all.funda f
WHERE      f.fyear = 2023;
'''

tobin = db.raw_sql("SELECT DISTINCT f.tic, f.at/f.mkvalt AS tobin_q FROM comp_na_daily_all.funda f WHERE f.fyear = 2023;")
tobin.to_csv('data_pull/db/tobin_q.csv', index=False, header=True)
# tobin = pd.read_csv('data_pull/db/tobin_q.csv')

##############################################################################################################################
############################################### ADD ROA AND TOBIN-Q TO CEO_V0: ############################################### 
##############################################################################################################################

ceo_050 = pd.read_csv('data_pull/db/ceo_040.csv')

ceo_v0 = pd.merge(ceo_050, return_020, how='left', left_on='ticker', right_on='ticker')
ceo_v0 = pd.merge(ceo_v0, tobin, how='left', left_on='ticker', right_on='tic')

# Drop tic and roe columns
ceo_v0 = ceo_v0.drop(columns=['tic','roe'])

ceo_v0.to_csv('data_pull/db/ceo_v0.csv', index=False, header=True)

##############################################################################################################################
###################################################### ADD ROE TO BOARD_V0: ##################################################
##############################################################################################################################

board_020 = pd.read_csv('data_pull/db/board_020.csv')

board_v0 = pd.merge(board_020, return_020, how='left', on='ticker')

# Drop tic and roe columns
board_v0 = board_v0.drop(columns=['roa'])

board_v0.to_csv('data_pull/db/board_v0.csv', index=False, header=True)

##############################################################################################################################
###################################################### ADD ROA TO DIV_V0: ####################################################
##############################################################################################################################

div_020 = pd.read_csv('data_pull/db/div_020.csv')
div_v0 = pd.merge(div_020, return_020, how='left', on='ticker')

div_v0 = div_v0.drop(columns=['roe'])

div_v0.to_csv('data_pull/db/div_v0.csv', index=False, header=True)
