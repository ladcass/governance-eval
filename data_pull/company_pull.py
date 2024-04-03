import pandas as pd 
import wrds
import time

start_time = time.time()

db = wrds.Connection(wrds_username='s2029442')

##############################################################################################################################
######################################## SCRAPE S&P 500 COMPANIES FROM WIKIPEDIA: ############################################ 
##############################################################################################################################

link = ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies#S&P_500_component_stocks")
sp500 = pd.read_html(link, header=0)[0]

sp500 = sp500.rename(columns={'Security':'company_name', 'Symbol':'ticker','GICS Sector':'GICS'})[['ticker', 'company_name','GICS']]
sp500.to_csv('data_pull/db/s&p500.csv', index=False, header=True)

##############################################################################################################################
############################################### GET S&P INDEX OF COMPANIES: ################################################## 
##############################################################################################################################

companies = db.raw_sql("SELECT ticker, spindex FROM risk_governance.rmgovernance WHERE year = 2023")
companies.to_csv('data_pull/db/companies.csv', index=False, header=True)

end_time = time.time()
runtime = end_time - start_time
print('Runtime:', runtime)
