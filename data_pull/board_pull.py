import wrds
import pandas as pd
import time

start_time = time.time()
# Credentials to log into WRDS
db = wrds.Connection(wrds_username='s2029442')

# Get company list
companies = pd.read_csv('data_pull/db/companies.csv')
companies.dropna()

# Get number of committees from BoardEx
''''
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


            CHECK STRUCTURE OF DIRECTORS US. IF MORE THAN 1 ROW PER DIR PER MEETING CHANGE IT! THIS IS WRONG! GROUP BY DIRECTOR ID AND BOARD
'''

dirs = db.raw_sql("WITH temp AS (SELECT dir.ticker, case when dir.attend_less75_pct='Yes' then 0 else 1 end as attend, dir.meetingdate FROM risk.rmdirectors dir WHERE dir.year = 2023.0) SELECT t.ticker, sum(t.attend)/count(t.attend) as over75_pct, count(t.meetingdate) as num_meetings FROM temp t GROUP BY t.ticker;")

# Join tables to only keep companies wanted
board_v0 = pd.merge(companies, comms, how='left', on='ticker')
board_v0 = pd.merge(board_v0, dirs, how='left', on='ticker')

board_v0.drop(columns=['spindex'])

board_v0['avg_meetings'] = board_v0['num_meetings'] / board_v0['num_committees']

# Save data to a CSV to not have to pull data over and over again
board_v0.to_csv('data_pull/db/board_v0.csv', index=False, header=True)

end_time = time.time()
runtime = end_time - start_time
print('Runtime:', runtime)
