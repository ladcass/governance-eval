import pandas as pd
import time

start_time = time.time()

ceo_v0 = pd.read_csv('data_pull/db/ceo_v0.csv')
div_v0 = pd.read_csv('data_pull/db/div_v0.csv')
div_v0.drop(columns=['spindex'], inplace=True)
board_v0 = pd.read_csv('data_pull/db/board_v0.csv')
roa_v0 = pd.read_csv('data_pull/db/roa_v0.csv')

full_v0 = pd.merge(ceo_v0, div_v0, on=['ticker'])
full_v0 = pd.merge(full_v0, board_v0, on=['ticker'])
full_v0 = pd.merge(full_v0, roa_v0, on=['ticker'])

full_v0.to_csv('data_pull/db/full_v0.csv', index=False, header=True)

end_time = time.time()
runtime = end_time - start_time
print('Runtime:', runtime)
