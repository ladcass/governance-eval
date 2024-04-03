import pandas as pd

full = pd.read_csv('data_pull/db/full_v0.csv')
final = pd.read_csv('data_pull/db/final.csv')

# Print the proportion of each class predicted in roapredicted column
print(final['roapredicted'].value_counts(normalize=True) * 100)
print(full['roachange'].value_counts(normalize=True) * 100)

