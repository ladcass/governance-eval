# Dissertation: Creating a Data Pipeline to Automate the Creation of a Dataset to Evaluate Corporate Governance of S\&P 1500 Companies
Final year dissertation project. Creating the data pipeline to create a comprehensive dataset to evaluate the corporate governance of select companies.

The files are separated in the following logic:
1) Data pull
 1.1) company_pull.py -- Gets all the base information of the companies we will be looking at and creates db/companies.csv
 
 1.2) ceo_pull.py     -- Gathers information on CEO background and creates db/ceo_v0.csv
 
 1.3) board_pull.py   -- Gathers information on board structure and creates db/board_v0.csv
 
 1.4) div_pull.py     -- Gathers information on diversity within directors and creates db/div_v0.csv
 
 1.5) roa_pull.py     -- Gathers the target variable (ROA) for the appropriate companies and creates db/roa.csv
 
 1.6) full.py         -- Combines all subtables generated by previous Python scripts, to finalize the dataset with target and explanatory variables. It generates db/full_v0.csv
   
3) Model training & evaluation
 2.1) experiments.py         -- Runs all models being tested and records their accuracies in .txt files in the accuracies directory

 2.2) experiment_results.py  -- Runs the statistical tests to identify the best-performing model
 
 2.3) final_model.py         -- Uses the best-performing model to predict the ROA change, creating db/final.csv as well as recording the LIME explanations in the lime_explanations directory

