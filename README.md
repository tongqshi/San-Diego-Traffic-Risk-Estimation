# San-Diego-Traffic-Risk-Estimation
Visualization of San Diego traffic risk based on historical collision data

# Data sources:
pd_beat_codes_list_datasd.csv: 	https://data.sandiego.gov/datasets/police-beats/

pd_collisions_details_datasd.csv: https://data.sandiego.gov/datasets/police-collisions-details/

traffic_counts_datasd.csv: https://data.sandiego.gov/datasets/traffic-volumes/

# Files

data_visualization.ipynb: Visualization of time and street information in collisions.

Classification.py: A classification model using Logistic Regression to predict whether there will be a collision.  

Regression.py: A regression model using Linear Regression to predict the number of people who will be injured by the collision.  

traffic_counts_datasd.csv: original dataset including traffic count data.  

traffic_counts_processed.csv: processed dataset of traffic_counts_datasd.csv.  

pd_collisions_details_datasd.csv: original dataset including the collisions.  

pd_beat_codes_list_datasd.csv: original dataset including the police beat codes

collision_reports_processed.csv: cleaned traffic_account_datasd.csv file that will be used in predictions.  

data_preprocessing.py: processing for the original dataset.  

data_statistic.py: Apply probability and statistical methods to the collision data.

# How to run these scripts
## For data analysis:

data_preprocessing.py:
1. Download pd_collisions_details_datasd.csv and traffic_counts_datasd.csv
2. Ensure the script and datasets are the in the same directory
3. Ensure all third party modules are installed
4. Run

data_visualization.ipynb:
1. Download all raw data files
2. Ensure the script and datasets are the in the same directory or change the file path in the read_csv function
3. Ensure all third party modules are installed
4. More detailed description step by step can be found in the ipynb file, hope you can find something interesting results in this file.
5. Run

data_statistic.py:
1. Download collision_reports_processed.csv and traffic_counts_processed.csv file
2. Ensure the script and datasets are the in the same directory or change the file path in the read_csv function
3. Ensure all third party modules are installed
4. More detailed description step by step can be found in the python file, hope you can find something interesting results in this file.
5. Run

## For Regression.py and Classification.py:
1. Download the collision_report_processed.csv
2. Download the Classification.py and Regression.py
3. Put them into the same dictionary.
4. Run these two python files and you will get the output.

# Third Party Modules
csv  
Ridge  
numpy  
LogisticRegression  
random  
matplotlib.pyplot  
