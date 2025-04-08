# import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

## Let er op, de Closing prices hebben geen currency teken meer

# Read the data
main_df = pd.read_csv('Data/Cleaned_Indices_Assignment1.csv', sep=';')

# Read the interest rate data
interest_rate_df = pd.read_csv('Data/ECB_Rates_2012_to_2022.csv', sep=';')

# Convert date columns to datetime format for proper merging
main_df['Date'] = pd.to_datetime(main_df['Date'], format='%d-%m-%Y')
interest_rate_df['Date'] = pd.to_datetime(interest_rate_df['Date'], format='%d-%m-%Y')

# Merge the dataframes on the Date column
main_df = pd.merge(main_df, interest_rate_df, on='Date', how='left')

# Check the first few rows of the merged dataframe
print(main_df.head(15))