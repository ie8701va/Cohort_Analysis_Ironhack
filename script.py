# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 18:58:30 2024

@author: ievav
"""
#%%# Importing necessary packages and data

import matplotlib.pyplot as plt  # For creating static, animated, and interactive visualizations
import pandas as pd  # For data manipulation and analysis, particularly with DataFrames
import numpy as np  # For numerical operations and handling arrays
from ydata_profiling import ProfileReport  # For generating comprehensive data profiling reports
import webbrowser  # For opening URLs in the web browser
import seaborn as sns  # For statistical data visualization, built on Matplotlib
import plotly.express as px  # For creating interactive plots and visualizations

# Set the color for plots to create coherence
color = '#007A7C'  

# Specify the desired bar width for subsequent plotting
bar_width = 0.5 

# Load datasets from CSV files into pandas DataFrames
cash_request = pd.read_csv('extract - cash request - data analyst.csv')  # Cash request data
fees = pd.read_csv('extract - fees - data analyst - .csv')  # Fees data

# Display the first few rows of each DataFrame for initial inspection
print(cash_request.head())  # Preview cash request data
print(fees.head())  # Preview fees data

# Check the data types of the columns in cash_requests to understand the data structure
print("Data types in cash_requests:")
print(cash_request.dtypes)

# Check the data types of the columns in fees to verify data types
print("\nData types in fees:")
print(fees.dtypes)

len(fees)
len(cash_request)
#%%# Handling missing values in both dataframes


# Check for missing values in both datasets and summarize the count of missing entries
print(cash_request.isnull().sum())  # Count missing values in cash_request
print(fees.isnull().sum())  # Count missing values in fees


print(f'Null values in user_id of cash_request: {cash_request["user_id"].isnull().sum()}. That\'s how many accounts were deleted')  # Check for null user_ids
print(f'The number of accounts that were deleted: {np.sum(cash_request["deleted_account_id"].value_counts())}') # Display the number of accounts that were deleted
# Seems that there is one account that has been deleted  yet still has a different user_id  == 3161.


# Save rows where 'user_id' is NaN in cash_request to a separate DataFrame for record-keeping
cash_request_deleted_accounts = cash_request[cash_request['user_id'].isna()] 

# Replace NaN values in 'user_id' with corresponding values from 'deleted_account_id'
cash_request['user_id'].fillna(cash_request['deleted_account_id'], inplace=True)

# # Save rows where 'cash_request_id' is NaN in fees to a separate DataFrame for record-keeping
fees_na = fees[fees['cash_request_id'].isna()]

# Find the rows where 'cash_request_id' is NaN
na_rows = fees['cash_request_id'].isna()

# Extract the last 5 characters from 'reason' where 'cash_request_id' is NaN
fees.loc[na_rows, 'cash_request_id'] = fees.loc[na_rows, 'reason'].str[-5:]
fees['cash_request_id'] = pd.to_numeric(fees['cash_request_id']) # Converting the column to numerical

# Confirming that there are no empty values in cash_request_id in fees dataframe
print(f'The number of NAs in cash_request_id fees dataframe is {fees["cash_request_id"].isnull().sum()}.')

# Display the number of na rows for transparency
print(f'Before, there were {len(fees_na)} NAs in fees.')  # Report on na cash request id in fees dataframe


#%%# Checking duplicate rows in ids

# Check for duplicates in cash_request and fees DataFrames to ensure data integrity
print(f'Duplicated rows in cash_request (by id): {cash_request.duplicated(subset=["id"]).sum()}')  # Duplicates by 'id'
print(f'Duplicated rows in cash_request (by user_id): {cash_request.duplicated(subset=["user_id"]).sum()}. This number of clients returned for another cash request.')  # Duplicates by 'user_id'
print(f'Duplicated rows in fees (by id): {fees.duplicated(subset=["id"]).sum()}.')  # Duplicates in fees by 'id'


#%%# Exploring cash requests dataframe

# Count unique occurrences of different statuses in cash_request for understanding data distribution
unique_status_counts = cash_request['status'].value_counts()  # Count status occurrences
unique_recovery_status_values = cash_request['recovery_status'].value_counts()  # Count recovery status occurrences
unique_transfer_type_counts = cash_request['transfer_type'].value_counts()  # Count transfer type occurrences
unique_id_counts = cash_request['id'].value_counts()  # Count occurrences of unique IDs

# Print counts of unique statuses for reporting
print(unique_status_counts)  # Print status counts
print(unique_recovery_status_values)  # Print recovery status counts
print(unique_transfer_type_counts)  # Print transfer type counts
print(unique_id_counts)  # Print ID counts - no repeating data points 


#%%# Exploring fees dataframe

# Count unique occurrences of different statuses in fees for understanding data distribution
unique_fee_type_counts = fees['type'].value_counts()  # Count fee type occurrences
unique_fee_status_values = fees['status'].value_counts()  # Count fee status occurrences
unique_fee_category_counts = fees['category'].value_counts()  # Count payment category type occurrences
unique_fee_charge_moment_counts = fees['charge_moment'].value_counts()  # Count occurrences of when fees should be charged

# Print counts of unique statuses for reporting
print(unique_fee_type_counts)  # Print fee type  counts
print(unique_fee_status_values)  # Print fee status counts
print(unique_fee_category_counts)  # Print payment category type counts
print(unique_fee_charge_moment_counts)  # Print when fees should be charged counts


#%%# Preliminary vizualizations 

# Plot the values to have a better idea about the data visually

plt.figure(figsize=(16, 12))  # Adjust the size to fit all plots

plt.subplot(2, 4, 1)
sns.countplot(x=fees['type'], color=color, width=bar_width)
plt.title('Fee Types')
plt.xticks(rotation=45)

plt.subplot(2, 4, 2)
fee_type_sum = fees.groupby('type')['total_amount'].sum().sort_values(ascending=False)
sns.barplot(x=fee_type_sum.index, y=fee_type_sum.values, color=color, width=bar_width)
plt.title('Total Amount Owed by Fee Type')
plt.xticks(rotation=45)

plt.subplot(2, 4, 3)
sns.countplot(x=fees['status'], color=color, width=bar_width)
plt.title('Fee Status')
plt.xticks(rotation=45)

plt.subplot(2, 4, 4)
sns.countplot(x=fees['category'], color=color, width=bar_width)
plt.title('Fee Categories')
plt.xticks(rotation=45)

plt.subplot(2, 4, 5)
sns.countplot(x=fees['charge_moment'], color=color, width=bar_width)
plt.title('Fee Charge Moment')
plt.xticks(rotation=45)

plt.subplot(2, 4, 6)
sns.countplot(x=cash_request['recovery_status'], color=color, width=bar_width)
plt.title('Cash Request Recovery Status')
plt.xticks(rotation=45)

plt.subplot(2, 4, 7)
sns.countplot(x=cash_request['transfer_type'], color=color, width=bar_width)
plt.title('Cash Request Transfer Type')
plt.xticks(rotation=45)

plt.subplot(2, 4, 8)
sns.countplot(x=cash_request['status'], color=color, width=bar_width)
plt.title('Cash Request Status')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

#%%# Date Format

# List of date columns to convert to datetime format for consistent date handling
date_columns_format = ['created_at', 'updated_at', 'moderated_at', 'reimbursement_date', 
                       'send_at', 'paid_at', 'from_date', 'to_date', 
                       'money_back_date', 'reco_last_update', 'reco_creation', 
                      ]

# Convert each date column to datetime format in cash_request for accurate date operations
for column in date_columns_format:
    if column in cash_request:
        cash_request[column] = pd.to_datetime(cash_request[column], format='ISO8601', errors='coerce')
  
# Convert each date column to datetime format in fees for accurate date operations
for column in date_columns_format:
    if column in fees:            
        fees[column] = pd.to_datetime(fees[column], format='ISO8601')

#%%# ProfileReport package to visualize the data 


# cash_request_profile = ProfileReport(cash_request, title="cash_request_Profiling Report")
# # Display the report in a web browser
# cash_request_profile.to_file("cash_request_profiling_report.html")
# webbrowser.open("cash_request_profiling_report.html")

# fees_profile = ProfileReport(fees, title="Profiling Report")
# # Display the report in a web browser
# fees_profile.to_file("fees_profiling_report.html")
# webbrowser.open("fees_profiling_report.html")


#%%%# Cash Request Frequnecy Cohort Analysis

# Step 1: Find the earliest request date for each user to create a cohort month
cash_request['cohort_month'] = cash_request.groupby('user_id')['created_at'].transform('min')

# Step 2: Convert cohort month to year-month format for easier analysis (to_period)
cash_request['cohort_month'] = cash_request['cohort_month'].dt.to_period('M')

# Display the new column for inspection to ensure correctness
print(cash_request['cohort_month'])

# Step 3: Create the 'usage_month' column for the month the service was used, formatted as year-month
cash_request['usage_month'] = cash_request['created_at'].dt.to_period('M')

# Step 4: Calculate the 'cohort_index' for each request (number of months since the cohort month)
cash_request['cohort_index'] = [(usage - cohort).n for usage, cohort in zip(cash_request['usage_month'], cash_request['cohort_month'])]

# Step 5: Group by cohort month and cohort index, and count the number of requests for cohort analysis
cohort_usage_frequency_table = cash_request.groupby(['cohort_month', 'cohort_index']).size().unstack(fill_value=0)  # Pivot the data for better readability


# Step 10: Plotting the cohort usage frequency as a heatmap for visual analysis
plt.figure(figsize=(10, 6))  # Set the figure size for the heatmap
sns.heatmap(cohort_usage_frequency_table, annot=True, fmt='d', cmap='Blues',  # Create a heatmap with annotations
            cbar_kws={'label': 'Number of Requests', 'pad': 0.01},  # Add color bar with label
            linewidths=0.5, linecolor='white')  # Set linewidth and line color for clarity

# Set titles and labels for the heatmap
plt.title('Cash Request Frequency by Cohort', pad=20, fontsize=16)  # Set the title
plt.xlabel('Months since the User\'s First Cash Request', labelpad=15, fontsize=14)  # Set the x-axis label
plt.ylabel('Acquisition Month', labelpad=15, fontsize=14)  # Set the y-axis label
plt.tight_layout()  # Adjust layout for better appearance

# Save the plot with a transparent background
plt.savefig('heatmap.png', transparent=True, dpi=300)  # Save the figure with a transparent background

plt.show()  # Display the plot



#%%# All types of fees frequency cohort analysis

# Step 1: Convert 'cash_request_id' to int64 data type to ensure consistency with the cash_requests DataFrame
fees['cash_request_id'] = fees['cash_request_id'].astype('int64')

# Verify the data type change in the fees DataFrame
print(fees.dtypes)

# Step 2: Perform an inner merge between cash_request and fees DataFrames
fees_alltypes_cohort_analysis = pd.merge(
    cash_request[['id', 'cohort_month']],  # Select 'id' and 'cohort_month' columns from cash_request for merging
    fees[['cash_request_id', 'type', 'created_at']],  # Select 'cash_request_id', 'type', and 'created_at' columns from fees
    left_on='id',  # Use 'id' from cash_request as the left join key
    right_on='cash_request_id',  # Use 'cash_request_id' from fees as the right join key
    how='inner'  # Specify an inner join to keep only matching records from both DataFrames
)

# Display the resulting merged DataFrame for verification
print(fees_alltypes_cohort_analysis)

# Step 3: Create the 'incident_month' column to represent the month when the fee was charged
fees_alltypes_cohort_analysis['incident_month'] = fees_alltypes_cohort_analysis['created_at'].dt.to_period('M')

# Step 4: Calculate the 'incident_index' for each request, representing the number of months since the cohort month
fees_alltypes_cohort_analysis['incident_index'] = [
    (incident - cohort).n for incident, cohort in zip(fees_alltypes_cohort_analysis['incident_month'], fees_alltypes_cohort_analysis['cohort_month'])
]

# Step 5: Group the data by cohort month and incident index, counting the number of incidents
fees_alltypes_cohort_analysis_table = fees_alltypes_cohort_analysis.groupby(
    ['cohort_month', 'incident_index']
).size().unstack(fill_value=0)  # Pivot the data to create a table of counts for easy visualization

# Display the cohort analysis table to observe fee occurrences over time
print(fees_alltypes_cohort_analysis_table)

# Step 6: Plotting the cohort fee frequency as a heatmap for visual analysis
plt.figure(figsize=(10, 6))  # Set the figure size for the heatmap

# Create a heatmap to visualize the frequency of fees by cohort month and incident index
plt.figure(figsize=(10, 6))  # Set the figure size for the heatmap
sns.heatmap(
    fees_alltypes_cohort_analysis_table,
    annot=True,  # Annotate the cells with the counts
    fmt='d',  # Format the annotation as integers
    cmap='Reds',  # Set the color palette for the heatmap
    cbar_kws={'label': 'Number of Incidents', 'pad': 0.01},  # Add a color bar with a label and padding
    linewidths=0.5,  # Set the width of the lines separating the cells
    linecolor='white'  # Set the color of the lines separating the cells
)

# Adding titles and labels with increased font size for clarity
plt.title('The Frequency of all Types of Fees', pad=20, fontsize=16)  # Title of the heatmap
plt.xlabel('Months since the User\'s First Cash Request', labelpad=15, fontsize=14)  # X-axis label
plt.ylabel('Acquisition Month', labelpad=15, fontsize=14)  # Y-axis label

# Adjust layout to prevent clipping of titles and labels
plt.tight_layout()  # Adjust layout for better presentation

# Save the plot with a transparent background
plt.savefig('fees_heatmap.png', transparent=True, dpi=300)  # Save the figure with a transparent background

plt.show()  # Display the heatmap

#%%# Incidence Frequency Analysis

# Step 1: Filter the DataFrame to keep only rows where the type is 'incident'
fees_incidents_cohort_analysis = fees_alltypes_cohort_analysis[fees_alltypes_cohort_analysis['type'] == 'incident']

# Step 2: Create the 'incident_month' column to represent the month when the incident occurred
fees_incidents_cohort_analysis['incident_month'] = fees_incidents_cohort_analysis['created_at'].dt.to_period('M')

# Step 3: Calculate the 'incident_index' for each request, indicating the number of months since the cohort month
fees_incidents_cohort_analysis['incident_index'] = [(incident - cohort).n for incident, cohort in zip(fees_incidents_cohort_analysis['incident_month'], fees_incidents_cohort_analysis['cohort_month'])]

# Step 4: Group the data by cohort month and incident index, counting the number of incidents
fees_incidents_cohort_analysis_table = fees_incidents_cohort_analysis.groupby(['cohort_month', 'incident_index']).size().unstack(fill_value=0)  # Pivot the grouped data to create a table of counts for visualization

# Display the cohort analysis table to observe incident occurrences over time
print(fees_incidents_cohort_analysis_table)

# Step 5: Plotting the cohort incident frequency as a heatmap for visual analysis
plt.figure(figsize=(10, 6))  # Set the figure size for the heatmap

# Create a heatmap to visualize the frequency of incidents by cohort month and incident index
sns.heatmap(
    fees_incidents_cohort_analysis_table,
    annot=True,  # Annotate the cells with the counts
    fmt='d',  # Format the annotations as integers
    cmap='PuRd',  # Set the color palette for the heatmap
    cbar_kws={'label': 'Number of Incidents', 'pad': 0.01},  # Add a color bar with a label and padding
    linewidths=0.5,  # Set the width of the lines separating the cells
    linecolor='white'  # Set the color of the lines separating the cells
)

# Adding titles and labels with increased font size for clarity
plt.title('Incident Frequency Since Acquisition', pad=20, fontsize=16)  # Title of the heatmap
plt.xlabel('Months since the User\'s First Cash Request', labelpad=15, fontsize=14)  # X-axis label
plt.ylabel('Acquisition Month', labelpad=15, fontsize=14)  # Y-axis label

# Adjust layout to prevent clipping of titles and labels
plt.tight_layout()  # Ensure a clean layout for the plot

# Save the plot with a transparent background
plt.savefig('incident_frequency_heatmap.png', transparent=True, dpi=300)  # Save the figure with a transparent background

plt.show()  # Display the heatmap


#%%# Ratio between incidence and number of cash requests 

# Step 1: Calculate the ratio of incident frequency to cash request frequency
ratio_incident_to_cash_request = fees_incidents_cohort_analysis_table / cohort_usage_frequency_table

# Convert all NaN values to 0
ratio_incident_to_cash_request.fillna(0, inplace=True)

# Round values to two decimal places
ratio_incident_to_cash_request = ratio_incident_to_cash_request.round(3)

# Step 2: Plotting the ratio of incident frequency to cash requests as a heatmap
plt.figure(figsize=(10, 6))  # Set the figure size for the heatmap

# Create a heatmap to visualize the ratio of incidents to cash requests
sns.heatmap(
    ratio_incident_to_cash_request,
    annot=True,  # Annotate the cells with the ratio values
    cmap='Reds',  # Set the color palette for the heatmap
    cbar_kws={'label': 'Ratio', 'pad': 0.01},  # Add a color bar with a label and padding
    linewidths=0.5,  # Set the width of the lines separating the cells
    linecolor='white'  # Set the color of the lines separating the cells
)

# Adding titles and labels with increased font size for better readability
plt.title('Ratio of Incident Frequency to Cash Requests', pad=20, fontsize=16)  # Title of the heatmap
plt.xlabel('Months Since the User\'s First Cash Request', labelpad=15, fontsize=14)  # X-axis label
plt.ylabel('Acquisition Month', labelpad=15, fontsize=14)  # Y-axis label

# Adjust layout to prevent clipping of titles and labels
plt.tight_layout()  # Ensure a clean layout for the plot

# Save the plot with a transparent background
plt.savefig('ratio_incident_to_cash_request_heatmap.png', transparent=True, dpi=300)  # Save the figure with a transparent background

plt.show()  # Display the heatmap


#%%# Postponed payment ratio with cash requests cohort analysis

# Step 1: Remove all rows that do not contain a postponement incident
fees_postpone_cohort_analysis = fees_alltypes_cohort_analysis[fees_alltypes_cohort_analysis['type'] == 'postpone']

# Step 2: Create the 'postpone_month' column indicating when the service was used
fees_postpone_cohort_analysis['postpone_month'] = fees_postpone_cohort_analysis['created_at'].dt.to_period('M')

# Step 3: Calculate the 'postpone_index' for each request (months since the cohort month)
fees_postpone_cohort_analysis['postpone_index'] = [
    (incident - cohort).n for incident, cohort in zip(
        fees_postpone_cohort_analysis['postpone_month'], 
        fees_postpone_cohort_analysis['cohort_month']
    )
]

# Step 4: Group by cohort month and postpone index, counting the number of requests
fees_postpone_cohort_analysis_table = fees_postpone_cohort_analysis.groupby(
    ['cohort_month', 'postpone_index']
).size().unstack(fill_value=0)  # Count occurrences and pivot the data for analysis

# Step 5: Calculate the ratio of postponed requests to cash requests
# Perform element-wise division and handle division by zero
ratio_postpone_to_cash_request = (
    fees_postpone_cohort_analysis_table
    .div(cohort_usage_frequency_table)
    .replace([np.inf, -np.inf], np.nan)  # Replace infinite values with NaN
    .fillna(0)  # Fill NaN values with 0
)

# Print the ratio table for verification
print(ratio_postpone_to_cash_request)

# Step 6: Plotting the ratio of instant payment requests as a heatmap
plt.figure(figsize=(10, 6))  # Set the figure size for the heatmap

# Create a heatmap to visualize the ratio of instant payment requests to cash requests
sns.heatmap(
    ratio_postpone_to_cash_request,
    annot=True,  # Annotate the cells with the ratio values
    cmap='Reds',  # Set the color palette for the heatmap
    cbar_kws={'label': 'Ratio of Instant Payment Requests', 'pad': 0.01},  # Add a color bar with a label and padding
    linewidths=0.5,  # Set the width of the lines separating the cells
    linecolor='white'  # Set the color of the lines separating the cells
)

# Adding titles and labels with increased font size for better readability
plt.title('Instant Payment Frequency', pad=20, fontsize=16)  # Title of the heatmap
plt.xlabel('Months Since the User\'s First Instant Payment Request', labelpad=15, fontsize=14)  # X-axis label
plt.ylabel('Acquisition Month', labelpad=15, fontsize=14)  # Y-axis label

# Adjust layout to prevent clipping of titles and labels
plt.tight_layout()  # Ensure a clean layout for the plot

# Save the plot with a transparent background
plt.savefig('ratio_postpone_to_cash_request_heatmap.png', transparent=True, dpi=300)  # Save the figure with a transparent background

plt.show()  # Display the heatmap


#%%# Instant payment ratio with cash requests cohort analysis

# Step 1: Include all rows that do not contain an instant payment incident
fees_instant_payment_cohort_analysis = fees_alltypes_cohort_analysis[fees_alltypes_cohort_analysis['type'] == 'instant_payment']

# Step 2: Create the 'instant_payment_month' column indicating when the service was used
fees_instant_payment_cohort_analysis['instant_payment_month'] = fees_instant_payment_cohort_analysis['created_at'].dt.to_period('M')

# Step 3: Calculate the 'instant_payment_index' for each request (months since the cohort month)
fees_instant_payment_cohort_analysis['instant_payment_index'] = [
    (incident - cohort).n for incident, cohort in zip(
        fees_instant_payment_cohort_analysis['instant_payment_month'], 
        fees_instant_payment_cohort_analysis['cohort_month']
    )
]

# Step 4: Group by cohort month and instant payment index, counting the number of requests
fees_instant_payment_cohort_analysis_table = fees_instant_payment_cohort_analysis.groupby(
    ['cohort_month', 'instant_payment_index']
).size().unstack(fill_value=0)  # Count occurrences and pivot the data for analysis

# Step 5: Calculate the ratio of instant payment requests to cash requests
# Perform element-wise division and handle division by zero
ratio_instant_payment_to_cash_request = (
    fees_instant_payment_cohort_analysis_table
    .div(cohort_usage_frequency_table)
    .replace([np.inf, -np.inf], np.nan)  # Replace infinite values with NaN
    .fillna(0)  # Fill NaN values with 0
)

# Print the ratio table for verification
print(ratio_instant_payment_to_cash_request)

# Step 6: Plotting the ratio of instant payment requests as a heatmap
plt.figure(figsize=(10, 6))  # Set the figure size for the heatmap

# Create a heatmap to visualize the ratio of instant payment requests to cash requests
sns.heatmap(
    ratio_instant_payment_to_cash_request,
    annot=True,  # Annotate the cells with the ratio values
    cmap='Reds',  # Set the color palette for the heatmap
    cbar_kws={'label': 'Ratio of Instant Payment Requests', 'pad': 0.01},  # Add a color bar with a label and padding
    linewidths=0.5,  # Set the width of the lines separating the cells
    linecolor='white'  # Set the color of the lines separating the cells
)

# Adding titles and labels with increased font size for better readability
plt.title('Instant Payment Frequency', pad=20, fontsize=16)  # Title of the heatmap
plt.xlabel('Months Since the User\'s First Instant Payment Request', labelpad=15, fontsize=14)  # X-axis label
plt.ylabel('Acquisition Month', labelpad=15, fontsize=14)  # Y-axis label

# Adjust layout to prevent clipping of titles and labels
plt.tight_layout()  # Ensure a clean layout for the plot

# Save the plot with a transparent background
plt.savefig('instant_payment_frequency_heatmap.png', transparent=True, dpi=300)  # Save the figure with a transparent background

plt.show()  # Display the heatmap



#%%# Cohort revenue generated analysis (debt still unpaid)

# Step 1: Filter out unpaid fees into a separate DataFrame
unpaid_fees = fees[fees['paid_at'].isna()]  # Save the rows where 'paid_at' is NaT
fees_cleaned = fees.dropna(subset=['paid_at'])  # Drop rows where 'paid_at' is NaT
print(fees_cleaned['paid_at'].isna().sum())  # Check that no NaT values remain (should return 0)

# Step 2: Perform an inner merge to combine relevant data from fees and cash_request
revenue_analysis = pd.merge(
    fees_cleaned[['cash_request_id', 'paid_at', 'total_amount']],  # Select relevant columns from fees
    cash_request[['id', 'cohort_month']],  # Select relevant columns from cash_request
    left_on='cash_request_id',  # Merge based on 'cash_request_id'
    right_on='id',  # Merge based on 'id' from cash_request
    how='inner'  # Perform an inner join to retain only matching records
)

# Step 3: Create the 'payment_month' column to indicate when the payment was made
revenue_analysis['payment_month'] = revenue_analysis['paid_at'].dt.to_period('M')

# Step 4: Calculate the 'payment_index' for each payment (months since the cohort month)
revenue_analysis['payment_index'] = [
    (payment - cohort).n for payment, cohort in zip(
        revenue_analysis['payment_month'], 
        revenue_analysis['cohort_month']
    )
]

# Step 5: Group by cohort month and payment index, and sum the total revenue
revenue_analysis_table = revenue_analysis.groupby(
    ['cohort_month', 'payment_index']
)['total_amount'].sum().unstack(fill_value=0)  # Pivot the data for easier analysis

# Display the revenue analysis table to observe total revenue by cohort and payment index
print(revenue_analysis_table)  # Print the revenue analysis table
print(revenue_analysis_table.sum().sum())  # Print the total revenue across all cohorts

# Convert the revenue values in the table to integers for plotting and annotations
revenue_analysis_table_int = revenue_analysis_table.astype(int)

# Step 6: Plotting the cohort revenue as a heatmap
plt.figure(figsize=(10, 6))  # Set the figure size for the heatmap

# Create a heatmap with integer annotations for revenue
sns.heatmap(
    revenue_analysis_table_int,  # Data for the heatmap
    annot=revenue_analysis_table_int,  # Annotations for the heatmap
    fmt="d",  # Format the annotations as integers
    cmap='Greens',  # Set the color palette for the heatmap
    cbar_kws={'label': 'Revenue', 'pad': 0.01},  # Add a color bar with a label and padding
    linewidths=0.5,  # Set the width of the lines separating the cells
    linecolor='white'  # Set the color of the lines separating the cells
)

# Adding titles and labels with increased font size for better readability
plt.title('Revenue Generated by Each Cohort', pad=20, fontsize=16)  # Title of the heatmap
plt.xlabel('Months of Payment', labelpad=15, fontsize=14)  # X-axis label
plt.ylabel('Acquisition Month', labelpad=15, fontsize=14)  # Y-axis label

# Adjust layout to prevent clipping of titles and labels
plt.tight_layout()  # Ensure a clean layout for the plot

# Save the plot with a transparent background
plt.savefig('cohort_revenue_heatmap.png', transparent=True, dpi=300)  # Save the figure with a transparent background

plt.show()  # Display the heatmap



#%%# How much has each cohort owes

# Step 1: Filter the fees DataFrame to keep rows where 'paid_at' is NaT (i.e., not paid yet)
unpaid_fees = fees[fees['paid_at'].isna()]

# Step 2: Perform an inner merge to add 'cohort_month' to the unpaid fees
revenue_analysis_with_debt = pd.merge(
    cash_request[['id', 'cohort_month']],  # Select only the 'cohort_month' from cash_request
    unpaid_fees[['cash_request_id', 'total_amount']],  # Select unpaid rows from fees
    left_on='id',  # Column in cash_request to merge on
    right_on='cash_request_id',  # Column in unpaid_fees to merge on
    how='inner'  # Perform an inner merge to retain only matching records
)

# Step 3: Group by 'cohort_month' and sum the 'total_amount' for each cohort
debt_per_cohort = revenue_analysis_with_debt.groupby('cohort_month')['total_amount'].sum().reset_index()

# Convert 'cohort_month' to a string format for plotting
debt_per_cohort['cohort_month'] = debt_per_cohort['cohort_month'].astype(str)

# Step 4: Plotting the bar graph to visualize total amount owed by cohort over time
plt.figure(figsize=(10, 6))  # Set the figure size for the plot

# Plotting the data as a bar graph
plt.bar(debt_per_cohort['cohort_month'], debt_per_cohort['total_amount'], color=color)  # Use the specified color for bars

# Adding titles and labels with increased font size for better readability
plt.title('Total Amount Owed by Cohort Over Time', pad=20, fontsize=16)  # Title of the plot
plt.xlabel('Cohort Month', labelpad=15, fontsize=14)  # X-axis label
plt.ylabel('Amount Owed (€)', labelpad=15, fontsize=14)  # Y-axis label

# Rotating the x-axis labels for better readability
plt.xticks(rotation=45)

# Adding grid for better visualization of the data
plt.grid(axis='y')

# Remove the top and right spines (box)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)


# Save the plot with a transparent background
plt.savefig('total_amount_owed_heatmap.png', transparent=True, dpi=300)  # Save the figure with a transparent background

# Show the plot with adjusted layout
plt.tight_layout()  # Adjust layout to make room for titles and labels
plt.show()  # Display the bar plot


#%%# Loan duration analysis

# Perform an inner merge
loan_return_duration_cohort_analysis = pd.merge(
    cash_request[['id', 'cohort_month', 'created_at', 'money_back_date']],  # Select columns from cash_request
    fees[['cash_request_id', 'status', 'paid_at']],  # Select columns from fees
    left_on='id',  # Column in cash_request
    right_on='cash_request_id',  # Column in fees
    how='inner'  # Type of merge
)

# Calculate the duration in days 
loan_return_duration_cohort_analysis['loan_duration_days'] = (loan_return_duration_cohort_analysis['money_back_date'] - loan_return_duration_cohort_analysis['created_at']).dt.days

# Summary statistics
duration_summary = loan_return_duration_cohort_analysis['loan_duration_days'].describe()
print(duration_summary)


# Plot the distribution of cash advance durations
plt.figure(figsize=(10, 6))
sns.histplot(loan_return_duration_cohort_analysis['loan_duration_days'], bins=30, kde=True, color=color)
plt.title('Distribution of Loan Durations')
plt.xlabel('Loan Duration (Days)')
plt.ylabel('Frequency of Loans')

# Remove the top and right spines (box)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Save the plot with a transparent background and higher resolution
plt.savefig('distribution_of_loan_durations.png', transparent=True, dpi=300)

plt.show()  # Display the histogram



#%%# New Relevant Metric - Churn Rate

# Step 1: Calculate total requests per user
churn_rate = cash_request.groupby('user_id').agg(
    total_requests=('id', 'count'),
    cohort_month=('cohort_month', 'first')
).reset_index()

# Step 2: Define churn status
churn_rate['churned'] = churn_rate['total_requests'] == 2  # True if churned, False otherwise

# Step 3: Calculate churn rate for each cohort month and cohort index
churn_rate_cohort_analysis = cash_request.groupby(['cohort_month', 'cohort_index']).agg(
    total_users=('user_id', 'nunique'),  # Total unique users
    churned_users=('user_id', lambda x: (churn_rate.loc[churn_rate['user_id'].isin(x), 'churned']).sum())  # Churned users
).reset_index()

churn_rate_cohort_analysis['churn_rate'] = (churn_rate_cohort_analysis['churned_users'] / churn_rate_cohort_analysis['total_users']) * 100
churn_rate_cohort_analysis['churn_rate'] = churn_rate_cohort_analysis['churn_rate'].round().astype(int)

# Step 5: Create a pivot table for the heatmap
churn_rate_cohort_analysis_table = churn_rate_cohort_analysis.pivot(index='cohort_month', columns='cohort_index', values='churn_rate')

# Convert all NaN values to 0
churn_rate_cohort_analysis_table.fillna(0, inplace=True)
churn_rate_cohort_analysis['churn_rate'] = churn_rate_cohort_analysis['churn_rate'].round().astype(int)
churn_rate_cohort_analysis_table = churn_rate_cohort_analysis_table.astype(int)

print(churn_rate_cohort_analysis_table.dtypes)


# Step 6: Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(
    churn_rate_cohort_analysis_table, 
    annot=True, 
    fmt='.1f', 
    cmap='Purples', 
    cbar_kws={'label': 'Churn Rate (%)'}
)

plt.title('Cohort Churn Rate (%)')
plt.xlabel('Months since the Cohort\'s First Cash Request')
plt.ylabel('Cohort Month')

# Save the plot with a transparent background
plt.savefig('cohort_churn_rate_heatmap.png', transparent=True, dpi=300)  # Save the figure with a transparent background

plt.tight_layout()
plt.show()  # Display the heatmap
