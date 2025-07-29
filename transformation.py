import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from scipy.stats import boxcox



'''
transformation.py contains functions that used to clean the data, calculating RFM features from
the transactional dataset, stadardizing RFM features, and handling the absent customers in current
given a custome set.
'''



def cleaning(clean_df):
    """
    Standard cleaning method used at the beginning of the data pipeline.

    Parameters
    ----------
    clean_df : DataFrame
        The DataFrame to be cleaned.

    Returns
    -------
    DataFrame
        The cleaned DataFrame with the following transformations applied:
        - Set the index to the 'Date' column and convert it to datetime.
        - Remove rows with 'All Customers' aggregation.
        - Remove data for September 2022.
    """
    # Set the index to the 'Date' column and convert it to datetime
    clean_df.set_index('Date', inplace=True)
    clean_df.index = pd.to_datetime(clean_df.index)

    # Eliminate 'All Customers' 
    all_cust_ids = clean_df[clean_df['Customer Group'] == 'All Customers']['Customer ID'].unique()
    clean_df = clean_df[~clean_df['Customer ID'].isin(all_cust_ids)]

    clean_df = clean_df[clean_df.index < '2022-09-01']

    return clean_df



def get_first_and_last_day(year, month):
    first_day = datetime(year, month, 1)
    if month == 12:
        last_day = datetime(year + 1, 1, 1) - timedelta(days=1)
    else:
        last_day = datetime(year, month + 1, 1) - timedelta(days=1)
    return first_day, last_day


'''
The following function is used to get the Monday (specific date) of the week where the given date is in
'''
def get_monday(date):
    return date - timedelta(days=date.weekday())

'''
The following function is used to get the Suanday (specific date) of the week where the given date is in
'''
def get_sunday(date):
    return date + timedelta(days=(6 - date.weekday()))


def calculate_monthly_rfm(transaction_data, customer_list, start_date, end_date):
    """
    Iterates from start_date to end_date to calculate monthly RFM features.

    Parameters
    ----------
    transaction_data : DataFrame
        DataFrame containing transaction data.
    customer_list : list
        List of customer IDs.
    start_date : datetime
        The start date for the iteration in 'YYYY-MM-DD' format.
    end_date : datetime
        The end date for the iteration in 'YYYY-MM-DD' format.

    Returns
    -------
    list
        A list containing the RFM features for each month within the date range.
    """
    rfm_data = []
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')

    current_date = start_date
    while current_date <= end_date:
        first_day_of_month = current_date.replace(day=1)
        last_day_of_month = (current_date.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        
        first_monday = get_monday(first_day_of_month)
        last_sunday = get_sunday(last_day_of_month)
        print(first_monday)
        print(last_sunday)
        
        monthly_rfm = rfm_in_weeks_calculation(transaction_data, first_monday, last_sunday)
        current_date = (current_date.replace(day=28) + timedelta(days=4)).replace(day=1)
        rfm_data.append(monthly_rfm)
    return rfm_data


         
def rfm_calculation(transaction_data, start_date, end_date, customer_list):
    """
    Calculate Recency, Frequency, and Monetary (RFM) metrics for a list of customers 
    within a specified date range. For detailed definition, please see the paper.

    Parameters
    ----------
    transaction_data : DataFrame
        A DataFrame containing at least the following columns:
        - 'Customer ID'
        - 'date' (datetime or convertible to datetime)
        - 'Revenue' (numeric)
    start_date : str or datetime
        The start date of the analysis period (inclusive).
    end_date : str or datetime
        The end date of the analysis period (inclusive).
    customer_list : list
        A list of all sampled customer IDs.

    Returns
    -------
    DataFrame
        An RFM DataFrame indexed by 'Customer ID', containing columns (For detailed definition, please see the paper):
        - 'recency'
        - 'frequency'
        - 'monetary_value'
        - 'As of': The end date of the analysis period.
        Any customers in `customer_list` without transactions in the period 
        are assigned zeros for R, F, M.
    """

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    filtered_transactions = transaction_data[(transaction_data['date'] >= start_date) & (transaction_data['date'] <= end_date)].copy()
    
    recency = filtered_transactions.groupby('Customer ID').agg(
        t_n=('date', 'max')
    ).reset_index()
    recency['recency'] = (recency['t_n'] - start_date) / (end_date - start_date)
    
    purchase_days = filtered_transactions.groupby(['Customer ID', 'date']).size().reset_index(name='purchase_count')
    frequency = purchase_days.groupby('Customer ID').size().reset_index(name='purchase_days')
    
    frequency['frequency'] = frequency['purchase_days'] / (end_date - start_date).days
    
    monetary = filtered_transactions.groupby('Customer ID')['Revenue'].sum().reset_index(name='total_revenue')
    monetary['monetary_value'] = monetary['total_revenue'] / (end_date - start_date).days

    resulting_rfm = pd.merge(recency[['Customer ID', 'recency']], frequency[['Customer ID', 'frequency']], on='Customer ID')
    resulting_rfm = pd.merge(resulting_rfm, monetary[['Customer ID', 'monetary_value']], on='Customer ID')
    customer_df = pd.DataFrame(customer_list, columns=['Customer ID'])
    resulting_rfm = pd.merge(customer_df, resulting_rfm, on='Customer ID', how='left')

    # Fill missing values for customers with no purchases during the period
    resulting_rfm['recency'].fillna(0, inplace=True)
    resulting_rfm['frequency'].fillna(0, inplace=True)
    resulting_rfm['monetary_value'].fillna(0, inplace=True)  
    resulting_rfm['As of'] = end_date

    resulting_rfm.set_index('Customer ID', inplace=True)
    
    return resulting_rfm


def rfm_calculation_CLV(transaction_data, start_date, end_date, customer_list):
    
    """
    Calculate RFM metrics for the Customer Lifetime Value (CLV) modeling. This function is for calculating 
    transformed RFM metrics from the RFM definitions. 

    This function computes extended RFM metrics commonly used for CLV analysis:
      - recency: The number of days between the first and the last purchase.
      - frequency: The number of distinct purchase days in the analysis period.
      - monetary_value: The average revenue per purchase (total revenue / frequency).
      - T: The number of days from the first purchase to the end of the analysis period.
      - As of: The end date of the analysis period (timestamp).

    Parameters
    ----------
    transaction_data : DataFrame
        A DataFrame of transactions, which must include:
        - 'Customer ID' (identifier for each customer)
        - 'date' (datetime or convertible to datetime)
        - 'Revenue' (numeric, indicating revenue from each transaction)
    start_date : str or datetime
        Start date of the analysis period (inclusive).
    end_date : str or datetime
        End date of the analysis period (inclusive).
    customer_list : list
        A list of all sampled customer IDs.

    Returns
    -------
    DataFrame
        A DataFrame indexed by 'Customer ID' with columns (For detailed definition, please see the CLV section of the paper):
        - 'recency'
        - 'frequency'
        - 'monetary_value'
        - 'T'
    """
     
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    filtered_transactions = transaction_data[(transaction_data['date'] >= start_date) & (transaction_data['date'] <= end_date)].copy()
    
    recency = filtered_transactions.groupby('Customer ID').agg(
        t_n=('date', 'max'),
        t_min = ('date', 'min')
    ).reset_index()
    
    recency['recency'] = (recency['t_n'] - recency['t_min']).dt.days
    recency['T'] = (end_date - recency['t_min']).dt.days
    
    purchase_days = filtered_transactions.groupby(['Customer ID', 'date']).size().reset_index(name='purchase_count')
    frequency = purchase_days.groupby('Customer ID').size().reset_index(name='purchase_days')
    frequency['frequency'] = frequency['purchase_days']
    
    monetary = filtered_transactions.groupby('Customer ID')['Revenue'].sum().reset_index(name='total_revenue')
    monetary = pd.merge(frequency, monetary, on='Customer ID', how='left')  
    monetary['monetary_value'] = monetary['total_revenue'] / frequency['frequency']

    resulting_rfm = pd.merge(recency[['Customer ID', 'recency', 'T']], frequency[['Customer ID', 'frequency']], on='Customer ID')
    resulting_rfm = pd.merge(resulting_rfm, monetary[['Customer ID', 'monetary_value']], on='Customer ID')
    customer_df = pd.DataFrame(customer_list, columns=['Customer ID'])
    resulting_rfm = pd.merge(customer_df, resulting_rfm, on='Customer ID', how='left')

    resulting_rfm['recency'].fillna(0, inplace=True)
    resulting_rfm['frequency'].fillna(0, inplace=True)
    resulting_rfm['monetary_value'].fillna(0, inplace=True)
    resulting_rfm['T'].fillna(0, inplace=True)
    resulting_rfm['As of'] = end_date

    resulting_rfm.set_index('Customer ID', inplace=True)
    
    return resulting_rfm


def rfm_in_weeks_calculation_evaluation(transaction_data, start, end):
    '''
    the start as the start date for the rfm calculation
    return a dataframe contains frequency, recency, and T(age from the end date), and the average monetary value 
    '''
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
    
    filtered_transactions = transaction_data[(transaction_data['date'] >= start_date) & (transaction_data['date'] <= end_date)]
    
    filtered_transactions['week'] = ((filtered_transactions['date'] - start_date) / np.timedelta64(1, 'W')).astype(int)
    
    resulting_rfm = pd.DataFrame()

    
        
    recency_and_T = filtered_transactions.groupby(['Customer ID']).agg(
        recency=('date', lambda x: ((x.max() - start_date).days)/7),
        T=('date', lambda x: ((end_date - x.min()).days)/7),
    ).reset_index()

    frequency_and_revenue_by_weeks = filtered_transactions.groupby(['Customer ID', 'week'])['Revenue'].sum()
    frequency = (frequency_and_revenue_by_weeks.groupby('Customer ID').size()).reset_index(name='frequency')
    monetary = frequency_and_revenue_by_weeks.groupby('Customer ID').mean().reset_index(name='monetary_value')

        
    resulting_rfm = pd.merge(recency_and_T, frequency, on='Customer ID')
    resulting_rfm = pd.merge(resulting_rfm, monetary, on='Customer ID')
    resulting_rfm = resulting_rfm.set_index('Customer ID')
    
    resulting_rfm['As of']  = end_date

         
    return resulting_rfm




def scaling(purchases_sorted_rfm):
    """
    Scale the 'Revenue', 'Recency', and 'Frequency' columns of each DataFrame in the list using MinMax scaling.

    Parameters
    ----------
    purchases_sorted_rfm : list of pd.DataFrame
        List of DataFrames containing RFM (Revenue, Recency, Frequency) data for different months.
        Each DataFrame should have columns: 'monetary_value', 'recency', 'frequency'.

    Returns
    -------
    list of pd.DataFrame
        List of DataFrames with scaled 'Revenue', 'Recency', and 'Frequency' columns.
    """
    scaler = MinMaxScaler()

    for month in purchases_sorted_rfm:
        month[['Revenue', 'Recency', 'Frequency']] = scaler.fit_transform(month[['monetary_value', 'recency', 'frequency']])

    purchases_rfm_stand = purchases_sorted_rfm.copy()

    return purchases_rfm_stand



def scaling_revenue(purchases_sorted_rfm_list):
    """
    Scale the revenue column of the calculated RFM dataframe.
    """

    scaler = MinMaxScaler()

    for purchases_sorted_rfm in purchases_sorted_rfm_list:
        purchases_sorted_rfm[['revenue']] = scaler.fit_transform(purchases_sorted_rfm[['monetary_value']])

    purchases_rfm_stand = purchases_sorted_rfm_list.copy()

    return purchases_rfm_stand




def box_cox_transformation(Lambda, RFM, Dimension):
    '''
    This function takes the lambda value for the Box-Cox transformation
    and the dimension in which the transformation applies. Note that the Box-Cox transformation requires
    the data to be positive. For those absent customers who has been assigned with [0,0,0] for their 
    RFM features, we use 0.1 to replace the zeros
    '''
    transformed = []

    for month in range(len(RFM)):
        current_rfm = RFM[month].copy()
        if Dimension == 'monetary_value':
            current_rfm['monetary_value'] = boxcox(current_rfm['monetary_value'] + 0.1, Lambda)
        if Dimension == 'recency':
            current_rfm['recency'] = boxcox(current_rfm['recency'] + 0.1, Lambda)
        if Dimension == 'frequency':
            current_rfm['frequency'] = boxcox(current_rfm['frequency'] +0.1, Lambda)
        transformed.append(current_rfm)

    return transformed





def cleaning(clean_df):
    # This is the standard cleaning method used at the beginning of the data pipeline.

    # Below we read the .csv file specified through the filename parameter.

    # From there we set the index to the date column.
    clean_df.set_index('Date', inplace=True)
    # We convert the index into a datatime value type
    clean_df.index = pd.to_datetime(clean_df.index)

    # Elimination of 'All Customers' aggregation
    all_cust_ids = clean_df[clean_df['Customer Group'] == 'All Customers']['Customer ID'].unique()

    clean_df = clean_df[~clean_df['Customer ID'].isin(all_cust_ids)]

    # Elimination of September
    clean_df = clean_df[clean_df.index < '2022-09-01']


    return clean_df
