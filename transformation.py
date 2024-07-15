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

The following function is used to 
'''
def get_monday(date):
    return date - timedelta(days=date.weekday())

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

    rfm_data = absence_handler_unstand(rfm_data, customer_list)
    return rfm_data



def rfm_in_weeks_calculation(transaction_data, start, end):
    '''
    the start as the start date for the rfm calculation
    return a dataframe contains frequency, recency, and T(age from the end date), and the average monetary value
    this function uses week as the unit of time
    
    '''
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
    
    filtered_transactions = transaction_data[(transaction_data['date'] >= start_date) & (transaction_data['date'] <= end_date)]
    
    filtered_transactions['week'] = ((filtered_transactions['date'] - start_date) / np.timedelta64(1, 'W')).astype(int)
    
    resulting_rfm = pd.DataFrame()

    
        
    recency_and_T = filtered_transactions.groupby(['Customer ID']).agg(
        recency=('date', lambda x: ((x.max() - x.min()).days)/7),
        T=('date', lambda x: ((end_date - x.min()).days)/7)
    ).reset_index()

    frequency_and_revenue_by_weeks = filtered_transactions.groupby(['Customer ID', 'week'])['Revenue'].sum()
    frequency = (frequency_and_revenue_by_weeks.groupby('Customer ID').size() -1).reset_index(name='frequency')
    monetary = frequency_and_revenue_by_weeks.groupby('Customer ID').sum().reset_index(name='monetary_value')

        
    resulting_rfm = pd.merge(recency_and_T, frequency, on='Customer ID')
    resulting_rfm = pd.merge(resulting_rfm, monetary, on='Customer ID')
    resulting_rfm = resulting_rfm.set_index('Customer ID')
    resulting_rfm.loc[resulting_rfm['frequency'] == 0, 'recency'] = 0
    resulting_rfm['As of']  = end

         
    return resulting_rfm


def rfm_in_weeks_calculation_CLV(transaction_data, start, end):
    '''
    the start as the start date for the rfm calculation
    return a dataframe contains frequency, recency, and T(age from the end date), and the average monetary value
    this function uses week as the unit of time
    
    '''
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
    
    filtered_transactions = transaction_data[(transaction_data['date'] >= start_date) & (transaction_data['date'] <= end_date)]
    
    filtered_transactions['week'] = ((filtered_transactions['date'] - start_date) / np.timedelta64(1, 'W')).astype(int)
    
    resulting_rfm = pd.DataFrame()

    
        
    recency_and_T = filtered_transactions.groupby(['Customer ID']).agg(
        recency=('date', lambda x: ((x.max() - x.min()).days)/7),
        T=('date', lambda x: ((end_date - x.min()).days)/7)
    ).reset_index()

    frequency_and_revenue_by_weeks = filtered_transactions.groupby(['Customer ID', 'week'])['Revenue'].sum()
    frequency = (frequency_and_revenue_by_weeks.groupby('Customer ID').size() -1).reset_index(name='frequency')
    monetary = frequency_and_revenue_by_weeks.groupby('Customer ID').mean().reset_index(name='monetary_value')

        
    resulting_rfm = pd.merge(recency_and_T, frequency, on='Customer ID')
    resulting_rfm = pd.merge(resulting_rfm, monetary, on='Customer ID')
    resulting_rfm = resulting_rfm.set_index('Customer ID')
    resulting_rfm.loc[resulting_rfm['frequency'] == 0, 'recency'] = 0
    resulting_rfm['As of']  = end

         
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
    
    resulting_rfm['As of']  = end

         
    return resulting_rfm



def rfm_calculation(purchases_sorted):
    # the array below will contain the RFM calculations for each month
    purchases_sorted_rfm = []

    # iterating month-by-month...
    for month in purchases_sorted:

        # sum every customer's revenue
        monthly_revenue = month.groupby(['Customer ID'])['Revenue'].sum()

        # add a new column, where the value is derived from subtracting the last day in the month by the index value
        # e.g. if a purchase was made on October 21, this new column would be 31 - 21 = 10
        month["Recency"] = (month.index[-1] - month.index).days
        # sorting by customer ID, find the last recency value for a customer
        monthly_recency = month.groupby(['Customer ID'])['Recency'].last()

        # count the number of purchases a customer made in a month
        monthly_frequency = month.groupby(['Customer ID'])['Revenue'].count()

        # By taking the three series we extracted above, we can convert these into a Pandas dataframe
        # Important note: while the RFM framework has recency at the beginning, in our dataframe revenue (or monetary
        # value) is at the beginning. The order here is important.
        monthly_df = pd.DataFrame([monthly_revenue, monthly_recency, monthly_frequency])
        # We need to transpose this dataframe since each customer is a column. This will make every customer is a row.
        monthly_df = monthly_df.transpose()
        monthly_df = monthly_df.set_axis(['Revenue', 'Recency', 'Frequency'], axis=1)
        # As of date is for our ARIMA work
        monthly_df['As of'] = month.index[-1]
        # Last, we append the months RFM values to the associated array
        purchases_sorted_rfm.append(monthly_df)

    return purchases_sorted_rfm



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



def absence_handler_unstand(purchases_rfm, customer_list):

    '''
    Handling the customers in the sample customers set who had made purhcase before but
    had not make any purchase in the current period
    '''
    running_list = set()
    absence_list = []
    rfm_stand_complete = []

    customer_set = set(customer_list)

    # Iterating month-by-month...
    for month in purchases_rfm:
        # Get the list of customers in a given month and convert them into a set.
        monthly_set = set(month.index.unique())

        # Update the running_list set with these IDs.
        running_list.update(month.index.unique())
        

        
        
        customer_set_all = customer_set.union((running_list))
        # Compute the set difference, considering only customers in customer_list
        absent_customers = customer_set_all - monthly_set
        
        absence_list.append(absent_customers)

    # Fill in the values for absent customers.
    for x, month in enumerate(purchases_rfm):
        # For every absence cusomers:
        for absence in absence_list[x]:
            # Calculate recency in weeks (assuming 'As of' column is datetime)
            last_day_of_month = month['As of'].iloc[-1]
            first_day_of_month = month['As of'].iloc[0].replace(day=1)
            
            recency_weeks = (last_day_of_month - first_day_of_month).days / 7
            T = recency_weeks
            month.loc[absence] = [0, T, 0, 0, last_day_of_month]

        
        rfm_stand_complete.append(month.sort_index())

    return rfm_stand_complete


def absence_handler(purchases_rfm_stand):
    # One of the last steps of our preprocessing is filling in absent customers. By tracking when customers start making
    # purchases, we can determine if a customer is absent or not.
    # The running list below will contain a set of all customer IDs who have made a purchase to date. By comparing the
    # customers in a month to the running list, we can find the customers who are absent.
    running_list = set()
    # The absence list will contain an array of arrays, which will track the absent customers for each month of the
    # data set.
    absence_list = []
    # The array below will contain the RFM values for each data set, with the absent customers 'filled in' with a
    # revenue of 0, recency of 1, and frequency of 0.
    rfm_stand_complete = []

    # iterating month-by-month...
    for month in purchases_rfm_stand:
        # we get the list of customers in a given month and convert them into a set.
        monthly_set = set(month.index.unique())

        # we then update the running_list set with these IDs.
        running_list.update(month.index.unique())

        # we then compute the set difference, appending it to the absence list for that month.
        absence_list.append(running_list - monthly_set)

    # here we fill in the values for absent customers.
    for x in range(len(purchases_rfm_stand)):
        # we first fetch the RFM values of the associated month
        month = purchases_rfm_stand[x]

        # for every absence...
        for absence in absence_list[x]:
            # fill in a row with these values
            month.loc[absence] = [0, 1, 0, month['As of'].values[-1]]

        # we then append the completed RFM values to the associated array
        rfm_stand_complete.append(month)

    return rfm_stand_complete, absence_list



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
