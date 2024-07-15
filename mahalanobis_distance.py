import pandas as pd
import numpy as np
from scipy.spatial import distance
from scipy.stats import norm
from datetime import datetime, timedelta

def mahalanobis_distance(x, mean_vector, inv_cov_matrix):
    return (distance.mahalanobis(x, mean_vector, inv_cov_matrix))


def calculate_mahalanobis_monthly(rfm_dataframe):

    '''
    Iterating through each month, calculating the the Mahalanobis distance of each customer's RFM movement 
    to the mean RFM movement among all customers
    '''
    result_movement_rfm = []
    for month in range(0, len(rfm_dataframe)-1): 
        current_rfm = rfm_dataframe[month][['recency', 'frequency', 'monetary_value']]
        next_rfm = rfm_dataframe[month+1][['recency', 'frequency', 'monetary_value']]
        monthly_change = (next_rfm - current_rfm)
        monthly_change = monthly_change.dropna()
        covariance = monthly_change.cov()
        inv_cov = np.linalg.inv(covariance)
        mean = np.asarray(monthly_change.mean())
        monthly_change['Mahalanobis_Distance'] = monthly_change.apply(lambda x: mahalanobis_distance(x, mean, inv_cov), axis=1)
        result_movement_rfm.append(monthly_change)
    return result_movement_rfm


def flag_outlier_customers(rfm_movement_transformed, timestamps, flag_start, critical_value=7.81):
    """
    Flag outlier customers based on their Mahalanobis Distance of monthly change in RFM
    to the mean change. The flagging process begins from the month of the flag_start
    parameter. 

    Parameters:
    - rfm_movement_transformed (list of DataFrame): List of DataFrames with transformed Mahalanobis distances.
    - timestamps (DatetimeIndex): Timestamps corresponding to each month in the data.
    - flag_start(string): the string 
    - critical_value (float): Critical value for Mahalanobis Distance to flag outliers. Default is 7.81.

    Returns:
    - md_customers_flagged (list of list): List of lists containing indices of outlier customers for each month after June 2022.
    """
    md_customers_flagged = []

    for time, current_month_rfm in zip(timestamps, rfm_movement_transformed):
        outlier_customers = current_month_rfm[(current_month_rfm['Mahalanobis_Distance']**2 > critical_value) 
                                              & (current_month_rfm['frequency'] < 0)
                                              & (current_month_rfm['monetary_value'] < 0)].index

        if time >= pd.Timestamp(flag_start):
            md_customers_flagged.append(outlier_customers.to_list())
            print(f"{time.strftime('%Y-%m')}: {outlier_customers.size} outliers flagged")

    return md_customers_flagged