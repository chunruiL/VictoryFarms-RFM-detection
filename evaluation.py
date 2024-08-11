import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import ttest_ind
from . import transformation

def frequency_evaluation(customer_id_list, transaction_data, detection_time):
    """
    Evaluate the change in frequency of customer transactions before and after a detection time.

    Parameters
    ----------
    customer_id_list : list
        List of customer IDs to evaluate.
    detection_time : str
        The detection time in 'YYYY-MM-DD' format.
    transaction_data : DataFrame
        DataFrame containing transaction data with columns such as 'Customer ID', 'date', and 'Revenue'.


    Returns
    -------
    float
        The mean ratio of transaction frequency after detection to before detection.
    """
    customer_transactions = transaction_data[transaction_data['Customer ID'].isin(customer_id_list)]
    rfm_before_detection = transformation.rfm_in_weeks_calculation_evaluation(customer_transactions,
                                                                      '2022-01-01', detection_time)

    after_detection_date = datetime.strptime(detection_time, "%Y-%m-%d") + timedelta(days=1)
    after_detection_date = after_detection_date.strftime("%Y-%m-%d")
    rfm_after_detection = transformation.rfm_in_weeks_calculation_evaluation(customer_transactions, after_detection_date, '2022-08-31')

    missing_customers_after = rfm_before_detection.index.difference(rfm_after_detection.index)
    df_missing_after = pd.DataFrame(index=missing_customers_after, columns=rfm_after_detection.columns).fillna(0)
    rfm_after_detection = pd.concat([rfm_after_detection, df_missing_after])
    rfm_before_detection = rfm_before_detection.drop(rfm_after_detection.index.difference(rfm_before_detection.index))
    
    ratios = rfm_after_detection['frequency'] / rfm_before_detection['frequency']
    mean_ratio = np.mean(rfm_after_detection['frequency'] / rfm_before_detection['frequency'])
    std =  np.std(rfm_after_detection['frequency']/rfm_before_detection['frequency'])
    
    return ratios, mean_ratio, std 


def monetary_value_evaluation(customer_id_list, transaction_data, detection_time):
    """
    Evaluate the change in monetary value of customer transactions before and after a detection time.

    Parameters
    ----------
    customer_id_list : list
        List of customer IDs to evaluate.
    detection_time : str
        The detection time in 'YYYY-MM-DD' format.
    transaction_data : DataFrame
        DataFrame containing transaction data for all custoemrs
    Returns
    -------
    tuple
        A tuple containing the mean and standard deviation of the ratio of monetary value after detection to before detection.
    """
    customer_transactions = transaction_data[transaction_data['Customer ID'].isin(customer_id_list)]
    rfm_before_detection = transformation.rfm_in_weeks_calculation_evaluation(customer_transactions, '2022-01-01', detection_time)
    rfm_before_detection = rfm_before_detection[rfm_before_detection['monetary_value'] > 0]

    after_detection_date = datetime.strptime(detection_time, "%Y-%m-%d") + timedelta(days=1)
    after_detection_date = after_detection_date.strftime("%Y-%m-%d")
    rfm_after_detection = transformation.rfm_in_weeks_calculation_evaluation(customer_transactions, after_detection_date, '2022-08-31')

    missing_customers_after = rfm_before_detection.index.difference(rfm_after_detection.index)
    df_missing_after = pd.DataFrame(index=missing_customers_after, columns=rfm_after_detection.columns).fillna(0)
    rfm_after_detection = pd.concat([rfm_after_detection, df_missing_after])
    rfm_before_detection = rfm_before_detection.drop(rfm_after_detection.index.difference(rfm_before_detection.index))
    
    ratios = rfm_after_detection['monetary_value'] / rfm_before_detection['monetary_value']
    mean_ratio = np.mean(rfm_after_detection['monetary_value'] / rfm_before_detection['monetary_value'])
    std_ratio = np.std(rfm_after_detection['monetary_value'] / rfm_before_detection['monetary_value'])

    return ratios, mean_ratio, std_ratio


def recency_evaluation(customer_id_list, transaction_data, detection_time):
    """
    Evaluate the change in recency of customer transactions before and after a detection time.

    Parameters
    ----------
    customer_id_list : list
        List of customer IDs to evaluate.
    transaction_data : DataFrame
        DataFrame containing transaction data for all custoemrs
    detection_time : str
        The detection time in 'YYYY-MM-DD' format.

    Returns
    -------
    tuple
        A tuple containing the mean and standard deviation of the ratio of recency after detection to before detection.
    """
    
    customer_transactions = transaction_data[transaction_data['Customer ID'].isin(customer_id_list)]
    rfm_before_detection = transformation.rfm_in_weeks_calculation_evaluation(transaction_data, '2022-01-01', detection_time)
    rfm_before_detection = rfm_before_detection[rfm_before_detection['recency'] > 0]

    after_detection_date = datetime.strptime(detection_time, "%Y-%m-%d") + timedelta(days=1)
    after_detection_date = after_detection_date.strftime("%Y-%m-%d")
    rfm_after_detection = transformation.rfm_in_weeks_calculation_evaluation(customer_transactions, after_detection_date, '2022-08-31')

    missing_customers_after = rfm_before_detection.index.difference(rfm_after_detection.index)
    df_missing_after = pd.DataFrame(index=missing_customers_after, columns=rfm_after_detection.columns).fillna(0)
    rfm_after_detection = pd.concat([rfm_after_detection, df_missing_after])
    rfm_before_detection = rfm_before_detection.drop(rfm_after_detection.index.difference(rfm_before_detection.index))

    ratios = rfm_after_detection['recency'] / rfm_before_detection['recency']
    mean_ratio = np.mean(rfm_after_detection['recency'] / rfm_before_detection['recency'])
    std_ratio = np.std(rfm_after_detection['recency'] / rfm_before_detection['recency'])

    return ratios, mean_ratio, std_ratio


def perform_t_tests(monthly_data1, monthly_data2, months):
    """
    Perform t-tests for each pair of monthly data.

    Parameters
    ----------
    monthly_data1: list
        List of data for the first group, with each element being the data for one month.
    :param monthly_data2: list
        List of data for the second group, with each element being the data for one month.
    :param months: list
        List of month names corresponding to the data.
    """
    def t_test(data1, data2, month):
        t_stat, p_value = ttest_ind(data1, data2)
        print(f"{month} - T-statistic: {t_stat}, P-value: {p_value}")
    
    for data1, data2, month in zip(monthly_data1, monthly_data2, months):
        t_test(data1, data2, month)