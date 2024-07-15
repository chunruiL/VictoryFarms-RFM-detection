import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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
    rfm_after_detection = transformation.rfm_in_weeks_calculation(customer_transactions, after_detection_date, '2022-08-31')

    missing_customers = rfm_before_detection.index.difference(rfm_after_detection.index)
    df_missing = pd.DataFrame(index=missing_customers, columns=rfm_after_detection.columns).fillna(0)
    rfm_after_detection = pd.concat([rfm_after_detection, df_missing])

    mean_ratio = np.mean(rfm_after_detection['frequency'] / rfm_before_detection['frequency'])
    std =  np.std(rfm_after_detection['frequency']/rfm_before_detection['frequency'])
    return mean_ratio, std 


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
    after_detection_date = datetime.strptime(detection_time, "%Y-%m-%d") + timedelta(days=1)
    after_detection_date = after_detection_date.strftime("%Y-%m-%d")
    rfm_after_detection = transformation.rfm_in_weeks_calculation_evaluation(customer_transactions, after_detection_date, '2022-08-31')

    mean_ratio = np.mean(rfm_after_detection['monetary_value'] / rfm_before_detection['monetary_value'])
    std_ratio = np.std(rfm_after_detection['monetary_value'] / rfm_before_detection['monetary_value'])

    return mean_ratio, std_ratio


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
    after_detection_date = datetime.strptime(detection_time, "%Y-%m-%d") + timedelta(days=1)
    after_detection_date = after_detection_date.strftime("%Y-%m-%d")
    rfm_after_detection = transformation.rfm_in_weeks_calculation_evaluation(transaction_data, after_detection_date, '2022-08-31')

    mean_ratio = np.mean(rfm_after_detection['recency'] / rfm_before_detection['recency'])
    std_ratio = np.std(rfm_after_detection['recency'] / rfm_before_detection['recency'])

    return mean_ratio, std_ratio
