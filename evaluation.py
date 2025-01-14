import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
import itertools

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
    rfm_before_detection = transformation.rfm_in_weeks_calculation_evaluation(customer_transactions,'2022-01-01', detection_time)
    rfm_before_detection = rfm_before_detection[rfm_before_detection['frequency'] > 0]

    after_detection_date = datetime.strptime(detection_time, "%Y-%m-%d") + timedelta(days=1)
    after_detection_date = after_detection_date.strftime("%Y-%m-%d")
    rfm_after_detection = transformation.rfm_in_weeks_calculation_evaluation(customer_transactions, after_detection_date, '2022-08-31')

    missing_customers_after = rfm_before_detection.index.difference(rfm_after_detection.index)
    df_missing_after = pd.DataFrame(index=missing_customers_after, columns=rfm_after_detection.columns).fillna(0)
    rfm_after_detection = pd.concat([rfm_after_detection, df_missing_after])

    missing_customers_before =  rfm_after_detection.index.difference(rfm_before_detection.index)
    df_missing_before = pd.DataFrame(index=missing_customers_before, columns=rfm_after_detection.columns)
    df_missing_before = df_missing_before.fillna(np.mean(rfm_after_detection['frequency']))
    rfm_before_detection = pd.concat([rfm_before_detection, df_missing_before])
    
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

    missing_customers_before =  rfm_after_detection.index.difference(rfm_before_detection.index)
    df_missing_before = pd.DataFrame(index=missing_customers_before, columns=rfm_after_detection.columns)
    df_missing_before = df_missing_before.fillna(np.mean(rfm_after_detection['monetary_value']))
    rfm_before_detection = pd.concat([rfm_before_detection, df_missing_before])

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

    missing_customers_before =  rfm_after_detection.index.difference(rfm_before_detection.index)
    df_missing_before = pd.DataFrame(index=missing_customers_before, columns=rfm_after_detection.columns)
    df_missing_before = df_missing_before.fillna(np.mean(rfm_after_detection['recency']))
    rfm_before_detection = pd.concat([rfm_before_detection, df_missing_before])

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
        data1 = pd.to_numeric(data1, errors='coerce')
        data2 = pd.to_numeric(data2, errors='coerce')
        t_stat, p_value = ttest_ind(data1, data2)
        print(f"{month} - T-statistic: {t_stat}, P-value: {p_value}")
    
    for data1, data2, month in zip(monthly_data1, monthly_data2, months):
        t_test(data1, data2, month)



def combine_and_aggregate(evaluation_list, comparison_type = 'previous_year'):
    # Specifying the comparison type
    if comparison_type == 'previous_year':
        frequency_col = 'frequency_ratio_previous_year'
        monetary_col = 'monetary_value_ratio_previous_year'
    elif comparison_type == 'same_period_last_year':
        frequency_col = 'frequency_ratio_same_period_last_year'
        monetary_col = 'monetary_value_ratio_same_period_last_year'
   
    valid_dfs = []
    for i, eval_result in enumerate(evaluation_list):
        if isinstance(eval_result, tuple):
            eval_df = eval_result[0]
        else:
            eval_df = eval_result
        
        if isinstance(eval_df, pd.DataFrame):
            valid_dfs.append(
                eval_df[[frequency_col, monetary_col]]
                .assign(detection=f"detection_{i+1}")  # Add detection ID if needed
            )
        else:
            print(f"Warning: Skipping invalid result at index {i}: {type(eval_result)}")

    # Combine all valid DataFrames
    combined = pd.concat(valid_dfs, ignore_index=True)

    # Return the combined ratios only
    return combined[[frequency_col, monetary_col]]




def pairwise_hypothesis_testing_with_means(results_dict, key_list, column='frequency_ratio_previous_year'):
    """
    Perform pairwise Mann-Whitney U tests between groups specified in key_list.
    
    Parameters
    ----------
    results_dict : dict
        A dictionary of DataFrames. Each key in `results_dict` corresponds to a group,
        and the value is a DataFrame that contains the data for that group.
    key_list : list
        A list of keys (group names) to include in the pairwise tests.
    column : str, optional
        The name of the column in each group's DataFrame which contains the numeric 
        values to be tested (default is 'frequency_ratio_previous_year').
    
    Returns
    -------
    pd.DataFrame
        A DataFrame containing the pairwise test results, including group names,
        group means, Mann-Whitney U statistics, and associated p-values.
    """

    # Initialize an empty list to store the results of each pairwise test
    test_results = []

    # Compute mean of the chosen column for each group in advance to avoid repeated calculations
    mean_ratios = {key: results_dict[key][column].mean() for key in key_list}

    # Generate all unique pairs of groups from key_list using combinations
    for key1, key2 in itertools.combinations(key_list, 2):
        # Extract the data series for the two groups
        group1 = results_dict[key1][column]
        group2 = results_dict[key2][column]

        # Perform the two-sided Mann-Whitney U test on the two groups
        stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')

        # Append a dictionary of results for the current pair to our list
        test_results.append({
            'Group 1': key1,
            'Group 1 Mean': mean_ratios[key1],
            'Group 2': key2,
            'Group 2 Mean': mean_ratios[key2],
            'Mann-Whitney U Statistic': stat,
            'P-value': p_value
        })

    # Convert the list of test results into a DataFrame and return it
    return pd.DataFrame(test_results)


def perform_statistical_test(detected_data, non_detected_data, comparison_type='previous'):
    """
    Perform the Mann-Whitney U test on frequency and monetary value ratios between detected and non-detected customers.
    
    Parameters:
    - detected_data (DataFrame): Data of detected customers with RFM ratios.
    - non_detected_data (DataFrame): Data of non-detected customers with RFM ratios.
    - comparison_type (str): The type of comparison, either 'previous', 'previous_year', or 'same_period_last_year'.
    
    Returns:
    - tuple: p-values for frequency and monetary value ratios.
    """
    
    # Choose the correct columns based on the comparison type
    if comparison_type == 'previous':
        frequency_col = 'frequency_ratio_previous'
        monetary_col = 'monetary_value_ratio_previous'
    elif comparison_type == 'previous_year':
        frequency_col = 'frequency_ratio_previous_year'
        monetary_col = 'monetary_value_ratio_previous_year'
    elif comparison_type == 'same_period_last_year':
        frequency_col = 'frequency_ratio_same_period_last_year'
        monetary_col = 'monetary_value_ratio_same_period_last_year'
   
    frequency_stat, frequency_p_value = mannwhitneyu(detected_data[frequency_col], non_detected_data[frequency_col], alternative='less')
    
    monetary_stat, monetary_p_value = mannwhitneyu(detected_data[monetary_col], non_detected_data[monetary_col], alternative='less')
    
    return frequency_p_value, monetary_p_value


def detection_evaluation_last_year(transaction_data, customer_ids, detection_date, validation_months=1):
    """
    Evaluate the ratios of customers' RFM metrics for validation periods of 1 or 3 months after detection, 
    comparing them to previous periods, previous year, and the same time a year ago. (This is the first detection type)

    Parameters:
    - transaction_data (DataFrame): Contains 'Customer ID', 'date', and 'Revenue' columns.
    - customer_ids (list): List of customer IDs to evaluate.
    - detection_date (str): The date of detection in 'YYYY-MM-DD' format.
    - validation_months (int): Number of months to evaluate after detection (1 or 3 months).

    Returns:
    - DataFrame: Contains the RFM ratios (validation / previous, validation / previous year, validation / same period last year) for each customer.
    """
    detection_date = pd.to_datetime(detection_date) 
    start_validation = detection_date
    end_validation = start_validation + pd.DateOffset(months=validation_months) - pd.Timedelta(days=1)

    start_previous = detection_date - pd.DateOffset(months=validation_months)
    end_previous = detection_date - pd.Timedelta(days=1)

    start_previous_year = detection_date - pd.DateOffset(years=1)
    end_previous_year = end_previous - pd.Timedelta(days=1)
    start_same_period_last_year = start_validation - pd.DateOffset(years=1)
    end_same_period_last_year = end_validation - pd.DateOffset(years=1)


    def calculate_rfm_for_period(start, end):
        return transformation.rfm_calculation(transaction_data, start, end, customer_ids)

    validation_rfm = calculate_rfm_for_period(start_validation, end_validation)
    previous_year_rfm = calculate_rfm_for_period(start_previous_year, end_previous_year)

    rfm_ratios = pd.DataFrame(index = customer_ids)
    rfm_ratios = rfm_ratios.join(validation_rfm[['recency', 'frequency', 'monetary_value']], how='left')
    rfm_ratios = rfm_ratios.join(previous_year_rfm[['recency', 'frequency', 'monetary_value']], how='left', rsuffix='_previous_year')


    absent_customers = rfm_ratios[
        (rfm_ratios['recency_previous_year'].isna() | (rfm_ratios['recency_previous_year'] == 0))  # Absent in previous period
    ]
    
    rfm_ratios = rfm_ratios[
        (rfm_ratios['recency_previous_year'] != 0) &
        (rfm_ratios['frequency_previous_year'] != 0) &
        (rfm_ratios['monetary_value_previous_year'] != 0)
    ]


    rfm_ratios['recency_ratio_previous_year'] = rfm_ratios['recency'] / rfm_ratios['recency_previous_year']
    rfm_ratios['frequency_ratio_previous_year'] = rfm_ratios['frequency'] / rfm_ratios['frequency_previous_year']
    rfm_ratios['monetary_value_ratio_previous_year'] = rfm_ratios['monetary_value'] / rfm_ratios['monetary_value_previous_year']
    

    return  rfm_ratios[['recency_ratio_previous_year', 'frequency_ratio_previous_year', 'monetary_value_ratio_previous_year']], absent_customers



def detection_evaluation_same_period_last_year(transaction_data, customer_ids, detection_date, validation_months=1):
    """
    Evaluate the ratios of customers' RFM metrics for validation periods of 1 or 3 months after detection, 
    comparing them to previous periods, previous year, and the same time a year ago. This is the second detection type.

    Parameters:
    - transaction_data (DataFrame): Contains 'Customer ID', 'date', and 'Revenue' columns.
    - customer_ids (list): List of customer IDs to evaluate.
    - detection_date (str): The date of detection in 'YYYY-MM-DD' format.
    - validation_months (int): Number of months to evaluate after detection (1 or 3 months).

    Returns:
    - DataFrame: Contains the RFM ratios (validation / previous, validation / previous year, validation / same period last year) for each customer.
    """
    detection_date = pd.to_datetime(detection_date) 
    start_validation = detection_date
    end_validation = start_validation + pd.DateOffset(months=validation_months) - pd.Timedelta(days=1)

    # Previous months immediately before detection
    start_previous = detection_date - pd.DateOffset(months=validation_months)
    end_previous = detection_date - pd.Timedelta(days=1)

    # Previous year
    start_previous_year = detection_date - pd.DateOffset(years=1)
    end_previous_year = end_previous - pd.Timedelta(days=1)


    # Same period last year
    start_same_period_last_year = start_validation - pd.DateOffset(years=1)
    #print(start_same_period_last_year)
    end_same_period_last_year = end_validation - pd.DateOffset(years=1)
    #print(end_same_period_last_year)
    
    # Helper function to calculate RFM metrics
    def calculate_rfm_for_period(start, end):
        return transformation.rfm_calculation(transaction_data, start, end, customer_ids)

    validation_rfm = calculate_rfm_for_period(start_validation, end_validation)
    same_period_last_year_rfm = calculate_rfm_for_period(start_same_period_last_year, end_same_period_last_year)

    rfm_ratios = pd.DataFrame(index = customer_ids)
    rfm_ratios = rfm_ratios.join(validation_rfm[['recency', 'frequency', 'monetary_value']], how='left')
    rfm_ratios = rfm_ratios.join(same_period_last_year_rfm[['recency', 'frequency', 'monetary_value']], how='left', rsuffix='_same_period_last_year')
    
    absent_customers = rfm_ratios[
        (rfm_ratios['recency_same_period_last_year'].isna() | (rfm_ratios['recency_same_period_last_year'] == 0))  # Absent in previous period
    ]
    
    rfm_ratios = rfm_ratios[
        (rfm_ratios['recency_same_period_last_year'] != 0) &
        (rfm_ratios['frequency_same_period_last_year'] != 0) &
        (rfm_ratios['monetary_value_same_period_last_year'] != 0)
    ]


    rfm_ratios['recency_ratio_same_period_last_year'] = rfm_ratios['recency'] / rfm_ratios['recency_same_period_last_year']
    rfm_ratios['frequency_ratio_same_period_last_year'] = rfm_ratios['frequency'] / rfm_ratios['frequency_same_period_last_year']
    rfm_ratios['monetary_value_ratio_same_period_last_year'] = rfm_ratios['monetary_value'] / rfm_ratios['monetary_value_same_period_last_year']
    

    return  rfm_ratios[['recency_ratio_same_period_last_year', 'frequency_ratio_same_period_last_year', 'monetary_value_ratio_same_period_last_year']], absent_customers



