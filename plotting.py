import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from scipy import stats
from sklearn.cluster import KMeans


def elbow_plot(*dfs, labels=None):
    """
    This function generates an elbow plot for multiple datasets to help determine the optimal number of clusters (K)
    for K-means clustering.

    Parameters:
    *dfs (DataFrame): The input dataframes with columns 'Revenue', 'Recency', and 'Frequency'.
    labels (list): A list of labels for the datasets.

    Returns:
    None: Displays an elbow plot.
    """
    
    plt.figure(figsize=(16, 8))

    for df, label in zip(dfs, labels):
        distortions = []
        K = range(2, 10)
        
        for k in K:
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
            kmeans.fit(df[['revenue', 'recency', 'frequency']])
            distortions.append(kmeans.inertia_)
        
        plt.plot(K, distortions, marker='o', label=label)

    plt.xlabel('Number of Clusters (K)', fontsize=17)
    plt.ylabel('Distortion', fontsize=17)
    plt.xticks(K)
    plt.legend()
    plt.show()
    

def plot3d(df, month_number=0):
    # this function plots the graph in 3D without any colors
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(df['Revenue'], df['Recency'], df['Frequency'])

    fig.suptitle('Month ' + str(month_number))
    ax.set_xlabel('Revenue', fontsize=14)
    ax.set_ylabel('Recency', fontsize=14)
    ax.set_zlabel('Frequency', fontsize=14)

    plt.show()


def create_color_map(df, group_column):
    unique_labels = sorted(df[group_column].unique())
    palette = sns.color_palette("coolwarm", len(unique_labels))
    return dict(zip(unique_labels, palette))

# Initialize the color map outside the functions
color_map_inner = None
color_map_outer = None

def plot3d_clusters_inner(df, centroids=None, month_number=0, elev=10, azim=-40):
    """
    Plots inner scope clusters in a 3D scatter plot with consistent color mapping.
    """
    global color_map_inner
    if color_map_inner is None:
        color_map_inner = create_color_map(df, 'group')

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(projection='3d')

    ax.scatter(df['recency'], df['revenue'], df['frequency'], c=df['group'].map(color_map_inner), s=50)

    ax.set_xlabel('Recency', fontsize=13, labelpad=2)
    ax.set_ylabel('Monetary Value', fontsize=13, labelpad=2)
    ax.set_zlabel('Frequency', fontsize=13, labelpad=0.1)

    ax.set_xlim3d(1, 0)
    ax.set_ylim3d(0, 1)
    ax.set_zlim3d(0, 1)

    ax.view_init(elev=elev, azim=azim)  
    handles = [mpatches.Patch(color=color_map_inner[label], label=label) for label in sorted(color_map_inner.keys())]
    ax.legend(handles=handles, title="Clusters & Sub-Clusters", fontsize=12, title_fontsize=13)
    plt.show()


def plot3d_clusters_outer(df, centroids=None, month_number=0, elev=10, azim=-40):
    """
    Plots outer scope clusters in a 3D scatter plot with consistent color mapping.
    """
    global color_map_outer
    if color_map_outer is None:
        color_map_outer = create_color_map(df, 'outer_scope_group')

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(projection='3d')

    ax.scatter(df['recency'], df['revenue'], df['frequency'], c=df['outer_scope_group'].map(color_map_outer), s=50)

    ax.set_xlabel('Recency', fontsize=13, labelpad=2)
    ax.set_ylabel('Monetary Value', fontsize=13, labelpad=2)
    ax.set_zlabel('Frequency', fontsize=13, labelpad=0.1)

    ax.set_xlim3d(1, 0)
    ax.set_ylim3d(0, 1)
    ax.set_zlim3d(0, 1)

    ax.view_init(elev=elev, azim=azim) 
    handles = [mpatches.Patch(color=color_map_outer[label], label=label) for label in sorted(color_map_outer.keys())]
    ax.legend(handles=handles, title="Clusters", fontsize=12, title_fontsize=13)
    plt.show()

def plot3d_clusters_inner_color(dfs, centroids=None):
    # this function plots the outer scope clusters with colors that are consistent montt to month
    number_of_colors = dfs[0]["inner_scope_group"].nunique()
    color_labels = dfs[0]["inner_scope_group"].unique()
    palette = sns.color_palette(n_colors=number_of_colors)
    color_map = dict(zip(color_labels, palette))

    for x in range(0, len(dfs)):

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ax.scatter(dfs[x]['Revenue'], dfs[x]['Recency'], dfs[x]['Frequency'],
                   c=dfs[x]['inner_scope_group'].map(color_map))

        fig.suptitle('Month ' + str(x+16))
        ax.set_xlabel('Revenue', fontsize=14)
        ax.set_ylabel('Recency', fontsize=14)
        ax.set_zlabel('Frequency', fontsize=14)
        ax.set_xlim3d(0, 1)
        ax.set_ylim3d(0, 1)
        ax.set_zlim3d(0, 1)

        if centroids is not None:
            ax.text(centroids[0][0], centroids[0][1], centroids[0][2], 'Centroid 1')
            ax.text(centroids[1][0], centroids[1][1], centroids[1][2], 'Centroid 2')
            ax.text(centroids[2][0], centroids[2][1], centroids[2][2], 'Centroid 3')
            ax.text(centroids[3][0], centroids[3][1], centroids[3][2], 'Centroid 4')
            ax.text(centroids[4][0], centroids[4][1], centroids[4][2], 'Centroid 5')

        plt.show()



def plot_calibration_purchases_vs_holdout_purchases(
    model, calibration_holdout_matrix, kind="frequency_cal", n=20, **kwargs
):
    """
    Plot calibration purchases vs holdout.

    This currently relies too much on the lifetimes.util calibration_and_holdout_data function.

    Parameters
    ----------
    model: lifetimes model
        A fitted lifetimes model.
    calibration_holdout_matrix: pandas DataFrame
        DataFrame from calibration_and_holdout_data function.
    kind: str, optional
        x-axis :"frequency_cal". Purchases in calibration period,
                 "recency_cal". Age of customer at last purchase,
                 "T_cal". Age of customer at the end of calibration period,
                 "time_since_last_purchase". Time since user made last purchase
    n: int, optional
        Number of ticks on the x axis
    Returns
    -------
    axes: matplotlib.AxesSubplot

    """

    summary = calibration_holdout_matrix.copy()
    duration_holdout = summary.iloc[0]["duration_holdout"]

    summary["model_predictions"] = model.conditional_expected_number_of_purchases_up_to_time(
            duration_holdout, summary["frequency_cal"].values, summary["recency_cal"].values, summary["T_cal"].values)

    ax = summary.groupby("frequency_cal")[["frequency_holdout", "model_predictions"]].mean().iloc[:n].plot(**kwargs)


    plt.title(" ")
    plt.xlabel("frequency_cal", size = 14)
    plt.ylabel("Average of Purchases in Validation Period", size = 12)
    plt.legend(['Actual Purchases', 'Model Predictions'])

    return ax


def plot_customer_clv_changes(clv_list, customer_id, time_standing, ylim):

    clvs = []
    valid_dates = []
    for period, clv in zip(time_standing, clv_list):
        if customer_id in clv.index:
            clvs.append(clv[customer_id])
            valid_dates.append(period)

    if clvs:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(valid_dates, clvs, linestyle='-', color='b')
    
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=len(valid_dates), maxticks=len(valid_dates)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()
        ax.set_xlabel('Date Making Calculation', size = 14)
        ax.set_ylim(0, ylim)
        ax.set_ylabel('Four-Week Customer Lifetime Value', size = 14)
        plt.show()
    else:
        print(f"No CLV data found for customer ID {customer_id}.")


def plot_mahalanobis_distances_qq(mahalanobis_distances_transformed, mahalanobis_distances, chi2_sparams=(3,)):

    """
    Plot the Q-Q plot of the distribution Mahalanobis distances of RFM changes their mean
    against the theoretical distribution of the Chi-square distribution.

    Parameters
    ----------
    mahalanobis_distances_transformed : array_like
        Array of Mahalanobis distances for the transformed data.
    mahalanobis_distances : array_like
        Array of Mahalanobis distances for the original data.
    chi2_sparams : tuple, optional
        Parameters for the chi-squared distribution. Default is (3,).

    Returns
    -------
    None
    """
    plt.figure(figsize=(13, 10))

    # Plot for transformed data
    res_transformed = stats.probplot(mahalanobis_distances_transformed**2, dist="chi2", sparams=chi2_sparams, plot=None)
    plt.plot(res_transformed[0][0], res_transformed[0][1], 'go', label='Transformed Data')
    transformed_line = np.poly1d(np.polyfit(res_transformed[0][0], res_transformed[0][1], 1))
    plt.plot(res_transformed[0][0], transformed_line(res_transformed[0][0]), 'r-')

    # Plot for original data
    res_original = stats.probplot(mahalanobis_distances**2, dist="chi2", sparams=chi2_sparams, plot=None)
    plt.plot(res_original[0][0], res_original[0][1], 'bx', label='Raw Data')
    original_line = np.poly1d(np.polyfit(res_original[0][0], res_original[0][1], 1))

    plt.plot(res_original[0][0], original_line(res_original[0][0]), 'b-')
    plt.xlabel('Chi-squared Distribution Quantiles', size=18)
    plt.ylabel('Mahalanobis Distances^2', size=18)
    plt.ylim([0, 100])
    plt.legend(loc='upper left', fontsize=14)
    plt.show()