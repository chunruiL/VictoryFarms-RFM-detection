import pandas as pd
import csv
from sklearn.cluster import KMeans


def k(df, clusters=2, random_state=40):
    # Here we set up a generic KMeans function
    kmeans = KMeans(n_clusters=clusters, random_state=random_state)
    clustering = kmeans.fit(df[['revenue', 'recency', 'frequency']])

    return clustering.cluster_centers_


def distance(df, centroids):
    '''
    this function computes the distances for each customer to each centroid. The shortest distance is then deemed to
    be the customer's respective cluster. The  calculation used is the Euclidian distance.
    '''
    
    distances_list = []
    for centroid in centroids:
        distances = ((df['revenue'] - centroid[0])**2 + (df['recency'] - centroid[1])**2 + (df['frequency'] -
                                                                                            centroid[2])**2)**.5
        distances_list.append(distances)

    # The distances for each customer to each centroid is then converted into a Pandas dataframe and transposed.
    distance_df = pd.DataFrame(distances_list)
    distance_df = distance_df.transpose()

    return distance_df


def outer_scope_centroids(outer_scope_unclassified, detection_idx, num_of_clusters):
    outer_anchors = k(outer_scope_unclassified[detection_idx], clusters=num_of_clusters)
    # these centroids are then sorted based on recency.
    outer_anchors = outer_anchors[outer_anchors[:, 0].argsort()]

    return outer_anchors


def outer_scope_handler(outer_scope_unclassified, outer_anchors):
    # We then compute the distances from each customer to each centroid
    outer_scope_classified = distance(outer_scope_unclassified, outer_anchors)
    # We then create a new column that represents the customer's cluster membership based on the distances returned
    outer_scope_unclassified['outer_scope_group'] = outer_scope_classified.idxmin(axis="columns")

    # Note: this variable name is misleading - the customers returned here are assigned an outer scope cluster.

    return outer_scope_unclassified


''''
def elimination(outer_scope_classified):
    # we eliminate the outer scope clusters 1 and 2 in this step
    inner_scope_unclassified1 = outer_scope_classified[outer_scope_classified['outer_scope_group'] != 0].copy()
    outer_scope_dropped = outer_scope_classified[outer_scope_classified['outer_scope_group'] == 0].copy()
    outer_scope_dropped['group'] = outer_scope_dropped['outer_scope_group']

    return inner_scope_unclassified1, outer_scope_dropped
'''

def elimination(outer_scope_classified):
    # we eliminate the outer scope clusters 1 and 2 in this step
    inner_scope_unclassified1 = outer_scope_classified[~outer_scope_classified['outer_scope_group'].isin([0, 1])].copy()
    outer_scope_dropped = outer_scope_classified[outer_scope_classified['outer_scope_group'].isin([0, 1])].copy()
    outer_scope_dropped['group'] = outer_scope_dropped['outer_scope_group']

    return inner_scope_unclassified1, outer_scope_dropped


def inner_scope_centroids(inner_scope_unclassified, num_of_clusters):
    # here we do the same thing as with the outer, except using 3 clusters (which we got from our elbow plot)
    # we also obviously are only using the remaining cluster
    inner_anchors = k(inner_scope_unclassified, clusters=num_of_clusters)

    # We then sort these centroids based on revenue
    inner_anchors = inner_anchors[inner_anchors[:, 0].argsort()]

    return inner_anchors


def inner_scope_handler(inner_scope_unclassified, inner_anchors):
    # We then compute the distances from each customer to each centroid
    inner_scope_classified = distance(inner_scope_unclassified, inner_anchors)
    # We then create a new column that represents the customer's cluster membership based on the distances returned
    inner_scope_unclassified['group'] = (inner_scope_classified.idxmin(axis="columns") +2)

    # Note: this variable name is misleading - the customers returned here are assigned an inner scope cluster.

    return inner_scope_unclassified



def run_clustering_pipeline(training_data, testing_data, num_detections=3, outer_clusters=4, inner_clusters=3, threshold = -3):
    """
    Run a multi-step clustering pipeline on training and test data for a specified number of detections.

    Parameters
    ----------
    training_data : list of pd.DataFrame
        A list of training DataFrames.
        Each entry should have scaled RFM columns..
    testing_data : list of pd.DataFrame
        A list of testing DataFrames.
        Each entry should have scaled RFM columns.
    num_detections : int, optional
        The number of times (or iterations) to run the pipeline. Default is 3.
    outer_clusters : int, optional
        The number of clusters for the "outer" scope. Default is 4.
    inner_clusters : int, optional
        The number of clusters for the "inner" scope. Default is 3.
    threshold: int, optional
        The threshold for determining the signficant negative movements. Default is -3.
        Change this according to how many customers how many customers we want to flag. 

    Returns
    -------
    classified_training : list of pd.DataFrame
        A list of DataFrames, each containing the final classified (clustered) training data
        for the corresponding detection.
    classified_test : list of pd.DataFrame
        A list of DataFrames, each containing the final classified (clustered) test data
        for the corresponding detection.
    cluster_movements : list
        A list of lists, where each inner list contains the detected customer indices
        whose cluster label moved significantly (<= -3).
    """

    classified_training = []
    classified_test = []

    # 1. Run the multi-step clustering pipeline for each detection
    for detection in range(num_detections):
        # ---- Outer scope (training) ----
        outer_anchors = outer_scope_centroids(
            training_data, detection_idx=detection, num_of_clusters=outer_clusters
        )
        outer_scope_classified = outer_scope_handler(training_data[detection], outer_anchors)

        # Eliminate out-of-scope clusters
        inner_scope, dropped_cluster = elimination(outer_scope_classified)

        # ---- Inner scope (training) ----
        inner_anchor = inner_scope_centroids(inner_scope, num_of_clusters=inner_clusters)
        inner_scope_classified = inner_scope_handler(inner_scope, inner_anchor)

        # Final classified training data
        final_classified_training = pd.concat([dropped_cluster, inner_scope_classified], axis=0)

        # ---- Outer scope (test) ----
        outer_scope_classified_test = outer_scope_handler(testing_data[detection], outer_anchors)

        # Eliminate out-of-scope clusters (test)
        inner_scope_test, dropped_cluster_test = elimination(outer_scope_classified_test)

        # Reuse the training inner_anchor for test data
        inner_scope_classified_test = inner_scope_handler(inner_scope_test, inner_anchor)

        # Final classified test data
        final_classified_test = pd.concat([dropped_cluster_test, inner_scope_classified_test], axis=0)

        # Store the classified results
        classified_training.append(final_classified_training)
        classified_test.append(final_classified_test)

    # 2. Analyze cluster movement from training to test for each detection
    cluster_movements = []
    for detection in range(num_detections):
        # Summarzing the inter-cluster movement
        movement_info = classified_test[detection]['group'] - classified_training[detection]['group']

        # Detect customers whose cluster label moved significantly using pre-determined threshold
        detection_list = [idx for idx, v in movement_info.items() if (v <= threshold)]
        cluster_movements.append(detection_list)

        # Printing the results
        print(f"Detection {detection+1} cluster movement distribution:\n{movement_info.value_counts()}\n")

    return classified_training, classified_test, cluster_movements

