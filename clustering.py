import pandas as pd
import csv
from sklearn.cluster import KMeans
'''
transformation.py takes the transformed data frames and clusters them. The clustering is done in two "scopes", an outer
scope and an inner scope. The centroids themselves are sorted based on recency for the outer scope and based off
revenue for the inner scope.

Input: data frame with Customer ID as index and standardized RFM values.

Step 0 - Setup KMeans function + Euclidean distance calculation
Input: Data frame, w/optional inputs of clusters and n_init
Output: centroids

Step 1 - Compute anchor centroids
Input: outer_scope_unclassified
Output: outer_anchors
KMeans will be run on entire data set (the "outer scope") and the centroids from this clustering will be returned. It is
important to note that the ordering of centroids is based on their recency (2nd column) values.

Step 2 - Run KMeans on entire data set w/anchor
Input: rfm_stand_complete, outer_anchors
Output: outer_scope_classified
Using the outer scope anchors, a new data frame with the Euclidean to each centroid will be computed. A column called
"outer_scope_group" will added, representing the customers cluster classification based on the shortest distance to a
particular centroid.

Step 3 - Eliminate irrelevant clusters
Input: outer_scope_classified
Output: inner_scope_unclassified
Based on the customer's classification, they will be eliminated if they are in the 2 clusters w/highest recency.

Step 4 - Recalculate anchor based on new "inner" scope data set
Input: inner_scope_unclassified
Output: inner_anchors
KMeans will be run on remaining cluster and the centroids from this clustering will be returned. It is important to note 
that the ordering of centroids is based on their revenue (1st column) values.

Step 5 - Rerun KMeans on inner scope data set w/new anchors
Input: inner_scope_unclassified, inner_anchors
Output: inner_scope_classified
Using the inner scope anchors, a new data frame with the Euclidean to each centroid will be computed. A column called
"inner_scope_group" will added, representing the customers cluster classification based on the shortest distance to a
particular centroid.
'''


def k(df, clusters=2, random_state=None):
    # Here we set up a generic KMeans function
    kmeans = KMeans(n_clusters=clusters, random_state=random_state)
    clustering = kmeans.fit(df[['Revenue', 'Recency', 'Frequency']])

    return clustering.cluster_centers_


def distance(df, centroids):
    # this function computes the distances for each customer to each centroid. The shortest distance is then deemed to
    # be the customer's respective cluster. The  calculation used is the Euclidian distance.
    distances_list = []
    for centroid in centroids:
        distances = ((df['Revenue'] - centroid[0])**2 + (df['Recency'] - centroid[1])**2 + (df['Frequency'] -
                                                                                            centroid[2])**2)**.5
        distances_list.append(distances)

    # The distances for each customer to each centroid is then converted into a Pandas dataframe and transposed.
    distance_df = pd.DataFrame(distances_list)
    distance_df = distance_df.transpose()

    return distance_df


def outer_scope_centroids(outer_scope_unclassified, month_idx, num_of_clusters):
    # the outer anchors are computed by taking the outer scope RFM values for the anchor month and running the KMeans
    # clustering algorithm on them. The values returned are the coordinates of the respective centroids.
    # 3 clusters was determined by examining the elbow plot computed from the unclassified RFM outer scope values
    outer_anchors = k(outer_scope_unclassified[month_idx], clusters=num_of_clusters)
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


def elimination(outer_scope_classified):
    # we eliminate the outer scope clusters 1 and 2 in this step
    inner_scope_unclassified1 = outer_scope_classified[outer_scope_classified['outer_scope_group'] != 0].copy()
    outer_scope_dropped = outer_scope_classified[outer_scope_classified['outer_scope_group'] == 0].copy()
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
    inner_scope_unclassified['group'] = (inner_scope_classified.idxmin(axis="columns") +1)

    # Note: this variable name is misleading - the customers returned here are assigned an inner scope cluster.

    return inner_scope_unclassified
