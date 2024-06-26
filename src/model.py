import os
import logging
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn import decomposition
from sklearn.cluster import KMeans #, SpectralClustering, AffinityPropagation
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def main() -> None:

    log = logging.getLogger(__name__)
    
    # ---- Load ----

    df = pd.read_csv(os.path.join("data", "processed", "lmm_data.csv"))
    labels = df['drug'].values

    # ---- PCA ---- 

    pca = decomposition.PCA(n_components = 2, random_state = 1)
    data = np.column_stack([df[feature].values for feature in df.columns if feature not in ['id', 'drug', 'time']])
    X = pca.fit_transform(data)

    # ---- Clustering ----

    NB_CLUSTERS = 5

    # cluster_algo = KMeans(n_clusters = NB_CLUSTERS, random_state = 0, n_init = "auto").fit(X)

    # cluster_algo = SpectralClustering(n_clusters = NB_CLUSTERS, random_state = 0).fit(X)

    # cluster_algo = AffinityPropagation(random_state=0).fit(X)
    # cfg.hyperparameters.cluster_nb = 14
    
    class e():  
        def __init__(self):  
            self.labels_ = GaussianMixture(n_components=NB_CLUSTERS).fit_predict(X)
    cluster_algo = e()

    # ---- Results and scoring ----

    # Explain the distributions of drugs and clusters
    drug_total = {1:0, 2:0, 3:0}
    cluster_total = {}
    score = 0

    for i in range(NB_CLUSTERS):
        cluster_total[i] = 0

    drug_by_cluster = {1:deepcopy(cluster_total),
                    2:deepcopy(cluster_total),
                    3:deepcopy(cluster_total)}
    
    cluster_by_drug = {}

    for i in range(NB_CLUSTERS):
        cluster_by_drug[i] = deepcopy(drug_total)

    for drug_id, cluster_id in zip(labels, cluster_algo.labels_):

        drug_total[drug_id] += 1
        cluster_total[cluster_id] += 1

        drug_by_cluster[drug_id][cluster_id] += 1
        cluster_by_drug[cluster_id][drug_id] += 1

    print("Distribution by drug:")
    for drug_id in drug_total:
        print("Drug {} :".format(drug_id))
        for cluster_id in cluster_total:
            print("\t{} ({}%) in cluster {}".format(drug_by_cluster[drug_id][cluster_id],
                                                    np.round(drug_by_cluster[drug_id][cluster_id] / drug_total[drug_id] * 100, 1),
                                                    cluster_id))

    print("Composition of clusters:")
    for cluster_id in cluster_total:
        print("Cluster {} :".format(cluster_id))
        for drug_id in drug_total:
            print("\t{} ({}%) of drug {}".format(cluster_by_drug[cluster_id][drug_id],
                                                np.round(cluster_by_drug[cluster_id][drug_id] / cluster_total[cluster_id] * 100, 1),
                                                drug_id))
        min_drug_population = min([cluster_by_drug[cluster_id][drug_id] / cluster_total[cluster_id] for drug_id in drug_total])
        score += cluster_total[cluster_id] * (1 - min_drug_population)

    print("Experiment score = {}".format(score))

    # ---- Visualization ----

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(X[:,0], X[:,1], marker="o", c=labels, cmap=ListedColormap(['red', 'blue', 'green']))
    plt.title("Original data")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(X[:,0], X[:,1], marker="o", c=cluster_algo.labels_)#, cmap='hsv')
    plt.title("Clustered data")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")

    plt.show()


if __name__ == "__main__":
    main()
