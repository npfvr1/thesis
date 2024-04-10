import os
import logging
from copy import deepcopy

import numpy as np
from sklearn.cluster import KMeans, SpectralClustering, AffinityPropagation
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../config", config_name="model")
def main(cfg : DictConfig) -> None:

    log = logging.getLogger(__name__)
    
    # ---- Load ----

    X = np.load(os.path.join("data", "processed", "X.npy"))
    labels = np.load(os.path.join("data", "processed", "labels.npy"))

    # ---- Clustering ----

    # cluster_algo = KMeans(n_clusters = cfg.hyperparameters.cluster_nb, random_state = 0, n_init = "auto").fit(X)

    cluster_algo = SpectralClustering(n_clusters = cfg.hyperparameters.cluster_nb, random_state = 0).fit(X)

    # cluster_algo = AffinityPropagation(random_state=0).fit(X)
    # cfg.hyperparameters.cluster_nb = 14
    
    # cluster_algo = {'labels_':[]}
    # cluster_algo.labels_ = GaussianMixture(n_components=cfg.hyperparameters.cluster_nb).fit_predict(X)

    # ---- Results ----

    # Explain the distributions of drugs and clusters
    drug_total = {1:0, 2:0, 3:0}
    cluster_total = {}

    for i in range(cfg.hyperparameters.cluster_nb):
        cluster_total[i] = 0

    drug_by_cluster = {1:deepcopy(cluster_total),
                    2:deepcopy(cluster_total),
                    3:deepcopy(cluster_total)
                    }
    
    cluster_by_drug = {}

    for i in range(cfg.hyperparameters.cluster_nb):
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

    # ---- Visualization ----

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(X[:,0], X[:,1], marker="o", c=cluster_algo.labels_)#, cmap='hsv')
    plt.title("Clustered data")
    plt.xlabel("Principal component 1")
    plt.ylabel("Principal component 2")
    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()
