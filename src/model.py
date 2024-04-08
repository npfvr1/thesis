import os
import logging
from copy import deepcopy

import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
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

    kmeans = KMeans(n_clusters = cfg.hyperparameters.cluster_nb, random_state = 0, n_init = "auto").fit(X)
    # sp = SpectralClustering(n_clusters = cfg.hyperparameters.cluster_nb, random_state = 0).fit(X)

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

    for drug_id, cluster_id in zip(labels, kmeans.labels_):

        drug_total[drug_id] += 1
        cluster_total[cluster_id] += 1

        drug_by_cluster[drug_id][cluster_id] += 1
        cluster_by_drug[cluster_id][drug_id] += 1

    log.info("Distribution by drug:")
    for drug_id in drug_total:
        log.info("Drug {} :".format(drug_id))
        for cluster_id in cluster_total:
            log.info("\t{} ({}%) in cluster {}".format(drug_by_cluster[drug_id][cluster_id],
                                                    np.round(drug_by_cluster[drug_id][cluster_id] / drug_total[drug_id] * 100, 1),
                                                    cluster_id))

    log.info("Composition of clusters:")
    for cluster_id in cluster_total:
        log.info("Cluster {} :".format(cluster_id))
        for drug_id in drug_total:
            log.info("\t{} ({}%) of drug {}".format(cluster_by_drug[cluster_id][drug_id],
                                                np.round(cluster_by_drug[cluster_id][drug_id] / cluster_total[cluster_id] * 100, 1),
                                                drug_id))

    # ---- Visualization ----

    # fig = plt.figure()
    # ax = fig.add_subplot()
    # ax.scatter(X[:,0], X[:,1], marker="+", c=kmeans.labels_)
    # plt.title("k-Means")
    # plt.xlabel("Principal component 1")
    # plt.ylabel("Principal component 2")
    # plt.grid()

    # fig = plt.figure()
    # ax = fig.add_subplot()
    # ax.scatter(X[:,0], X[:,1], marker="+", c=sp.labels_)
    # plt.title("Spectral clustering")
    # plt.xlabel("Principal component 1")
    # plt.ylabel("Principal component 2")
    # plt.grid()

    # plt.show()


if __name__ == "__main__":
    main()
