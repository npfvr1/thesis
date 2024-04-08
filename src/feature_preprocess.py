import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn import decomposition, preprocessing
from matplotlib.colors import ListedColormap

        
def quantile_bucket(data, buckets):
    """
    Returns the quantile bucketed data.
    """
    assert len(data) > buckets, "More buckets than data points, not sure how that is going to work"
    q = np.linspace(0.0, 1.0, buckets + 1)
    bins = np.quantile(data, q)
    bins = bins[1:-1] # Without this, the last bucket will contain only the max value, because of how np.digitize works (see bins' bounds), and the buckets' indexes will start at 1 instead of 0
    data = np.digitize(data, bins = bins)
    return data


# ---- Load ----

df = pd.read_csv(os.path.join("data", "processed", "eeg_features4.csv"))

# ---- Substract baseline from corresponding post administration 1 and 2 ----

# Save the average/median baseline for the recordings that don't have a baseline 
average_baseline = df.drop(['drug', 'id'], axis=1).groupby("time").agg({'mean'}).values[0]
median_baseline = df.drop(['drug', 'id'], axis=1).groupby("time").agg({'median'}).values[0]

only_baseline_df = df[df['time'] == 0]
only_post_one_df = df[df['time'] == 1]
only_post_two_df = df[df['time'] == 2]
del df

for df in [only_post_one_df, only_post_two_df]:

    for index, row in df.iterrows():

        id = row['id']
        drug = row['drug']

        try:

            baseline_for_this_session = only_baseline_df.loc[(only_baseline_df['drug'] == drug) & (only_baseline_df['id'] == id)].values[0]

            # TODO : make it a loop over the features
            # Below depends on the structure of the feature file
            baseline_delta = baseline_for_this_session[3]
            baseline_theta = baseline_for_this_session[4]
            baseline_alpha = baseline_for_this_session[5]
            baseline_se = baseline_for_this_session[6]
            baseline_pe = baseline_for_this_session[7]
            baseline_zc = baseline_for_this_session[8]

            df.at[index, 'delta'] -= baseline_delta
            df.at[index, 'theta'] -= baseline_theta
            df.at[index, 'alpha'] -= baseline_alpha
            df.at[index, 'se'] -= baseline_se
            df.at[index, 'pe'] -= baseline_pe
            df.at[index, 'zc'] -= baseline_zc

        except Exception as e:
            # print(e)
            df.at[index, 'delta'] -= median_baseline[0]
            df.at[index, 'theta'] -= median_baseline[1]
            df.at[index, 'alpha'] -= median_baseline[2]
            df.at[index, 'se'] -= median_baseline[3]
            df.at[index, 'pe'] -= median_baseline[4]
            df.at[index, 'zc'] -= median_baseline[5]
            continue
        
# ---- Normalize data with quantile bucketing ----

df = pd.concat([only_post_one_df, only_post_two_df])
temp = []

for feature in df.columns:
    if feature in ['id', 'drug', 'time']:
        continue
    feature_vector = df[feature].values

    # # Visualize data distribution before
    # plt.figure()
    # plt.hist(feature_vector, bins=100)
    # plt.title("Data distribution for feature : {}".format(feature))

    feature_vector = quantile_bucket(feature_vector, 10) # TODO : hyperparameter
    temp.append(feature_vector)

    # # Visualize data distribution after
    # plt.figure()
    # plt.hist(feature_vector, bins=100)
    # plt.title("Data distribution for feature : {}".format(feature))

X = np.column_stack(temp)
# plt.show()
        
# ---- Normalize each feature in range [0, 1] ----

X = preprocessing.normalize(X, axis=1)

# ---- Dimensionality reduction with PCA ----

plt.cla()
nb_pca_components = 2 # TODO : hyperparameter
pca = decomposition.PCA(n_components = nb_pca_components)
pca.fit(X)
X = pca.transform(X)
for i in range(nb_pca_components):
    print("Coordinates of principal component {} in feature space: {}".format(i, pca.components_[i]))

# ---- One-way ANOVA to detect differences in features between DRUG groups ----

# population = pd.DataFrame(X)
# population = population.assign(drug=df['drug'].values)

# # Groups
# g1 = population[population['drug'] == 1]
# g2 = population[population['drug'] == 2]
# g3 = population[population['drug'] == 3]

# for i in range(nb_pca_components):
#     print("ANOVA test between drug groups based on principal component {} :".format(i + 1))
#     F, p = stats.f_oneway(g1[i], g2[i], g3[i])
#     print(F, p)

# ---- Visualize the PCA-transformed data ---

# x = X[:, 0]
# y = X[:, 1]
# #z = X[:, 2]
labels = df['drug'].values
# fig = plt.figure()
# ax = fig.add_subplot()#(projection='3d')
# ax.scatter(x, y, marker="+", c=labels, cmap=ListedColormap(['red', 'blue', 'green']))
# plt.xlabel("Principal component 1")
# plt.ylabel("Principal component 2")
# plt.grid()
# plt.show()

# df = pd.read_csv(os.path.join("data", "processed", "fnirs_features.csv"))
# fnirs_data = df.to_numpy()
# del df

# df = pd.read_csv(os.path.join("data", "processed", "pupillometry_features.csv")) # Make sure the keys are the same for pupillometry data
# pupillometry_data = df.to_numpy()
# del df

# ---- Concatenate into a single feature vector ----

# data = np.concatenate((eeg_data, fnirs_data, pupillometry_data), axis=?)

# ---- Save as a file ----

np.save(os.path.join("data", "processed", "X"), X)
np.save(os.path.join("data", "processed", "labels"), labels)