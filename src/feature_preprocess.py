import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn import decomposition, preprocessing
from sklearn.linear_model import LinearRegression
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

df_eeg = pd.read_csv(os.path.join("data", "processed", "eeg_features6 copy.csv"))
# df_fnirs = pd.read_csv(os.path.join("data", "processed", "fnirs_features.csv"))
df_pupillometry = pd.read_csv(os.path.join("data", "processed", "pupillometry_features.csv"))

# ---- Concatenate into a single feature vector ----

# print(df_pupillometry.shape)
# df_pupillometry = df_pupillometry.dropna()
# print(df_pupillometry.shape)

df_eeg.id.astype('Int64')
df_pupillometry.id.astype('Int64')

df = pd.merge(left=df_eeg, right=df_pupillometry, left_on=['id', 'drug', 'time'], right_on=['id', 'drug', 'time'])
del df_eeg, df_pupillometry

# print(df.shape)
# df = df.dropna()
# print(df.shape)

# ---- Substract baseline from corresponding post administration 1 and 2 ----

substract_baseline = False

if substract_baseline:
    # Save the average/median baseline for the recordings that don't have a baseline # TODO : is that a good idea? Probably not
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
                df.loc[index,df.columns.values[3:]] -= baseline_for_this_session[3:]
            except Exception as e: # Most likely the baseline was not found
                # print(e)
                df.loc[index,df.columns.values[3:]] -= median_baseline
                # df.loc[index,df.columns.values[3:]] = [np.nan] * len(median_baseline)

    df = pd.concat([only_post_one_df, only_post_two_df])
        
df = df.dropna()


# ---- REGRESSION TEST ----

drug_one_df = df[df['drug'] == 1]
drug_two_df = df[df['drug'] == 2]
drug_three_df = df[df['drug'] == 3]

c = len(df.columns) - 3
del df

time_dict = {0:0, 1:15, 2:60}
color_dict = {0:"red", 1:"green", 2:"blue"}

fig, axs = plt.subplots(3, c, constrained_layout=True)
for drug_id, df in enumerate([drug_one_df, drug_two_df, drug_three_df]):

    for feature_id, feature in enumerate(df.columns):

        if feature in ['id', 'drug', 'time']:
            continue

        y = []
        x = []
        weights = []

        for time in range(3):

            temp_df = df[df['time'] == time]
            temp_y = temp_df[feature].values
            temp_y = quantile_bucket(temp_y, 10)
            del temp_df
            y.append(np.mean(temp_y))
            x.append(time_dict[time])
            weights.append(len(temp_y))

            # y.append(temp_y)


        ax = axs[drug_id, feature_id - 3]

        # ax.boxplot(y)

        ax.scatter(x, y, c = color_dict[drug_id])
        reg = LinearRegression().fit(np.array([x]).T, y, sample_weight=weights)
        dummy_x = np.linspace(0, 60, 100)
        ax.plot(dummy_x, reg.predict(np.array([dummy_x]).T), c = "grey")

        if drug_id == 0:
            ax.set_title("{}".format(feature))

        ax.tick_params(bottom=False, labelbottom=False)          

    axs[drug_id, 0].set_ylabel("Drug {} ({} recordings)".format(drug_id + 1, weights))

plt.show()

# ---- Normalize data with quantile bucketing ----

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
    # plt.show()

X = np.column_stack(temp)
        
# ---- Normalize each feature in range [0, 1] ----

X = preprocessing.normalize(X, axis=1)

# ---- Dimensionality reduction with PCA ----

plt.cla()
nb_pca_components = 2 # TODO : hyperparameter
pca = decomposition.PCA(n_components = nb_pca_components, random_state = 1)
pca.fit(X)
X = pca.transform(X)
for i in range(nb_pca_components):
    print("PC {} in feature space: {}".format(i + 1, pca.components_[i]))
    print("Variance explained : {}".format(pca.explained_variance_[i]))

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

x = X[:, 0]
y = X[:, 1]
#z = X[:, 2]
labels = df['drug'].values
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot()#(projection='3d')
ax.scatter(x, y, marker="o", c=labels, cmap=ListedColormap(['red', 'blue', 'green']))
plt.title("Data distribution ({}D reduced to {}D, {} points)".format(len(df.columns.values[3:]),
                                                          nb_pca_components,
                                                          len(x)
                                                          )
                                                          )
plt.xlabel("PC 1\n\nFeatures are {}\n\nPC 1 is {}    PC 2 is {}".format(df.columns.values[3:],
                                                                        [np.round(x, 1) for x in pca.components_[0]],
                                                                        [np.round(x, 1) for x in pca.components_[1]]
                                                                        )
                                                                        )
plt.ylabel("PC 2")
plt.locator_params(nbins=3)
plt.subplots_adjust(bottom=0.25)
plt.grid()
plt.show()

# ---- Save as a file ----

np.save(os.path.join("data", "processed", "X"), X)
np.save(os.path.join("data", "processed", "labels"), labels)