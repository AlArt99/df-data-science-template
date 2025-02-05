import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/02_outliers_removed_chauvenet.pkl")

predictor_columns = list(df.columns[:6])

plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------

for col in predictor_columns:
    df[col] = df[col].interpolate()

df.info()

# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------

for set in df["set"].unique():
    start = df[df["set"] == set].index[0]
    end = df[df["set"] == set].index[-1]

    df.loc[(df["set"] == set), "duration"] = (end - start).seconds

df.groupby("label")["duration"].mean()

# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------

df_lowpass = df.copy()

lowpass = LowPassFilter()

sampling_freq = 1000 / 200
cutoff = 1.2

for col in predictor_columns:
    df_lowpass = lowpass.low_pass_filter(
        df_lowpass, col, sampling_freq, cutoff, order=5
    )
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    del df_lowpass[col + "_lowpass"]

df_lowpass.head()
df.head()
# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------

df_pca = df_lowpass.copy()

PCA = PrincipalComponentAnalysis()

pc_var = PCA.determine_pc_explained_variance(df_pca, predictor_columns)

plt.figure(figsize=(10, 10))
plt.plot(range(1, len(predictor_columns) + 1), pc_var, marker="o")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance")
plt.title("Scree Plot")
plt.show()

# As we can see, the first 3 components explain most of the variance in the data.
# We will use these 3 components to create new features.

df_pca = PCA.apply_pca(df_pca, predictor_columns, 3)

# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------

df_squared = df_pca.copy()

df_squared["acc_r"] = np.sqrt(
    df_squared["acc_x"] ** 2 + df_squared["acc_y"] ** 2 + df_squared["acc_z"] ** 2
)
df_squared["gyr_r"] = np.sqrt(
    df_squared["gyr_x"] ** 2 + df_squared["gyr_y"] ** 2 + df_squared["gyr_z"] ** 2
)


# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------

df_abstraction = df_squared.copy()

num_abstraction = NumericalAbstraction()

predictor_columns = predictor_columns + ["acc_r", "gyr_r"]

abstract_subset = []
for s in df_abstraction["set"].unique():
    subset = df_abstraction[df_abstraction["set"] == s].copy()
    for col in predictor_columns:
        subset = num_abstraction.abstract_numerical(subset, [col], 5, "mean")
        subset = num_abstraction.abstract_numerical(subset, [col], 5, "std")
    abstract_subset.append(subset)

df_abstraction = pd.concat(abstract_subset)


# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------

df_freq = df_abstraction.copy().reset_index()

freq_abstraction = FourierTransformation()

sampling_s = int(1000 / 200)
window_s = int(2800 / 200)

df_freq = freq_abstraction.abstract_frequency(df_freq, ["acc_y"], window_s, sampling_s)

df_freq_list = []
for s in df_freq["set"].unique():
    subset = df_freq[df_freq["set"] == s].reset_index(drop=True).copy()
    subset = freq_abstraction.abstract_frequency(
        subset, predictor_columns, window_s, sampling_s
    )
    df_freq_list.append(subset)

df_freq = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)

# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------

df_freq = df_freq.dropna()
df_freq = df_freq.iloc[::2]

# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------

df_cluster = df_freq.copy()

cluster_col = ["acc_x", "acc_y", "acc_z"]
k_values = range(2, 10)
inertias = []

for k in k_values:
    subset = df_cluster[cluster_col]
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
    cluster_labels = kmeans.fit_transform(subset)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 10))
plt.plot(k_values, inertias, marker="o")


kmeans = KMeans(n_clusters=5, n_init=20, random_state=0)
subset = df_cluster[cluster_col]
df_cluster["cluster"] = kmeans.fit_predict(subset)

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for c in df_cluster["cluster"].unique():
    subset = df_cluster[df_cluster["cluster"] == c]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=c)
ax.set_xlabel("acc_x")
ax.set_ylabel("acc_y")
ax.set_zlabel("acc_z")
plt.legend()
plt.show()

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for l in df_cluster["label"].unique():
    subset = df_cluster[df_cluster["label"] == l]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=l)
ax.set_xlabel("acc_x")
ax.set_ylabel("acc_y")
ax.set_zlabel("acc_z")
plt.legend()
plt.show()

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

df_cluster.to_pickle("../../data/interim/03_data_features.pkl")
