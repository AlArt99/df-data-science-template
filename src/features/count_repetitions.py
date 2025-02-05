import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter
from scipy.signal import argrelextrema
from sklearn.metrics import mean_absolute_error

pd.options.mode.chained_assignment = None


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/01_data_preprocessed.pkl")
df = df[df["label"] != "rest"]

df["acc_r"] = np.sqrt(df["acc_x"] ** 2 + df["acc_y"] ** 2 + df["acc_z"] ** 2)
df["gyr_r"] = np.sqrt(df["gyr_x"] ** 2 + df["gyr_y"] ** 2 + df["gyr_z"] ** 2)

# --------------------------------------------------------------
# Split data
# --------------------------------------------------------------
df["label"].unique()

df_bench = df[df["label"] == "bench"]
df_ohp = df[df["label"] == "ohp"]
df_row = df[df["label"] == "row"]
df_squat = df[df["label"] == "squat"]
df_dead = df[df["label"] == "dead"]

# --------------------------------------------------------------
# Visualize data to identify patterns
# --------------------------------------------------------------

plot_df = df_bench
plot_df[plot_df["set"] == 1]["acc_x"].plot()
plot_df[plot_df["set"] == 1]["acc_y"].plot()
plot_df[plot_df["set"] == 1]["acc_z"].plot()
plot_df[plot_df["set"] == 1]["acc_r"].plot()

plot_df[plot_df["set"] == 1]["gyr_x"].plot()
plot_df[plot_df["set"] == 1]["gyr_y"].plot()
plot_df[plot_df["set"] == 1]["gyr_z"].plot()
plot_df[plot_df["set"] == 1]["gyr_r"].plot()


# --------------------------------------------------------------
# Configure LowPassFilter
# --------------------------------------------------------------

lowpass = LowPassFilter()

sf = 1000 / 200

df_bench = df_bench.reset_index(drop=True)

bench_lowpass = lowpass.low_pass_filter(df_bench, "acc_r", sf, 0.6, order=5)

bench_lowpass[bench_lowpass["set"] == bench_lowpass["set"].unique()[-3]]["acc_r"].plot(
    legend=True
)
bench_lowpass[bench_lowpass["set"] == bench_lowpass["set"].unique()[-3]][
    "acc_r_lowpass"
].plot(legend=True)

# --------------------------------------------------------------
# Apply and tweak LowPassFilter
# --------------------------------------------------------------

bench_lowpass = lowpass.low_pass_filter(df_bench, "acc_r", sf, 0.6, order=5)

bench_lowpass[bench_lowpass["set"] == bench_lowpass["set"].unique()[-3]]["acc_r"].plot(
    legend=True
)
bench_lowpass[bench_lowpass["set"] == bench_lowpass["set"].unique()[-3]][
    "acc_r_lowpass"
].plot(legend=True)

# --------------------------------------------------------------
# Create function to count repetitions
# --------------------------------------------------------------


def count_repetitions(data, col, cut_off=0.4, sf=1000 / 200, order=5) -> pd.DataFrame:

    data = data.reset_index(drop=True)
    data = lowpass.low_pass_filter(
        data_table=data,
        col=col,
        sampling_frequency=sf,
        cutoff_frequency=cut_off,
        order=order,
    )
    indexes = argrelextrema(data[col + "_lowpass"].values, np.greater)
    pickes = data.iloc[indexes]

    fig, ax = plt.subplots()
    plt.plot(data[f"{col}_lowpass"])
    plt.plot(pickes[f"{col}_lowpass"], "ro", color="red")
    ax.set_ylabel(f"{col}_lowpass")
    exercise = data["label"].iloc[0].title()
    category = data["category"].iloc[0].title()
    plt.title(f"{exercise} {category}: {len(pickes)} repetitions")
    plt.show()

    return len(pickes)


df_bench = df_bench[df_bench["set"] == df_bench["set"].unique()[0]]
df_ohp = df_ohp[df_ohp["set"] == df_ohp["set"].unique()[0]]
df_row = df_row[df_row["set"] == df_row["set"].unique()[0]]
df_squat = df_squat[df_squat["set"] == df_squat["set"].unique()[0]]
df_dead = df_dead[df_dead["set"] == df_dead["set"].unique()[0]]

count_repetitions(df_squat, "acc_r", cut_off=1, order=10)

# --------------------------------------------------------------
# Create benchmark dataframe
# --------------------------------------------------------------

df["actual_reps"] = df["category"].apply(lambda x: 5 if x == "heavy" else 10)

rep_df = df.groupby(["label", "category", "set"])["actual_reps"].max().reset_index()
rep_df["predicted_reps"] = 0

# --------------------------------------------------------------
# Evaluate the results
# --------------------------------------------------------------

for s in rep_df["set"].unique():

    subset = df[df["set"] == s]

    col = "acc_r"
    cut_off = 0.4

    if subset["label"].iloc[0] == "squat":
        cut_off = 0.4

    if subset["label"].iloc[0] == "row":
        cut_off = 0.65
        col = "gyr_r"

    if subset["label"].iloc[0] == "ohp":
        cut_off = 0.35

    rep_df.loc[(rep_df["set"] == s), "predicted_reps"] = count_repetitions(
        subset, col, cut_off
    )


mae = round(mean_absolute_error(rep_df["actual_reps"], rep_df["predicted_reps"]), 2)
rep_df.groupby(["label", "category"])[
    ["actual_reps", "predicted_reps"]
].mean().plot.bar()
