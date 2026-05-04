import os
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# Settings
# --------------------------------------------------

CSV_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "Results",
    "Drugs_No_Growth",
    "Evaluation_Results_logbook_q_table_drug_reward_5_no_growth_TIME_04_05_2026_17-49-32.csv"
)

METRICS = [
    "Food_Consumed",
    "Drugs_Consumed",
    "Total_Reward",
    "Snake_Length"
]


# --------------------------------------------------
# Load Data
# --------------------------------------------------

df = pd.read_csv(CSV_PATH)

print("Loaded evaluation results:")
print(CSV_PATH)
print()
print(df.head())
print()


# --------------------------------------------------
# Print Summary Statistics
# --------------------------------------------------

print("Summary statistics:")
print(df[METRICS].describe())
print()


# --------------------------------------------------
# Print Loop Information
# --------------------------------------------------

if "Loop" in df.columns:
    number_of_loops = df["Loop"].sum()
    total_episodes = len(df)
    loop_percentage = number_of_loops / total_episodes * 100

    print("Loop information:")
    print(f"Number of evaluation episodes with loops: {number_of_loops}")
    print(f"Total evaluation episodes: {total_episodes}")
    print(f"Percentage with loops: {loop_percentage:.2f}%")
    print()
else:
    print("No 'Loop' column found in the CSV.")
    print()


# --------------------------------------------------
# Figure 1: Boxplots For Each Metric
# --------------------------------------------------

plt.figure(figsize=(9, 5))

plt.boxplot(
    [df[metric] for metric in METRICS],
    labels=METRICS,
    showmeans=True
)

plt.title("Baseline Evaluation: Metric Distributions")
plt.ylabel("Value")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# Figure 2: Line Plots Across Evaluation Episodes
# --------------------------------------------------

fig, axes = plt.subplots(
    nrows=1,
    ncols=len(METRICS),
    figsize=(5 * len(METRICS), 4),
    sharex=True
)

for ax, metric in zip(axes, METRICS):
    ax.plot(df["Episode"], df[metric], marker="o", linewidth=1.5, alpha=0.8)
    ax.axhline(df[metric].mean(), color="red", linestyle="--", label=f"Mean = {df[metric].mean():.2f}")

    ax.set_title(metric)
    ax.set_xlabel("Evaluation Episode")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.suptitle("Baseline Evaluation Across Episodes")
plt.tight_layout()
plt.show()


# --------------------------------------------------
# Figure 3: Histograms For Each Metric
# --------------------------------------------------

fig, axes = plt.subplots(
    nrows=1,
    ncols=len(METRICS),
    figsize=(5 * len(METRICS), 4)
)

for ax, metric in zip(axes, METRICS):
    ax.hist(df[metric], bins=10, alpha=0.75, edgecolor="black")
    ax.axvline(df[metric].mean(), color="red", linestyle="--", label=f"Mean = {df[metric].mean():.2f}")

    ax.set_title(metric)
    ax.set_xlabel(metric)
    ax.set_ylabel("Frequency")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

plt.suptitle("Baseline Evaluation: Metric Histograms")
plt.tight_layout()
plt.show()


# --------------------------------------------------
# Figure 4: Loop Counts
# --------------------------------------------------

if "Loop" in df.columns:
    loop_counts = df["Loop"].value_counts().sort_index()

    labels = ["No loop", "Loop"]
    values = [
        loop_counts.get(0, 0),
        loop_counts.get(1, 0),
    ]

    plt.figure(figsize=(5, 4))
    plt.bar(labels, values, color=["steelblue", "darkorange"])

    plt.title("Number of Loops in Evaluation Set")
    plt.ylabel("Number of Episodes")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()