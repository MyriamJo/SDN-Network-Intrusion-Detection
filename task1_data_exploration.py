import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

# Configuration
DATA_PATH = r"C:\Users\LENOVO\Desktop\data.csv"
OUTPUT_DIR = "figures"
SHOW_PLOTS = False                     
CATEGORY_THRESHOLD = 20                
QUARTER_COUNT = 4

ATTACK_TYPE_FOR_CONDITIONAL = "TCP-SYN"         
JOINT_FIELD_PAIR = ("Received Packets", "Sent Packets")  
JOINT_ATTACK_TYPE = "TCP-SYN"                   

ATTACK_LABELS = ["TCP-SYN", "PortScan", "Overflow", "Normal", "Diversion", "Blackhole"]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def safe_name(name):
    """Turns a field name into something safe to use in a filename."""
    return name.strip().replace("/", "-").replace(" ", "_").replace("(", "").replace(")", "")


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# Load
df = pd.read_csv(DATA_PATH)
field_names = list(df.columns)

# Field inventory
for name in field_names:
    print(f" - {name}")

# Field data types
print(df.dtypes)


# Data quality check
numeric_df_all = df.select_dtypes(include=[np.number])
missing_counts = df.isnull().sum()
inf_counts = numeric_df_all.isin([np.inf, -np.inf]).sum()
print("Missing values per column:")
print(missing_counts)
print("\nInfinite values per numeric column:")
print(inf_counts)
if missing_counts.sum() == 0 and inf_counts.sum() == 0:
    print("\n=> No missing or infinite values found anywhere in the dataset.")
else:
    print("\n=> Missing/infinite values were found - see the counts above.")

# Cardinality per field
category_counts = {name: int(df[name].nunique()) for name in field_names}
for name, count in category_counts.items():
    print(f" - {name}: {count} distinct values")


numeric_fields = [
    name for name in field_names
    if category_counts[name] > CATEGORY_THRESHOLD and pd.api.types.is_numeric_dtype(df[name])
]
categorical_fields = [name for name in field_names if name not in numeric_fields]


# Summary statistics for numeric fields
stats_summary = pd.DataFrame({
    "max": df[numeric_fields].max(),
    "min": df[numeric_fields].min(),
    "mean": df[numeric_fields].mean(),
    "variance": df[numeric_fields].var(),
})
print(stats_summary)


quarter_stats = {}
for name in numeric_fields:
    col = df[name]
    edges = np.linspace(col.min(), col.max(), QUARTER_COUNT + 1)
    rows = []
    for q in range(QUARTER_COUNT):
        lo, hi = edges[q], edges[q + 1]
        if q < QUARTER_COUNT - 1:
            bucket = col[(col >= lo) & (col < hi)]
        else:
            bucket = col[(col >= lo) & (col <= hi)]  # last bucket is inclusive on both ends
        rows.append({
            "quarter": f"Q{q + 1} [{lo:.2f}, {hi:.2f})",
            "count": len(bucket),
            "max": bucket.max() if len(bucket) else np.nan,
            "min": bucket.min() if len(bucket) else np.nan,
            "mean": bucket.mean() if len(bucket) else np.nan,
            "variance": bucket.var() if len(bucket) else np.nan,
        })
    quarter_df = pd.DataFrame(rows)
    quarter_stats[name] = quarter_df
    print(f"\n-- {name} --")
    print(quarter_df.to_string(index=False))


for label in ATTACK_LABELS:
    df[label] = (df["Label"] == label).astype(int)
print(f"Added columns: {ATTACK_LABELS}")


def field_pmf(series):
    """PMF for a categorical/discrete field: value -> probability."""
    counts = series.value_counts().sort_index()
    return counts / counts.sum()


def field_pdf(series, bins=200):
    """Histogram-based PDF for a continuous field. Returns (bin centers, density)."""
    clean = series.dropna()
    counts, edges = np.histogram(clean, bins=bins, density=False)
    width = edges[1] - edges[0]
    pdf = counts / (counts.sum() * width) if counts.sum() > 0 and width > 0 else counts.astype(float)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return centers, pdf


def cdf_from_pmf(pmf):
    return pmf.cumsum()


def cdf_from_pdf(pdf_values, width):
    return np.cumsum(pdf_values) * width


field_distributions = {}  

for name in field_names:
    fig, axes = plt.subplots(2, 1, figsize=(7, 6))
    if name in categorical_fields:
        pmf = field_pmf(df[name])
        cdf = cdf_from_pmf(pmf)
        x_labels = pmf.index.astype(str)
        axes[0].stem(x_labels, pmf.values)
        axes[0].set_title(f"{name} - PMF")
        axes[0].tick_params(axis="x", rotation=90)
        axes[1].step(x_labels, cdf.values, where="post")
        axes[1].set_title(f"{name} - CDF")
        axes[1].tick_params(axis="x", rotation=90)
        field_distributions[name] = ("pmf", pmf)
    else:
        centers, pdf = field_pdf(df[name])
        width = centers[1] - centers[0] if len(centers) > 1 else 1.0
        cdf = cdf_from_pdf(pdf, width)
        axes[0].plot(centers, pdf)
        axes[0].set_title(f"{name} - PDF")
        axes[1].plot(centers, cdf)
        axes[1].set_title(f"{name} - CDF")
        field_distributions[name] = ("pdf", (centers, pdf))
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"dist_{safe_name(name)}.png"))
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


attack_subset = df[df["Label"] == ATTACK_TYPE_FOR_CONDITIONAL]

for name in field_names:
    fig, ax = plt.subplots(figsize=(7, 4))
    kind, original = field_distributions[name]
    if kind == "pmf":
        cond_pmf = field_pmf(attack_subset[name])
        ax.stem(original.index.astype(str), original.values, linefmt="b-", markerfmt="bo", basefmt=" ", label="Overall")
        ax.stem(cond_pmf.index.astype(str), cond_pmf.values, linefmt="r-", markerfmt="ro", basefmt=" ", label=f"Given {ATTACK_TYPE_FOR_CONDITIONAL}")
        ax.tick_params(axis="x", rotation=90)
    else:
        centers, pdf = original
        cond_centers, cond_pdf = field_pdf(attack_subset[name])
        ax.plot(centers, pdf, color="blue", label="Overall")
        ax.plot(cond_centers, cond_pdf, color="red", label=f"Given {ATTACK_TYPE_FOR_CONDITIONAL}")
    ax.set_title(f"{name}: overall vs. given {ATTACK_TYPE_FOR_CONDITIONAL}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"conditional_{safe_name(name)}.png"))
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


fig, ax = plt.subplots(figsize=(6, 6))
df.plot.scatter(x=JOINT_FIELD_PAIR[0], y=JOINT_FIELD_PAIR[1], s=2, ax=ax)
fig.savefig(os.path.join(OUTPUT_DIR, "scatter_pair.png"))
if SHOW_PLOTS:
    plt.show()
plt.close(fig)


# Joint distribution of the two fields
f1, f2 = JOINT_FIELD_PAIR
both_categorical = f1 in categorical_fields and f2 in categorical_fields

if both_categorical:
    joint_pmf = df.value_counts([f1, f2], normalize=True)
    print(joint_pmf.head(20))
else:
    H, xedges, yedges = np.histogram2d(df[f1], df[f2], bins=(50, 50))
    width1 = xedges[1] - xedges[0]
    width2 = yedges[1] - yedges[0]
    joint_pdf = H / (H.sum() * width1 * width2) if H.sum() > 0 else H
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(joint_pdf.T, origin="lower", aspect="auto",
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    fig.colorbar(im, ax=ax, label="joint density")
    ax.set_xlabel(f1)
    ax.set_ylabel(f2)
    ax.set_title(f"Joint PDF: {f1} vs {f2}")
    fig.savefig(os.path.join(OUTPUT_DIR, "joint_pdf.png"))
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)

data_given_attack = df[df["Label"] == JOINT_ATTACK_TYPE]

if both_categorical:
    joint_pmf_given = data_given_attack.value_counts([f1, f2], normalize=True)
    diff = joint_pmf.subtract(joint_pmf_given, fill_value=0)
    print(diff.abs().sort_values(ascending=False).head(10))
else:
    H_given, _, _ = np.histogram2d(data_given_attack[f1], data_given_attack[f2], bins=(xedges, yedges))
    joint_pdf_given = H_given / (H_given.sum() * width1 * width2) if H_given.sum() > 0 else H_given
    diff = joint_pdf - joint_pdf_given

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    panels = [(joint_pdf, "Overall joint PDF"),
              (joint_pdf_given, f"Given {JOINT_ATTACK_TYPE}"),
              (diff, "Difference (overall - conditional)")]
    for ax, (mat, title) in zip(axes, panels):
        im = ax.imshow(mat.T, origin="lower", aspect="auto",
                        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        ax.set_title(title)
        ax.set_xlabel(f1)
        ax.set_ylabel(f2)
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "joint_pdf_conditional_diff.png"))
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)

corr = df[numeric_fields].corr()
print(corr)

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(numeric_fields)))
ax.set_xticklabels(numeric_fields, rotation=90)
ax.set_yticks(range(len(numeric_fields)))
ax.set_yticklabels(numeric_fields)
fig.colorbar(im, ax=ax, label="correlation")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"))
if SHOW_PLOTS:
    plt.show()
plt.close(fig)

groups_by_label = {label: df[df["Label"] == label] for label in ATTACK_LABELS}
dependency_results = []

for name in field_names:
    if name in ("Label", "Binary Label"):
        continue  # these define the attack type itself, not useful to test against it
    if name in numeric_fields:
        samples = [groups_by_label[label][name].values for label in ATTACK_LABELS]
        try:
            stat, p_value = scipy_stats.kruskal(*samples)
        except ValueError:
            stat, p_value = np.nan, np.nan
        test_used = "Kruskal-Wallis"
    else:
        contingency = pd.crosstab(df[name], df["Label"])
        try:
            stat, p_value, _, _ = scipy_stats.chi2_contingency(contingency)
        except ValueError:
            stat, p_value = np.nan, np.nan
        test_used = "Chi-square"
    dependency_results.append({
        "field": name,
        "test": test_used,
        "statistic": stat,
        "p_value": p_value,
        "depends_on_attack_type": bool(p_value < 0.05) if pd.notna(p_value) else None,
    })

dependency_df = pd.DataFrame(dependency_results).sort_values("p_value", na_position="last")
print(dependency_df.to_string(index=False))
dependency_df.to_csv(os.path.join(OUTPUT_DIR, "dependency_results.csv"), index=False)
