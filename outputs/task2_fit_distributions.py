import os
import warnings
import numpy as np
import pandas as pd
import scipy.stats as st

try:
    from scipy.stats._continuous_distns import _distn_names
except ImportError:
    _distn_names = [name for name in dir(st) if isinstance(getattr(st, name), st.rv_continuous)]

# Configuration 
DATA_PATH = r"C:\Users\LENOVO\Desktop\data.csv"
PICKLE_DIR = "pickles"
CATEGORY_THRESHOLD = 20   
RANDOM_STATE = 42         
HISTOGRAM_BINS = 250
MAX_FIT_SAMPLE_SIZE = 5000  

EXCLUDE_COLUMNS = []

ATTACK_LABELS = ["TCP-SYN", "PortScan", "Overflow", "Normal", "Diversion", "Blackhole"]

os.makedirs(PICKLE_DIR, exist_ok=True)


def pickle_path(name):
    return os.path.join(PICKLE_DIR, name)


# Split into 80% train / 20% test
df = pd.read_csv(DATA_PATH)
df_train = df.sample(frac=0.8, random_state=RANDOM_STATE)
df_test = df.drop(df_train.index)

pd.to_pickle(df_train, pickle_path("Train"))
pd.to_pickle(df_test, pickle_path("Test"))
print(f"Train: {len(df_train)} rows, Test: {len(df_test)} rows")

field_names = list(df_train.columns)

category_counts = {name: int(df_train[name].nunique()) for name in field_names}
numeric_fields = [
    name for name in field_names
    if name not in EXCLUDE_COLUMNS
    and category_counts[name] > CATEGORY_THRESHOLD
    and pd.api.types.is_numeric_dtype(df_train[name])
]
categorical_fields = [
    name for name in field_names
    if name not in EXCLUDE_COLUMNS and name not in numeric_fields and name != "Label"
]
print(f"Numeric fields ({len(numeric_fields)}): {numeric_fields}")
print(f"Categorical fields ({len(categorical_fields)}): {categorical_fields}")

for label in ATTACK_LABELS:
    df_train[label] = (df_train["Label"] == label).astype(int)

subsets = {"Original": df_train}
for label in ATTACK_LABELS:
    subsets[label] = df_train[df_train[label] == 1]
    print(f"  subset '{label}': {len(subsets[label])} rows")


# Best-fit continuous distribution search: histogram the data, then fit
# every candidate distribution in scipy.stats and rank by sum-of-squared
# error against that histogram

def best_fit_distribution(data, bins=HISTOGRAM_BINS):
    """Finds the continuous scipy.stats distribution whose PDF best matches
    data's histogram, ranked by sum-of-squared error. Returns a list of
    (distribution, params, sse) tuples, best fit first."""
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    results = []
    for distribution_name in [d for d in _distn_names if d not in ("levy_stable", "studentized_range")]:
        distribution = getattr(st, distribution_name)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                params = distribution.fit(data)
                arg, loc, scale = params[:-2], params[-2], params[-1]
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))
                if np.isfinite(sse):
                    results.append((distribution, params, sse))
        except Exception:
            continue

    return sorted(results, key=lambda item: item[2])


def sampled(data):
    """Caps the number of rows used to fit a distribution - see MAX_FIT_SAMPLE_SIZE."""
    if MAX_FIT_SAMPLE_SIZE is not None and len(data) > MAX_FIT_SAMPLE_SIZE:
        rng = np.random.default_rng(RANDOM_STATE)
        return rng.choice(data, size=MAX_FIT_SAMPLE_SIZE, replace=False)
    return data


numeric_fits = {}
for subset_name, subset_df in subsets.items():
    print(f"\nFitting numeric columns for subset: {subset_name} ({len(subset_df)} rows)")
    fits = {}
    for field in numeric_fields:
        data = np.asarray(subset_df[field], dtype=float)
        if len(data) < 2:
            print(f"  {field}: skipped (fewer than 2 rows in this subset)")
            continue
        candidates = best_fit_distribution(sampled(data))
        if not candidates:
            print(f"  {field}: no distribution could be fit - skipped")
            continue
        best_dist, best_params, best_sse = candidates[0]
        fits[field] = {
            "dist_name": best_dist.name,
            "dist": best_dist,
            "params": best_params,
            "sse": best_sse,
        }
        print(f"  {field}: best fit = {best_dist.name}  (sse={best_sse:.6g})")
    numeric_fits[subset_name] = pd.DataFrame(fits).T

# Categorical PMF table for every subset.

categorical_counts = {}
for subset_name, subset_df in subsets.items():
    counts_by_field = {}
    for field in categorical_fields:
        counts_by_field[field] = subset_df[field].value_counts()
    categorical_counts[subset_name] = counts_by_field

vocabularies = {field: sorted(df_train[field].dropna().unique().tolist(), key=str) for field in categorical_fields}

for subset_name in subsets:
    payload = {
        "numeric_fits": numeric_fits[subset_name],          # DataFrame indexed by field name: dist_name, dist, params, sse
        "categorical_counts": categorical_counts[subset_name],  # dict: field name -> pandas Series (value -> raw count)
        "subset_size": len(subsets[subset_name]),
        "numeric_fields": numeric_fields,
        "categorical_fields": categorical_fields,
        "vocabularies": vocabularies,
    }
    pd.to_pickle(payload, pickle_path(subset_name))

print(f"\nSaved fitted distributions and categorical counts for {list(subsets.keys())} to ./{PICKLE_DIR}/")
