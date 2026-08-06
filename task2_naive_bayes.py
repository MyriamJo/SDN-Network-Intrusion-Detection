import os

import numpy as np
import pandas as pd

# Configuration
PICKLE_DIR = "pickles"
EPSILON_PDF = 1e-300   
LAPLACE_SMOOTHING = 1  

ATTACK_LABELS = ["TCP-SYN", "PortScan", "Overflow", "Normal", "Diversion", "Blackhole"]


def pickle_path(name):
    return os.path.join(PICKLE_DIR, name)


# Load the train/test split and the fitted distributions for every subset
train = pd.read_pickle(pickle_path("Train"))
test = pd.read_pickle(pickle_path("Test"))

subset_names = ["Original"] + ATTACK_LABELS
fitted = {name: pd.read_pickle(pickle_path(name)) for name in subset_names}

numeric_fields = fitted["Original"]["numeric_fields"]
categorical_fields = fitted["Original"]["categorical_fields"]
vocabularies = fitted["Original"]["vocabularies"]

print(f"Numeric fields used ({len(numeric_fields)}): {numeric_fields}")
print(f"Categorical fields used ({len(categorical_fields)}): {categorical_fields}")


# Per-row log-likelihoods. Working in log space turns the long product in
# the Naive Bayes equation into a sum, which is both faster and avoids
# underflow from multiplying together dozens of small probabilities.

def numeric_log_likelihood(test_df, numeric_fits):
    """(n_rows, n_fields) matrix of log pdf values, one column per numeric field."""
    n = len(test_df)
    log_lik = np.zeros((n, len(numeric_fields)))
    for j, field in enumerate(numeric_fields):
        if field not in numeric_fits.index:
            continue  # this subset had too few rows to fit this field - treat as uninformative (log 0 = contributes nothing)
        dist = numeric_fits.loc[field, "dist"]
        params = numeric_fits.loc[field, "params"]
        arg, loc, scale = params[:-2], params[-2], params[-1]
        x = test_df[field].to_numpy(dtype=float)
        pdf_vals = dist.pdf(x, loc=loc, scale=scale, *arg)
        pdf_vals = np.nan_to_num(pdf_vals, nan=EPSILON_PDF, posinf=EPSILON_PDF, neginf=EPSILON_PDF)
        pdf_vals = np.clip(pdf_vals, EPSILON_PDF, None)
        log_lik[:, j] = np.log(pdf_vals)
    return log_lik


def categorical_log_likelihood(test_df, categorical_counts, subset_size):
    """(n_rows, n_fields) matrix of log pmf values, one column per categorical
    field, using Laplace (add-one) smoothing so a category never seen in this
    particular subset still gets a small, well-defined probability."""
    n = len(test_df)
    log_lik = np.zeros((n, len(categorical_fields)))
    for j, field in enumerate(categorical_fields):
        counts = categorical_counts.get(field)
        lookup = counts.to_dict() if counts is not None else {}
        vocab_size = max(len(vocabularies.get(field, [])), 1)
        denom = subset_size + LAPLACE_SMOOTHING * vocab_size
        probs = test_df[field].map(lambda v: (lookup.get(v, 0) + LAPLACE_SMOOTHING) / denom)
        log_lik[:, j] = np.log(probs.to_numpy(dtype=float))
    return log_lik


def row_log_likelihood(payload, test_df):
    return (
        numeric_log_likelihood(test_df, payload["numeric_fits"]).sum(axis=1)
        + categorical_log_likelihood(test_df, payload["categorical_counts"], payload["subset_size"]).sum(axis=1)
    )


# Marginal (unconditional) log-likelihood of every test row, using the
# "Original" (whole training set) fits - this is Pr(row), the denominator
# shared by every class in the Naive Bayes equation.

marginal_log_lik = row_log_likelihood(fitted["Original"], test)


# Naive Bayes score for every attack type, for every test row:
#    Pr(attack | row) = [ prod(pdf_i|attack(x_i)) * prod(pmf_j|attack(x_j)) * Pr(attack) ] / Pr(row)

scores = pd.DataFrame(index=test.index)
for label in ATTACK_LABELS:
    payload = fitted[label]
    conditional_log_lik = row_log_likelihood(payload, test)
    prior = payload["subset_size"] / len(train)
    log_score = conditional_log_lik + np.log(prior) - marginal_log_lik
    scores[label] = np.exp(log_score)

predicted_label = scores.idxmax(axis=1).reset_index(drop=True)
actual_label = test["Label"].reset_index(drop=True)

overall_accuracy = (predicted_label == actual_label).mean()
print(f"\nOverall multi-class accuracy: {overall_accuracy:.4f} "
      f"({(predicted_label == actual_label).sum()}/{len(actual_label)} correct)")

report_rows = []
for label in ATTACK_LABELS:
    predicted_positive = predicted_label == label
    actual_positive = actual_label == label

    true_positive = int((predicted_positive & actual_positive).sum())
    true_negative = int((~predicted_positive & ~actual_positive).sum())
    false_positive = int((predicted_positive & ~actual_positive).sum())
    false_negative = int((~predicted_positive & actual_positive).sum())

    accuracy = (true_positive + true_negative) / len(actual_label)
    false_positive_rate = false_positive / max(int((~actual_positive).sum()), 1)
    false_negative_rate = false_negative / max(int(actual_positive.sum()), 1)

    report_rows.append({
        "attack_type": label,
        "accuracy": round(accuracy, 4),
        "false_positive_rate": round(false_positive_rate, 4),
        "false_negative_rate": round(false_negative_rate, 4),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
    })

report_df = pd.DataFrame(report_rows)
print("\nPer attack-type results (one-vs-rest):")
print(report_df.to_string(index=False))

os.makedirs(PICKLE_DIR, exist_ok=True)
report_df.to_csv(os.path.join(PICKLE_DIR, "naive_bayes_results.csv"), index=False)
print(f"\nSaved full results to ./{PICKLE_DIR}/naive_bayes_results.csv")
