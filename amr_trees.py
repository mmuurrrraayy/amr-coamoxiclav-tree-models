"""E. coli co-amoxiclav resistance: tree model comparison.

Questions:
- Which tree best identifies those at risk of antibiotic resistance?
- What are the main risk factors?

Data: dig_health_hub_amr.csv (synthetic, modelled on UKHSA data). Columns used:
coamox (target, 1 = resistant), age, imd (1 = most deprived, 5 = least).

Usage:  python amr_trees.py [--data dig_health_hub_amr.csv] [--output Outputs]
"""
import argparse
import os
import stat
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier,
                              RandomForestClassifier, VotingClassifier)
from sklearn.inspection import permutation_importance
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.utils.class_weight import compute_sample_weight


'''
Configuration
'''
DEFAULT_DATA_PATH = Path("dig_health_hub_amr.csv")
DEFAULT_OUTPUT_DIR = Path("Outputs")
RANDOM_STATE = 42

TARGET = "coamox"

# Plausible value ranges used in cleaning
AGE_MIN, AGE_MAX = 0, 120
IMD_VALUES = {1, 2, 3, 4, 5}
TARGET_VALUES = {0, 1}

TEST_SIZE = 0.25

# raw_age: age in years; binned_age: decade band (e.g. 40-49)
FEATURE_SETS = {
    "raw_age": ["age", "imd"],
    "binned_age": ["age_band", "imd"],
}
FEATURES = FEATURE_SETS["raw_age"]  # columns read from the CSV and cleaned

# Model grid searched with cross-validation on the training set
DEPTHS = [1, 2, 3, 4, 5, 6]
WEIGHTINGS = ["unweighted", "balanced"]
N_ESTIMATORS = 100
CV_FOLDS = 5
METRICS = ["accuracy", "recall", "precision", "roc_auc"]
SELECTION_METRIC = "roc_auc"  #how well a model ranks patients by risk

VOTING_TOP_N = 3
N_PERM_REPEATS = 50
EXTRA_FEATURES = ["sex_male", "had_surgery_past_yr"] 

# Grouped by patient so repeat tests from one patient never sit on both sides of a split
CV = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

'''
Models
'''
def tree(depth):
    return DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)


MODEL_BUILDERS = {
    "decision_tree": tree,
    "bagging": lambda d: BaggingClassifier(tree(d), n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE),
    "random_forest": lambda d: RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=d, random_state=RANDOM_STATE),
    "adaboost": lambda d: AdaBoostClassifier(tree(d), n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE),
    "gradient_boosting": lambda d: GradientBoostingClassifier(n_estimators=N_ESTIMATORS, max_depth=d, random_state=RANDOM_STATE),
}


def fit_model(model, X, y, weighting):
    # Balancing via sample_weight works for every model type (gradient boosting has no class_weight)
    sample_weight = compute_sample_weight("balanced", y) if weighting == "balanced" else None
    return model.fit(X, y, sample_weight=sample_weight)


def score(y_true, prob):
    pred = (prob >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "recall": recall_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, prob),
    }


def model_id(model_type, depth, weighting, feature_set):
    # Used in output file names such as random_forest_depth3_balanced_raw_age
    return f"{model_type}_depth{depth}_{weighting}_{feature_set}"


def cv_scores(build, X, y, groups, weighting):
    # Mean and sd of each metric over the shared stratified, patient-grouped CV folds
    folds = pd.DataFrame([
        score(y.iloc[va], fit_model(build(), X.iloc[tr], y.iloc[tr], weighting).predict_proba(X.iloc[va])[:, 1])
        for tr, va in CV.split(X, y, groups)])
    return folds.mean(), folds.std()


def test_row(cfg, y_test, prob):
    # One row of the test-results table
    test_scores = score(y_test, prob)
    tn, fp, fn, tp = confusion_matrix(y_test, (prob >= 0.5).astype(int)).ravel()
    row = {"model_id": cfg["model_id"], "model_type": cfg["model_type"],
           "depth": cfg["depth"], "feature_set": cfg["feature_set"]}
    for m in METRICS:
        row[f"cv_{m}"] = cfg[f"{m}_mean"]
        row[f"test_{m}"] = test_scores[m]
    row.update({"test_resistant_flagged": tp, "test_resistant_missed": fn,
                "test_false_alarms": fp, "test_correctly_cleared": tn})
    return row


def save_fig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


'''
Setup
'''
def check_data_permissions(path):
    
    if os.name != "posix" or not path.exists():  # a missing file is reported by load_data
        return
    mode = path.stat().st_mode
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        print(f"WARNING: {path} is accessible to other users ({stat.filemode(mode)}). "
              f"Restrict it with: chmod 600 {path}")
    else:
        print(f"Data file permissions OK: {stat.filemode(mode)}")


def make_dirs(output_dir):
    dirs = {
        "data": output_dir / "01_data",
        "trees": output_dir / "02_trees",
        "metrics": output_dir / "03_metrics",
        "importance": output_dir / "04_importance",
        "extension": output_dir / "05_extension",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


'''Load, Clean and Validate'''
def load_data(path):
    if not path.exists():
        sys.exit(f"{path} not found in {Path.cwd()}. Download it with the curl command in the README.")
    raw = pd.read_csv(path)
    print(f"Loaded {len(raw):,} rows x {raw.shape[1]} columns from {path}")

    missing_cols = [c for c in [TARGET] + FEATURES if c not in raw.columns]
    if missing_cols:
        raise KeyError(f"Expected columns missing from the data: {missing_cols}")
    return raw


def clean(raw):
    # Each rule is applied in turn and the number of rows it removes is logged
    df = raw[[TARGET] + FEATURES].copy()
    log = [("Raw data", 0, len(df))]  # (rule, rows removed, rows remaining)

    def apply_rule(frame, keep_mask, rule):
        frame = frame[keep_mask]
        log.append((rule, int((~keep_mask).sum()), len(frame)))
        return frame

    for col in [TARGET] + FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = apply_rule(df, df.notna().all(axis=1), "Missing or non-numeric value in coamox/age/imd")
    df = apply_rule(df, df[TARGET].isin(TARGET_VALUES), f"coamox not in {sorted(TARGET_VALUES)}")
    df = apply_rule(df, df["age"].between(AGE_MIN, AGE_MAX), f"age outside {AGE_MIN}-{AGE_MAX}")
    df = apply_rule(df, df["imd"].isin(IMD_VALUES), f"imd not in {sorted(IMD_VALUES)}")

    df[TARGET] = df[TARGET].astype(int)
    df["imd"] = df["imd"].astype(int)
    raw_index = df.index.to_numpy()  # raw row of each cleaned row, used by the extension
    df = df.reset_index(drop=True)

    assert len(df) > 0, "No rows left after cleaning"
    assert df.notna().all().all(), "Missing values remain"
    assert set(df[TARGET].unique()) == TARGET_VALUES, "Target does not contain both classes"
    assert df["age"].between(AGE_MIN, AGE_MAX).all()
    assert df["imd"].isin(IMD_VALUES).all()

    return df, raw_index, pd.DataFrame(log, columns=["rule", "rows_removed", "rows_remaining"])


def class_stats(df):
    # Class counts, class % and resistance rate by IMD, shared by the report and the overview figure
    class_counts = df[TARGET].value_counts().sort_index()
    class_pct = (class_counts / len(df) * 100).round(1)
    imd_rate = df.groupby("imd")[TARGET].mean()
    return class_counts, class_pct, imd_rate


def save_cleaning_outputs(raw, df, log_df, data_path, data_dir):
    class_counts, class_pct, imd_rate = class_stats(df)
    n_dupes = int(df.duplicated().sum())
    if "id" in raw.columns:  # repeat-test check
        n_ids = raw["id"].nunique()
        id_line = (f"Unique patient ids: {n_ids:,} in {len(raw):,} rows"
                   + (" (one test per patient)" if n_ids == len(raw) else " (REPEAT TESTS PRESENT)"))
    else:
        id_line = "Unique patient ids: no 'id' column, repeat tests not checked"

    print(log_df.to_string(index=False))
    print(f"Rows kept: {len(df):,} of {len(raw):,} ({len(df) / len(raw):.1%})")
    print(f"Class balance: not resistant {class_counts[0]:,} ({class_pct[0]}%), "
          f"resistant {class_counts[1]:,} ({class_pct[1]}%)")
    print(id_line)

    with open(data_dir / "cleaning_report.txt", "w") as f:
        f.write("CLEANING REPORT\n")
        f.write(f"Source file: {data_path}\n")
        f.write(f"All columns in source: {list(raw.columns)}\n\n")
        f.write(log_df.to_string(index=False))
        f.write(f"\n\nRows kept: {len(df):,} of {len(raw):,} ({len(df) / len(raw):.1%})\n\n")
        f.write("Class balance:\n")
        f.write(f"  not resistant (0): {class_counts[0]:,} ({class_pct[0]}%)\n")
        f.write(f"  resistant (1):     {class_counts[1]:,} ({class_pct[1]}%)\n\n")
        f.write(f"Identical rows (reported, not removed): {n_dupes:,}\n")
        f.write(f"{id_line}\n\n")
        f.write("Summary statistics:\n")
        f.write(df.describe().to_string())
        f.write("\n\nResistance rate by IMD quintile:\n")
        f.write(imd_rate.round(3).to_string())

    log_df.to_csv(data_dir / "cleaning_log.csv", index=False)
    df.to_csv(data_dir / "cleaned_data.csv", index=False)


def plot_data_overview(df, path):
    # Class balance and feature distributions
    class_counts, class_pct, imd_rate = class_stats(df)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].bar(["Not resistant (0)", "Resistant (1)"], class_counts.values, color=["#4C78A8", "#E45756"])
    axes[0].set_title("Class balance: coamox")
    axes[0].set_ylabel("Count")
    axes[0].set_ylim(0, class_counts.max() * 1.2)
    for i, (n, p) in enumerate(zip(class_counts.values, class_pct.values)):
        axes[0].text(i, n, f"{n:,}\n({p}%)", ha="center", va="bottom")

    axes[1].hist([df.loc[df[TARGET] == 0, "age"], df.loc[df[TARGET] == 1, "age"]],
                 bins=range(int(df["age"].min()) // 5 * 5, int(df["age"].max()) + 6, 5), stacked=True,
                 label=["Not resistant", "Resistant"], color=["#4C78A8", "#E45756"])
    axes[1].set_title("Age distribution by resistance")
    axes[1].set_xlabel("Age (years)")
    axes[1].legend()

    axes[2].bar(imd_rate.index.astype(str), imd_rate.values * 100, color="#4C78A8")
    axes[2].set_title("Resistance rate by IMD quintile")
    axes[2].set_xlabel("IMD quintile (1 = most deprived)")
    axes[2].set_ylabel("% resistant")

    fig.tight_layout()
    save_fig(fig, path)


'''
Feature variants and test/train split
'''
def add_features_and_split(df, patient_ids, data_dir):
    df["age_band"] = (df["age"] // 10 * 10).astype(int)
    # Random number per patient used to check that importance scores beat pure noise
    df["noise"] = np.random.default_rng(RANDOM_STATE).random(len(df))
    df["patient_id"] = patient_ids  # grouping key only, never a model feature

    #test set is used once at the end; one fold of a grouped split, so no patient is in both sets
    splitter = StratifiedGroupKFold(n_splits=round(1 / TEST_SIZE), shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(df, df[TARGET], df["patient_id"]))
    train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]

    split_summary = pd.DataFrame({
        "rows": [len(train_df), len(test_df)],
        "resistant": [int(train_df[TARGET].sum()), int(test_df[TARGET].sum())],
        "resistant_pct": [round(train_df[TARGET].mean() * 100, 1), round(test_df[TARGET].mean() * 100, 1)],
    }, index=["train", "test"])
    split_summary.to_csv(data_dir / "train_test_split_summary.csv")
    print(split_summary.to_string())

    band_rate = df.groupby("age_band")[TARGET].agg(patients="size", pct_resistant="mean")
    band_rate["pct_resistant"] = (band_rate["pct_resistant"] * 100).round(1)
    print(f"Resistance by age band:\n{band_rate.to_string()}")
    return train_df, test_df

'''
Comparison
'''
def run_cv(train_df):
    y_train = train_df[TARGET]
    rows = []
    start = time.time()
    for feature_set, cols in FEATURE_SETS.items():
        X = train_df[cols]
        for model_type, build in MODEL_BUILDERS.items():
            for depth in DEPTHS:
                for weighting in WEIGHTINGS:
                    mean, std = cv_scores(lambda: build(depth), X, y_train, train_df["patient_id"], weighting)
                    row = {"model_id": model_id(model_type, depth, weighting, feature_set),
                           "model_type": model_type, "depth": depth,
                           "weighting": weighting, "feature_set": feature_set}
                    for m in METRICS:
                        row[f"{m}_mean"] = mean[m]
                        row[f"{m}_std"] = std[m]
                    rows.append(row)
            print(f"{feature_set:<11} {model_type:<18} done ({time.time() - start:.0f}s)", flush=True)

    cv_results = (pd.DataFrame(rows)
                  .sort_values(f"{SELECTION_METRIC}_mean", ascending=False)
                  .reset_index(drop=True))
    print(f"{len(cv_results)} configurations evaluated")
    return cv_results


def plot_metric_by_depth(cv_results, metric, metrics_dir):
    # One panel per weighting x feature set, one line per model type
    fig, axes = plt.subplots(len(WEIGHTINGS), len(FEATURE_SETS), figsize=(12, 8), sharex=True, sharey=True)
    for i, weighting in enumerate(WEIGHTINGS):
        for j, feature_set in enumerate(FEATURE_SETS):
            ax = axes[i, j]
            panel = cv_results[(cv_results["weighting"] == weighting) & (cv_results["feature_set"] == feature_set)]
            for model_type in MODEL_BUILDERS:
                line = panel[panel["model_type"] == model_type].sort_values("depth")
                ax.plot(line["depth"], line[f"{metric}_mean"], marker="o", label=model_type)
            ax.set_title(f"{feature_set} | {weighting}")
            ax.set_xlabel("Max depth")
            ax.set_ylabel(f"CV {metric} (mean)")
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Cross-validated {metric} by model type and depth (training set)")
    fig.tight_layout()
    save_fig(fig, metrics_dir / f"cv_{metric}_by_depth.png")


def select_best(cv_results, metrics_dir):
    """Best balanced configuration of each model type, per age version (5 x 2 = 10 models).

    Balanced only: weighting barely changes ROC-AUC, but unweighted models rarely predict
    above 0.5, so at the default threshold they flag almost nobody (recall near 0).
    """
    cv_results.round(4).to_csv(metrics_dir / "cv_results_all_models.csv", index=False)
    balanced = cv_results[cv_results["weighting"] == "balanced"]

    best_per_type = balanced.groupby("model_type", sort=False).head(1).reset_index(drop=True)
    best_per_type.round(4).to_csv(metrics_dir / "cv_best_per_model_type.csv", index=False)
    print(f"Mean CV ROC-AUC by weighting: {cv_results.groupby('weighting')['roc_auc_mean'].mean().round(3).to_dict()}")
    print("Best balanced configuration of each model type (ranked by CV ROC-AUC):")
    show_cols = ["model_id"] + [f"{m}_mean" for m in METRICS] + [f"{SELECTION_METRIC}_std"]
    print(best_per_type[show_cols].round(3).to_string(index=False))

    plot_metric_by_depth(cv_results, "roc_auc", metrics_dir)
    plot_metric_by_depth(cv_results, "recall", metrics_dir)

    best_by_age = (balanced.groupby(["model_type", "feature_set"], sort=False).head(1)
                   .sort_values(f"{SELECTION_METRIC}_mean", ascending=False)
                   .reset_index(drop=True))

    # raw age has 60 distinct values so deep trees can overfit it; binned has 6
    rows = []
    for model_type in MODEL_BUILDERS:
        best = best_by_age[best_by_age["model_type"] == model_type].set_index("feature_set")
        rows.append({
            "model_type": model_type,
            "raw_age_best_depth": best.loc["raw_age", "depth"],
            "raw_age_cv_roc_auc": best.loc["raw_age", "roc_auc_mean"],
            "binned_age_best_depth": best.loc["binned_age", "depth"],
            "binned_age_cv_roc_auc": best.loc["binned_age", "roc_auc_mean"],
        })
    raw_vs_binned = pd.DataFrame(rows)
    raw_vs_binned["difference_binned_minus_raw"] = raw_vs_binned["binned_age_cv_roc_auc"] - raw_vs_binned["raw_age_cv_roc_auc"]
    raw_vs_binned.round(4).to_csv(metrics_dir / "cv_raw_vs_binned_age.csv", index=False)
    plot_raw_vs_binned(balanced, metrics_dir / "cv_raw_vs_binned_age_by_depth.png")
    return best_by_age


def plot_raw_vs_binned(balanced, path):
    fig, axes = plt.subplots(1, len(MODEL_BUILDERS), figsize=(20, 4), sharey=True)
    for ax, model_type in zip(axes, MODEL_BUILDERS):
        for feature_set, colour in zip(FEATURE_SETS, ["#4C78A8", "#F58518"]):
            line = balanced[(balanced["model_type"] == model_type)
                            & (balanced["feature_set"] == feature_set)].sort_values("depth")
            mean, std = line["roc_auc_mean"], line["roc_auc_std"]
            ax.plot(line["depth"], mean, marker="o", color=colour, label=feature_set)
            ax.fill_between(line["depth"], mean - std, mean + std, color=colour, alpha=0.15)
        ax.set_title(model_type)
        ax.set_xlabel("Max depth")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("CV ROC-AUC (mean +/- 1 sd)")
    axes[0].legend()
    fig.suptitle("Raw vs binned age by depth (balanced models, training-set cross-validation)")
    fig.tight_layout()
    save_fig(fig, path)


'''
Refit, Evaluate and Draw the trees
'''
def evaluate_final(best_by_age, train_df, test_df, metrics_dir):
    y_train, y_test = train_df[TARGET], test_df[TARGET]
    final_models, rows = {}, []
    for _, cfg in best_by_age.iterrows():
        cols = FEATURE_SETS[cfg["feature_set"]]
        model = fit_model(MODEL_BUILDERS[cfg["model_type"]](int(cfg["depth"])), train_df[cols], y_train, cfg["weighting"])
        prob = model.predict_proba(test_df[cols])[:, 1]
        final_models[cfg["model_id"]] = {"model": model, "cfg": cfg, "cols": cols, "prob": prob}
        rows.append(test_row(cfg, y_test, prob))

    test_results = pd.DataFrame(rows)  # already in CV ROC-AUC order, the test set is used to confirm it
    test_results.round(4).to_csv(metrics_dir / "test_results.csv", index=False)
    print("Ranked by CV ROC-AUC; test scores shown alongside:")
    print(test_results[["model_id", "cv_roc_auc", "test_roc_auc", "test_recall", "test_precision",
                        "test_resistant_flagged", "test_resistant_missed", "test_false_alarms"]]
          .round(3).to_string(index=False))
    return final_models, test_results


def plot_roc_curves(final_models, test_results, y_test, path):
    # One panel per age version
    fig, axes = plt.subplots(1, len(FEATURE_SETS), figsize=(13, 6))
    for ax, feature_set in zip(axes, FEATURE_SETS):
        for mid, fm in final_models.items():
            if fm["cfg"]["feature_set"] != feature_set:
                continue
            fpr, tpr, _ = roc_curve(y_test, fm["prob"])
            auc = test_results.set_index("model_id").loc[mid, "test_roc_auc"]
            ax.plot(fpr, tpr, label=f"{fm['cfg']['model_type']} (depth {fm['cfg']['depth']}), AUC {auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random guess, AUC 0.5")
        ax.set_title(f"Test-set ROC curves: {feature_set}")
        ax.set_xlabel("False positive rate (non-resistant flagged)")
        ax.set_ylabel("True positive rate (recall)")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, path)


def plot_confusion_matrices(final_models, y_test, path):
    # Rows = age version, columns = model type
    fig, axes = plt.subplots(len(FEATURE_SETS), len(MODEL_BUILDERS), figsize=(20, 8))
    for i, feature_set in enumerate(FEATURE_SETS):
        for j, model_type in enumerate(MODEL_BUILDERS):
            fm = next(fm for fm in final_models.values()
                      if fm["cfg"]["feature_set"] == feature_set and fm["cfg"]["model_type"] == model_type)
            ConfusionMatrixDisplay.from_predictions(
                y_test, (fm["prob"] >= 0.5).astype(int), ax=axes[i, j], colorbar=False,
                display_labels=["Not resistant", "Resistant"])
            axes[i, j].set_title(f"{model_type} d{fm['cfg']['depth']}\n{feature_set}", fontsize=10)
    fig.suptitle("Test-set confusion matrices (threshold 0.5)")
    fig.tight_layout()
    save_fig(fig, path)


def representative_tree(model, mid):
    # The single tree drawn for a model, and a caption saying how far it represents the model
    if isinstance(model, DecisionTreeClassifier):
        return model, "single tree"
    whole_model = f"whole-model view: {mid}_surface.png"
    if isinstance(model, GradientBoostingClassifier):
        return model.estimators_[0, 0], f"first of {N_ESTIMATORS} trees; values are log-odds corrections; {whole_model}"
    first = model.estimators_[0]
    if isinstance(model, RandomForestClassifier):
        # Default max_features="sqrt" of 2 features = 1: each split may only use one randomly
        # chosen feature, so a single forest tree's splits are partly chance
        return first, (f"first of {N_ESTIMATORS} trees; each split considers {first.max_features_} of "
                       f"{model.n_features_in_} features at random, so its splits are partly chance; {whole_model}")
    return first, f"first of {N_ESTIMATORS} trees; {whole_model}"


def leaf_rules(fitted_tree, feature_names):
    # Map each leaf node id to the chain of conditions that leads to it
    t, rules = fitted_tree.tree_, {}

    def walk(node, conditions):
        if t.children_left[node] == -1:
            rules[node] = " AND ".join(conditions) or "(all patients)"
            return
        name, threshold = feature_names[t.feature[node]], t.threshold[node]
        walk(t.children_left[node], conditions + [f"{name} <= {threshold:.1f}"])
        walk(t.children_right[node], conditions + [f"{name} > {threshold:.1f}"])

    walk(0, [])
    return rules


def save_tree_diagrams(final_models, train_df, trees_dir, top_id):
    """For each model: <model_id>.png diagram and <model_id>_rules.txt with a leaf table.

    Models use balanced weights, so node `value`s in the diagram are weighted (resistant cases
    count about 3x). The leaf table gives the real, unweighted % resistant per leaf.
    """
    y_train = train_df[TARGET].to_numpy()
    for mid, fm in final_models.items():
        tree_, note = representative_tree(fm["model"], mid)
        cols = fm["cols"]
        depth = tree_.get_depth()

        fig, ax = plt.subplots(figsize=(min(40, 4 + 2 ** depth * 1.8), 2.5 + 1.8 * depth))
        is_classifier = isinstance(tree_, DecisionTreeClassifier)
        plot_tree(tree_, feature_names=cols, filled=True, rounded=True, proportion=is_classifier,
                  class_names=["not resistant", "resistant"] if is_classifier else None, ax=ax)
        ax.set_title(f"{mid}\n({note})", fontsize=12)
        save_fig(fig, trees_dir / f"{mid}.png")

        #Matched to expect data type. Ensemble: fitted on array. Tree: DataFrame
        X = train_df[cols] if hasattr(tree_, "feature_names_in_") else train_df[cols].to_numpy()
        leaves = pd.DataFrame({"leaf": tree_.apply(X), "resistant": y_train})
        leaf_table = leaves.groupby("leaf")["resistant"].agg(train_patients="size", pct_resistant="mean")
        leaf_table["pct_resistant"] = (leaf_table["pct_resistant"] * 100).round(1)
        leaf_table.insert(0, "rule", pd.Series(leaf_rules(tree_, cols)))
        leaf_table = leaf_table.sort_values(by="pct_resistant", ascending=False)
        fm["leaf_table"] = leaf_table

        with open(trees_dir / f"{mid}_rules.txt", "w") as f:
            f.write(f"{mid}\nTree shown: {note}\n\n")
            f.write(export_text(tree_, feature_names=cols))
            f.write("\nLeaves, highest risk first (actual training-set patients, unweighted):\n")
            f.write(leaf_table.to_string(index=False))
            f.write("\n")

        if mid == top_id:
            print(f"Leaf table for the top-ranked model, {mid} ({note}):")
            print(leaf_table.to_string(index=False))


def draw_surface(ax, fm, ages):
    # Whole-model predicted risk on an age x IMD grid
    imds = np.array(sorted(IMD_VALUES))
    grid_age, grid_imd = np.meshgrid(ages, imds)
    grid = pd.DataFrame({"age": grid_age.ravel(), "imd": grid_imd.ravel()})
    grid["age_band"] = grid["age"] // 10 * 10
    prob = fm["model"].predict_proba(grid[fm["cols"]])[:, 1].reshape(grid_age.shape)
    mesh = ax.pcolormesh(ages, imds, prob, shading="nearest", cmap="RdBu_r", vmin=0, vmax=1)
    ax.set_yticks(imds)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("IMD quintile (1 = most deprived)")
    return mesh


def save_decision_surfaces(final_models, df, trees_dir):
    """<model_id>_surface.png for every model, plus all models on one page.

    With only two features the surface shows everything a model has learned, including the
    100-tree ensembles and voting models that a single tree diagram cannot represent.
    """
    ages = np.arange(df["age"].min(), df["age"].max() + 1)
    label = "Predicted P(resistant)\n(balanced weights: red > 0.5 = flagged)"

    for mid, fm in final_models.items():
        fig, ax = plt.subplots(figsize=(9, 3.5))
        mesh = draw_surface(ax, fm, ages)
        ax.set_title(f"Decision surface: {mid}")
        fig.colorbar(mesh, ax=ax, label=label)
        save_fig(fig, trees_dir / f"{mid}_surface.png")

    # One row per age version, models in CV rank order within each row
    by_set = {fs: [mid for mid, fm in final_models.items() if fm["cfg"]["feature_set"] == fs] for fs in FEATURE_SETS}
    n_cols = max(len(mids) for mids in by_set.values())
    mesh = None
    fig, axes = plt.subplots(len(by_set), n_cols, figsize=(4.2 * n_cols, 3.4 * len(by_set)), squeeze=False,
                             layout="constrained")
    for i, mids in enumerate(by_set.values()):
        for j in range(n_cols):
            ax = axes[i, j]
            if j >= len(mids):
                ax.set_visible(False)
                continue
            mesh = draw_surface(ax, final_models[mids[j]], ages)
            ax.set_title(mids[j].replace("_balanced_", "\n"), fontsize=9)
            ax.label_outer()  # axis labels on the left column and bottom row only
    fig.colorbar(mesh, ax=axes, label=label, shrink=0.8)
    fig.suptitle("Decision surfaces: whole-model predicted risk by age and IMD")
    save_fig(fig, trees_dir / "decision_surfaces_all_models.png")


'''
Voting classifier, risk factors, extension and final comparison
'''
def run_voting(best_by_age, train_df, test_df, final_models, trees_dir, metrics_dir):
    """
    Soft-voting ensemble of the top 3 balanced models, one per age version.
    Scored like every other model
    """
    y_train, y_test = train_df[TARGET], test_df[TARGET]

    def build_voting(members):
        return VotingClassifier(
            [(r.model_type, MODEL_BUILDERS[r.model_type](int(r.depth))) for r in members.itertuples()],
            voting="soft")

    rows = []
    for feature_set, cols in FEATURE_SETS.items():
        members = best_by_age[best_by_age["feature_set"] == feature_set].head(VOTING_TOP_N)
        vid = f"voting_top{VOTING_TOP_N}_balanced_{feature_set}"

        mean, _ = cv_scores(lambda: build_voting(members), train_df[cols], y_train, train_df["patient_id"], "balanced")
        cfg = pd.Series({"model_id": vid, "model_type": "voting",
                         "depth": "/".join(str(d) for d in members["depth"]),  # member depths, in member order
                         "weighting": "balanced", "feature_set": feature_set,
                         **{f"{m}_mean": mean[m] for m in METRICS}})

        model = fit_model(build_voting(members), train_df[cols], y_train, "balanced")
        prob = model.predict_proba(test_df[cols])[:, 1]
        final_models[vid] = {"model": model, "cfg": cfg, "cols": cols, "prob": prob}
        rows.append(test_row(cfg, y_test, prob))

        with open(trees_dir / f"{vid}_components.txt", "w") as f:
            f.write(f"{vid}\nSoft voting (average of predicted probabilities). No single tree to draw;\n"
                    f"see {vid}_surface.png for the whole model.\n\n")
            f.write(f"Members (top {VOTING_TOP_N} balanced {feature_set} models by CV ROC-AUC):\n")
            f.write(members[["model_id", "roc_auc_mean", "recall_mean"]].round(3).to_string(index=False))
            f.write("\n\nSee each member's diagram and rules file in this folder.\n")

    voting_results = pd.DataFrame(rows)
    voting_results.round(4).to_csv(metrics_dir / "voting_results.csv", index=False)
    print("Voting models:")
    print(voting_results[["model_id", "depth", "cv_roc_auc", "test_roc_auc", "test_recall"]].round(3).to_string(index=False))
    return voting_results


def perm_importance(model, cols, test_df):
    # Drop in test ROC-AUC when one feature is shuffled; unlike impurity importance,
    # not biased towards features with many distinct values 
    r: Any = permutation_importance(model, test_df[cols], test_df[TARGET], scoring="roc_auc",
                               n_repeats=N_PERM_REPEATS, random_state=RANDOM_STATE)
    return pd.DataFrame({"permutation_mean": r.importances_mean, "permutation_std": r.importances_std}, index=cols)


def cached_perm_importance(fm, test_df):
    # Permutation importance of a final model, computed once and reused
    if "perm" not in fm:
        fm["perm"] = perm_importance(fm["model"], fm["cols"], test_df)
    return fm["perm"]


def impurity_importance(model):
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    # BaggingClassifier has no feature_importances_; average its trees (each sees every feature)
    return np.mean([est.feature_importances_ for est in model.estimators_], axis=0)


def importance_table(model, perm):
    table = perm.copy()
    table.insert(0, "impurity", impurity_importance(model))
    return table


def plot_permutation(ax, table):
    ax.bar(table.index, table["permutation_mean"], yerr=table["permutation_std"], capsize=4, color="#F58518")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Drop in ROC-AUC when shuffled (+/- 1 sd)")


def plot_importance(table, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(table.index, table["impurity"], color="#4C78A8")
    axes[0].set_title("Impurity importance (built-in)")
    axes[0].set_ylabel("Share of impurity reduction")
    plot_permutation(axes[1], table)
    axes[1].set_title(f"Permutation importance (test set, {N_PERM_REPEATS} repeats)")
    fig.suptitle(title)
    fig.tight_layout()
    save_fig(fig, path)


def risk_factors(top_id, final_models, train_df, test_df, importance_dir):
    """Impurity vs permutation importance for the top model, plus the noise check.

    The noise check refits the top configuration with the random noise column added. A real
    risk factor should clearly beat noise on permutation importance.
    """
    top = final_models[top_id]
    top_cfg = top["cfg"]

    top_importance = importance_table(top["model"], cached_perm_importance(top, test_df))
    top_importance.round(4).to_csv(importance_dir / "importance_permutation_vs_impurity.csv", index_label="feature")
    print(f"Top model: {top_id}\n{top_importance.round(3).to_string()}")
    plot_importance(top_importance, f"Feature importance: {top_id}",
                    importance_dir / "importance_permutation_vs_impurity.png")

    noise_cols = top["cols"] + ["noise"]
    noise_model = fit_model(MODEL_BUILDERS[top_cfg["model_type"]](int(top_cfg["depth"])),
                            train_df[noise_cols], train_df[TARGET], top_cfg["weighting"])
    noise_importance = importance_table(noise_model, perm_importance(noise_model, noise_cols, test_df))
    noise_importance.round(4).to_csv(importance_dir / "noise_check.csv", index_label="feature")
    plot_importance(noise_importance, f"Noise check: {top_id} + random noise column",
                    importance_dir / "noise_check.png")

    # Clearly above noise: the feature's lower 2-sd bound beats the noise column's upper 2-sd bound
    noise_perm, noise_sd = noise_importance.loc["noise", ["permutation_mean", "permutation_std"]]
    for feature in top["cols"]:
        perm, sd = float(noise_importance.loc[feature, "permutation_mean"]), float(noise_importance.loc[feature, "permutation_std"])
        verdict = "clearly above noise" if perm - 2 * sd > noise_perm + 2 * noise_sd else "NOT clearly above noise"
        print(f"{feature}: permutation {perm:.3f} +/- {sd:.3f} vs noise {noise_perm:.3f} +/- {noise_sd:.3f} -> {verdict}")
    print(f"Impurity importance given to pure noise: {noise_importance.loc['noise', 'impurity']:.1%}")
    return noise_importance


def run_extension(top_id, final_models, raw, raw_index, df, train_df, test_df, extension_dir):
    """Outside the brief: does adding sex_male and had_surgery_past_yr help the top configuration?

    Rows with a missing or non-0/1 value in either extra column are dropped for this comparison only.
    """
    missing_cols = [c for c in EXTRA_FEATURES if c not in raw.columns]
    if missing_cols:
        print(f"Extension skipped: columns missing from the data: {missing_cols}")
        return

    top = final_models[top_id]
    top_cfg = top["cfg"]
    ext_cols = top["cols"] + EXTRA_FEATURES

    extra = raw.loc[raw_index, EXTRA_FEATURES].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    valid = extra.isin([0, 1]).all(axis=1)
    print(f"Extension: {int((~valid).sum())} rows dropped for missing or non-0/1 extra values")
    for col in EXTRA_FEATURES:
        rate = df.loc[valid, TARGET].groupby(extra.loc[valid, col]).mean() * 100
        print(f"  % resistant by {col}: {rate.round(1).to_dict()}")

    ext_train = train_df.join(extra[valid].astype(int), how="inner")
    ext_test = test_df.join(extra[valid].astype(int), how="inner")
    build = lambda: MODEL_BUILDERS[top_cfg["model_type"]](int(top_cfg["depth"]))

    def evaluate(cols, label):
        mean, _ = cv_scores(build, ext_train[cols], ext_train[TARGET], ext_train["patient_id"], top_cfg["weighting"])
        model = fit_model(build(), ext_train[cols], ext_train[TARGET], top_cfg["weighting"])
        test_scores = score(ext_test[TARGET], model.predict_proba(ext_test[cols])[:, 1])
        row = {"version": label, "features": ", ".join(cols)}
        row.update({f"cv_{m}": mean[m] for m in METRICS})
        row.update({f"test_{m}": test_scores[m] for m in METRICS})
        return row, model

    base_row, _ = evaluate(top["cols"], "brief features")
    ext_row, ext_model = evaluate(ext_cols, "brief + extra features")
    extension_results = pd.DataFrame([base_row, ext_row])
    extension_results.round(4).to_csv(extension_dir / "extension_results.csv", index=False)
    print(extension_results[["version", "cv_roc_auc", "test_roc_auc"]].round(3).to_string(index=False))

    ext_importance = perm_importance(ext_model, ext_cols, ext_test)
    ext_importance.round(4).to_csv(extension_dir / "extension_importance.csv", index_label="feature")

    fig, ax = plt.subplots(figsize=(8, 4))
    plot_permutation(ax, ext_importance)
    ax.set_title(f"Extension: permutation importance ({top_cfg['model_type']}, depth {top_cfg['depth']}, test set)")
    fig.tight_layout()
    save_fig(fig, extension_dir / "extension_importance.png")


def final_comparison(test_results, voting_results, final_models, test_df, metrics_dir):
    """
    All 10 tree models plus the 2 voting models, ranked by CV ROC-AUC.
    age/imd importance: permutation importance on the test set
    highest_risk_group: highest-risk leaf of the drawn tree, with the real training-set % resistant
    """
    final = pd.concat([test_results, voting_results], ignore_index=True)

    rows = []
    for mid in final["model_id"]:
        fm = final_models[mid]
        age_col = "age" if "age" in fm["cols"] else "age_band"
        imp = cached_perm_importance(fm, test_df)
        leaf_table = fm.get("leaf_table")
        group = ""
        if leaf_table is not None:
            top_leaf = leaf_table.iloc[0]
            group = f"{top_leaf['rule']} ({top_leaf['pct_resistant']}% resistant, n={top_leaf['train_patients']})"
        rows.append({
            "model_id": mid,
            "age_importance": imp.loc[age_col, "permutation_mean"],
            "imd_importance": imp.loc["imd", "permutation_mean"],
            "top_risk_factor": "age" if imp.loc[age_col, "permutation_mean"] >= imp.loc["imd", "permutation_mean"] else "imd",
            "highest_risk_group": group,
        })

    final = (final.merge(pd.DataFrame(rows), on="model_id")
             .sort_values("cv_roc_auc", ascending=False)
             .reset_index(drop=True))
    final.insert(0, "rank", range(1, len(final) + 1))
    final.round(4).to_csv(metrics_dir / "final_model_comparison.csv", index=False)
    print(final[["rank", "model_id", "cv_roc_auc", "test_roc_auc", "test_recall",
                 "age_importance", "imd_importance", "top_risk_factor"]].round(3).to_string(index=False))
    return final


'''
Main
'''
def step(title):
    print(f"\n=== {title} ===", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="input CSV (default: %(default)s)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="output folder (default: %(default)s)")
    args = parser.parse_args(argv)

    os.umask(0o077)  # everything this script writes is private to the owner
    print("Python executable:", sys.executable)
    print(f"pandas {pd.__version__} | numpy {np.__version__} | scikit-learn {sklearn.__version__} | "
          f"matplotlib {matplotlib.__version__}")
    dirs = make_dirs(args.output)

    step("Step 1: load data")
    check_data_permissions(args.data)  # before the sensitive file is read
    raw = load_data(args.data)

    step("Step 2: clean and validate")
    df, raw_index, log_df = clean(raw)
    save_cleaning_outputs(raw, df, log_df, args.data, dirs["data"])
    plot_data_overview(df, dirs["data"] / "data_overview.png")

    step("Step 3: feature variants and train/test split")
    # Without an id column every row is treated as a separate patient
    patient_ids = raw.loc[raw_index, "id"].to_numpy() if "id" in raw.columns else np.arange(len(df))
    train_df, test_df = add_features_and_split(df, patient_ids, dirs["data"])

    step("Step 4: cross-validate model types and depths")
    cv_results = run_cv(train_df)
    best_by_age = select_best(cv_results, dirs["metrics"])
    top_id = best_by_age.loc[0, "model_id"]

    step("Step 5: final models, test evaluation and tree diagrams")
    final_models, test_results = evaluate_final(best_by_age, train_df, test_df, dirs["metrics"])
    plot_roc_curves(final_models, test_results, test_df[TARGET], dirs["metrics"] / "test_roc_curves.png")
    plot_confusion_matrices(final_models, test_df[TARGET], dirs["metrics"] / "test_confusion_matrices.png")
    save_tree_diagrams(final_models, train_df, dirs["trees"], top_id)

    step("Step 6a: soft-voting classifier")
    voting_results = run_voting(best_by_age, train_df, test_df, final_models, dirs["trees"], dirs["metrics"])
    save_decision_surfaces(final_models, df, dirs["trees"])

    step("Step 6b: main risk factors and noise check")
    noise_importance = risk_factors(top_id, final_models, train_df, test_df, dirs["importance"])

    step("Step 6c: extension with extra variables")
    run_extension(top_id, final_models, raw, raw_index, df, train_df, test_df, dirs["extension"])

    step("Step 6d: final model comparison")
    final = final_comparison(test_results, voting_results, final_models, test_df, dirs["metrics"])

    print(f"\nAll outputs saved under {args.output.resolve()}")
    return {"raw": raw, "df": df, "train_df": train_df, "test_df": test_df, "cv_results": cv_results,
            "final_models": final_models, "final": final, "noise_importance": noise_importance,
            "output_dir": args.output}


if __name__ == "__main__":
    main()
