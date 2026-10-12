"""Classifiers and evaluation, built to avoid data leakage.

Rules followed here:
  * the train/test split happens BEFORE any fitting;
  * every preprocessing step (scaling) lives inside a sklearn Pipeline, so in
    cross-validation it is re-fitted on each training fold only;
  * the decision tree (interpretable baseline, Gini) comes before the random forest;
  * K-means is label-free and exploratory -- its clusters are provisional groups,
    compared with the true labels only because the data are synthetic.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, adjusted_rand_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

SEED = 0


def split(X, y, test_size=0.3, seed=SEED):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def decision_tree(max_depth=3, seed=SEED):
    # Trees do not need scaling; the scaler is kept to show the leakage-safe pattern.
    return make_pipeline(StandardScaler(), DecisionTreeClassifier(criterion="gini", max_depth=max_depth, random_state=seed))


def random_forest(n_estimators=300, seed=SEED):
    return make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=n_estimators, random_state=seed))


def evaluate(model, X_train, X_test, y_train, y_test, labels, cv_folds=5, seed=SEED):
    """Fit on train, score on test, and cross-validate on the TRAINING set only."""
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    cv = StratifiedKFold(cv_folds, shuffle=True, random_state=seed)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="balanced_accuracy")
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "confusion": confusion_matrix(y_test, pred, labels=labels),
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std(ddof=1)),
        "pred": pred,
    }


def tree_rules(model, feature_names):
    """Readable if/else rules of the fitted tree (thresholds are in SCALED units)."""
    return export_text(model[-1], feature_names=list(feature_names))


def tree_rules_physical_units(model, feature_names):
    """Same rules with thresholds converted back to physical/feature units."""
    scaler, tree = model[0], model[-1].tree_
    lines = []

    def walk(node, depth):
        pad = "|   " * depth
        if tree.children_left[node] == -1:
            cls = model[-1].classes_[int(np.argmax(tree.value[node]))]
            lines.append(f"{pad}--> {cls}")
            return
        f = tree.feature[node]
        thr = tree.threshold[node] * scaler.scale_[f] + scaler.mean_[f]
        lines.append(f"{pad}if {feature_names[f]} <= {thr:.4g}:")
        walk(tree.children_left[node], depth + 1)
        lines.append(f"{pad}else:  # {feature_names[f]} > {thr:.4g}")
        walk(tree.children_right[node], depth + 1)

    walk(0, 0)
    return "\n".join(lines)


def kmeans_explore(X, k=4, seed=SEED):
    """Label-free clustering on standardized features. Returns cluster ids (provisional groups)."""
    Z = StandardScaler().fit_transform(X)
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)


def cluster_agreement(true_labels, clusters):
    """Adjusted Rand index -- only computable here because the synthetic labels are known."""
    return float(adjusted_rand_score(true_labels, clusters))
