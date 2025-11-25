#!/usr/bin/env python3
"""
Script version of the notebook `random_forest.ipynb`.

Features:
- Loads preprocessed data via `load_preprocessed_data()`
- Trains a baseline RandomForest (n_jobs=-1)
- Small timing helper to estimate randomized search runtime
- Optional RandomizedSearchCV tuning (keeps n_jobs=-1 as in notebook)

Usage:
    python src/classification/random_forest.py                # run baseline train + eval
    python src/classification/random_forest.py --time         # run timing probe
    python src/classification/random_forest.py --search       # run randomized search (costly)

Note: The notebook used `n_jobs=-1` for both the estimator and RandomizedSearchCV which can trigger nested parallelism
(depending on your BLAS/Scikit-learn config). If you see poor performance, consider running with `--search-parallel search`
or `--search-parallel estimator` to control which layer parallelizes.

"""

import argparse
import warnings
import time
from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, ParameterSampler, cross_validate
from scipy.stats import randint

from load_data import load_preprocessed_data
from evaluation import evaluate_model

warnings.filterwarnings("ignore")


def encode_categoricals(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].astype(str).fillna("__MISSING__")
            X[col] = pd.factorize(X[col])[0]
    return X


def run_baseline(X, y):
    print("Running baseline RandomForest training...")
    X_train, X_test = None, None
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1,  # use all cores
    )
    t0 = time.perf_counter()
    rf_model.fit(X_train, y_train)
    t1 = time.perf_counter()
    print(f"Baseline training time: {t1-t0:.2f}s")

    metrics = evaluate_model(rf_model, X_test, y_test)
    print("\n=== Random Forest Evaluation ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"PR AUC: {metrics['pr_auc']:.4f}")
    return rf_model


def timing_probe(X_train, y_train, sample_size=50_000, n_trials=30):
    print("Running a timing probe (single representative trial) to estimate tuning runtime")
    SAMPLE_SIZE = min(len(X_train), sample_size)
    X_sub = X_train.sample(n=SAMPLE_SIZE, random_state=42)
    y_sub = y_train.loc[X_sub.index]

    param_dist = {
        'n_estimators': np.arange(100, 400),
        'max_depth': np.arange(5, 30),
        'min_samples_split': np.arange(2, 10),
        'min_samples_leaf': np.arange(1, 5),
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }
    # ParameterSampler is iterable but not itself an iterator; wrap with iter() or list()
    params = next(iter(ParameterSampler(param_dist, n_iter=1, random_state=42)))

    rf_trial = RandomForestClassifier(random_state=42, n_jobs=-1, **params)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    t0 = time.perf_counter()
    _ = cross_validate(rf_trial, X_sub, y_sub, cv=cv, n_jobs=1)
    trial_time = time.perf_counter() - t0

    print(f"One trial (3-fold CV) on {len(X_sub):,} samples took {trial_time:.2f} s")
    est_total = trial_time * n_trials
    print(f"Estimated serial time for {n_trials} trials: {est_total/60:.2f} minutes (~{est_total/3600:.2f} hours)")
    return trial_time


def run_random_search(X_train, y_train, n_iter=30, parallel_mode='both', sample_size: int = 50_000):
    """Run RandomizedSearchCV on a sampled subset of the training data.

    This avoids running the expensive search on the full training set. The
    subset size defaults to 50,000 (as requested).
    """
    print(f"Running RandomizedSearchCV for RandomForest on up to {sample_size:,} samples (this can take a long time)")

    # Sample the training set for search to limit runtime/memory
    SAMPLE_SIZE = min(len(X_train), int(sample_size))
    X_sub = X_train.sample(n=SAMPLE_SIZE, random_state=42)
    y_sub = y_train.loc[X_sub.index]
    print(f"Using sampled subset: X_sub.shape={X_sub.shape}")

    rf_clf = RandomForestClassifier(random_state=42, n_jobs=-1 if parallel_mode in ('estimator','both') else 1)

    param_dist_rf = {
        'n_estimators': randint(100, 400),
        'max_depth': randint(5, 30),
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 5),
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }

    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # If parallel_mode='search' or 'both', RandomizedSearchCV will parallelize over candidates.
    rs_n_jobs = -1 if parallel_mode in ('search','both') else 1

    random_search_rf = RandomizedSearchCV(
        estimator=rf_clf,
        param_distributions=param_dist_rf,
        n_iter=n_iter,
        scoring='roc_auc',
        cv=kfold,
        random_state=42,
        n_jobs=rs_n_jobs,
        verbose=2,
    )

    t0 = time.perf_counter()
    random_search_rf.fit(X_sub, y_sub)
    t1 = time.perf_counter()
    print(f"Randomized search finished in {(t1-t0)/60:.2f} minutes")

    print("Best parameters for Random Forest:", random_search_rf.best_params_)
    print("Best ROC AUC (CV):", random_search_rf.best_score_)

    best_rf = random_search_rf.best_estimator_
    return best_rf, random_search_rf


def main(args):
    X, y = load_preprocessed_data()
    print("Data loaded: X.shape=", X.shape)
    X = encode_categoricals(X)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if args.time:
        timing_probe(X_train, y_train, sample_size=args.sample_size, n_trials=args.n_iter)
        return

    if args.search:
        best_rf, rs = run_random_search(X_train, y_train, n_iter=args.n_iter, parallel_mode=args.parallel, sample_size=args.sample_size)
        metrics = evaluate_model(best_rf, X_test, y_test)
        print("\n=== Best RF Evaluation ===")
        pprint(metrics)
        return

    # default: baseline train + evaluate
    rf = run_baseline(X, y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--time', action='store_true', help='Run a timing probe for randomized search')
    parser.add_argument('--sample-size', type=int, default=50_000, help='Sample size for timing probe')
    parser.add_argument('--n-iter', type=int, default=30, help='n_iter for RandomizedSearchCV or timing estimate')
    parser.add_argument('--search', action='store_true', help='Run RandomizedSearchCV (costly)')
    parser.add_argument('--parallel', choices=['search','estimator','both'], default='both',
                        help='Where to apply parallelism: search (parallelize candidates), estimator (parallelize trees), or both (default, may cause nested parallelism)')
    args = parser.parse_args()
    main(args)
