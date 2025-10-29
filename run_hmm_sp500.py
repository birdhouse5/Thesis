#!/usr/bin/env python3
"""
run_hmm_sp500.py

Train a Gaussian HMM on SP500 training data and produce regime classifications
on the test set. Saves predicted states and posterior probabilities to CSV.
"""

import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from pathlib import Path
import logging
from environments.data import PortfolioDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_and_classify_hmm_sp500(n_states: int = 4, seed: int = 0):
    """
    Trains a Gaussian HMM on the SP500 training split and produces regime
    classifications on the test split.
    """
    # === Load dataset ===
    logger.info("📊 Loading SP500 dataset...")
    dataset = PortfolioDataset(asset_class="sp500")
    train_split = dataset.get_split("train")
    test_split = dataset.get_split("test")

    # === Prepare data ===
    logger.info("Preparing training data for HMM fitting...")
    X_train = train_split.data[train_split.feature_cols].values.reshape(-1, train_split.num_features)
    X_test = test_split.data[test_split.feature_cols].values.reshape(-1, test_split.num_features)

    logger.info(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

    # === Fit HMM ===
    hmm = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=200,
        random_state=seed,
        tol=1e-3
    )

    logger.info(f"Fitting Gaussian HMM with {n_states} states...")
    hmm.fit(X_train)
    logger.info(f"HMM converged: {hmm.monitor_.converged}")

    # === Predict on test set ===
    logger.info("Predicting hidden states and regime probabilities on test set...")
    hidden_states = hmm.predict(X_test)
    _, posterior_probs = hmm.score_samples(X_test)

        # === Predict on test set ===
    logger.info("Predicting hidden states and regime probabilities on test set...")
    hidden_states = hmm.predict(X_test)
    _, posterior_probs = hmm.score_samples(X_test)

    # === Attach predictions ===
    test_df = test_split.data.copy().reset_index(drop=True)
    test_df["predicted_state"] = hidden_states

    for i in range(posterior_probs.shape[1]):
        test_df[f"regime_prob_{i}"] = posterior_probs[:, i]

    # === Aggregate by date (mean across tickers) ===
    logger.info("Aggregating to time × regime matrix (mean across tickers)...")
    agg_df = (
        test_df.groupby("date")[[f"regime_prob_{i}" for i in range(n_states)]]
        .mean()
        .reset_index()
        .sort_values("date")
    )

    # === Save outputs ===
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    detailed_path = output_dir / "hmm_sp500_detailed.csv"
    agg_path = output_dir / "hmm_sp500_time_regimes.csv"

    test_df.to_csv(detailed_path, index=False)
    agg_df.to_csv(agg_path, index=False)

    logger.info(f"💾 Saved detailed (per-ticker) results to: {detailed_path.resolve()}")
    logger.info(f"💾 Saved aggregated (time × regime) matrix to: {agg_path.resolve()}")

    # === Summary ===
    logger.info("State counts in test set:")
    for state, count in pd.Series(hidden_states).value_counts().sort_index().items():
        logger.info(f"  State {state}: {count} samples")

    return agg_df


if __name__ == "__main__":
    results = train_and_classify_hmm_sp500()
    print("\n✅ Done. Sample of aggregated (time × regime) output:")
    print(results.head())
