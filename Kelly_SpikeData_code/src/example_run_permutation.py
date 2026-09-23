#!/usr/bin/env python3
"""
Run circular permutation tests to evaluate whether specific neurons
fire in SWRs more often than expected by chance, accounting for firing rate.

Loads:
- spike_times.npy
- spike_clusters.npy
- cluster_KSLabel.tsv
- SWR.csv
Matches spikes to SWR windows and performs circular permutation testing.

THIS WILL TAKE SOME TIME:
~ 7 min for 1000 permutations on ~2000 spikes and ~300 SWRs.

Example usage
-------------
python src/example_run_permutation.py \
    --data_dir "/Users/kellyschulte/Desktop/full_data/7744/PartnerIntro" \
     --swr_csv "/Users/kellyschulte/Desktop/full_data/7744/PartnerIntro/\
7744_PartnerIntro_SWRs_ca2.csv" \
    --save_path "/Users/kellyschulte/Desktop/full_data/7744/PartnerIntro/\
first_spike_permutation.csv" \
    --plot_cluster 16
"""

import argparse
import os

from analysis_utils import (
    make_spike_df,
    match_up,
    circular_permutation_test_with_firing_rate,
    plot_permutation_histogram,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run circular permutation test on SWR spike data."
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing spike_times.npy, "
        "spike_clusters.npy, "
        "and cluster_KSLabel.tsv",
    )

    parser.add_argument(
        "--swr_csv",
        type=str,
        required=True,
        help="Path to SWR CSV file.",
    )

    parser.add_argument(
        "--n_permutations",
        type=int,
        default=1000,
        help="Number of permutations to run (default: 1000).",
    )

    parser.add_argument(
        "--shift",
        type=float,
        default=3,
        help=(
            "Maximum shift range in seconds (default: 3). "
            "Shifts are drawn uniformly from (-shift, +shift)."
        ),
    )

    parser.add_argument(
        "--tail",
        type=str,
        choices=["one", "two"],
        default="two",
        help="Type of statistical test (default: two).",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["all", "first"],
        default="all",
        help="Spike counting mode: 'all' (default) or 'first'.",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to save permutation results CSV.",
    )

    parser.add_argument(
        "--plot_cluster",
        type=int,
        default=None,
        help="Optional cluster ID to plot histogram for.",
    )

    parser.add_argument(
        "--binsize",
        type=int,
        default=30,
        help="Histogram binsize (default: 30).",
    )

    args = parser.parse_args()

    # Build file paths
    spike_times_path = os.path.join(args.data_dir, "spike_times.npy")
    spike_clusters_path = os.path.join(args.data_dir, "spike_clusters.npy")
    kslabels_path = os.path.join(args.data_dir, "cluster_KSLabel.tsv")

    # Confirm files exist
    for f in [
        spike_times_path,
        spike_clusters_path,
        kslabels_path,
        args.swr_csv,
    ]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Expected file not found: {f}")

    # Load spike data (auto-filters KSLabel == "good")
    spike_df = make_spike_df(
        spike_times_path,
        spike_clusters_path,
        kslabels_path,
    )

    # Load SWRs and match spikes
    swr_df = match_up(
        spike_df,
        swr_dir=args.swr_csv,
        only_keep_good=False,
        progress=True,
    )

    # Run permutation test
    result_df, true_counts, all_permuted_counts = (
        circular_permutation_test_with_firing_rate(
            swr_df,
            spike_df,
            tail=args.tail,
            n_permutations=args.n_permutations,
            shift_range_seconds=args.shift,
            save_path=args.save_path,
            mode=args.mode,
        )
    )

    # Optional histogram plot
    if args.plot_cluster is not None:
        cluster = args.plot_cluster

        if cluster not in true_counts:
            print(f"Cluster {cluster} not found in filtered spike data.")
        else:
            plot_permutation_histogram(
                cid=cluster,
                all_permuted_counts=all_permuted_counts,
                true_counts=true_counts,
                binsize=args.binsize,
                two_tailed=(args.tail == "two"),
            )


if __name__ == "__main__":
    main()
