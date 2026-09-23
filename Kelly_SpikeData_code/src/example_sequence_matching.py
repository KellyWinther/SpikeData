
"""
Sequence Overlap Analysis

This script:
1. Loads spike data, behavior windows, and SWR windows.
2. Computes cluster-ID sequences per window.
3. Calculates normalized overlap using Rabin Karp LCCS.
4. Produces a heatmap of SWR behavior overlaps.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from loading_utils import load_spike_data
from loading_utils import filter_dataframe
from loading_utils import group_dataframes_by_time
from sequence_matching_utils import lccs_rabin_karp


# build complex file paths for test or full data set
def get_base_directory(individual):
    """
    Return correct data root depending on individual.

    Parameters
    ----------
    individual :: str

    Returns
    -------
    str
        Base directory path.
    """
    if individual == "test":
        return "data/test_data"
    return "data/full_data"


def build_file_paths(individual, stimuli):
    """
    Construct all required file paths depending on individual.

    For real animals (7742, 7744):
        data/full_data/<ID>/stimuli/<files>

    For test:
        data/test_data/<test files>
    """
    base = get_base_directory(individual)

    # files for test data sets
    if individual == "test":
        root = base
        return {
            "spike_times": os.path.join(root, "test_spike_times.npy"),
            "spike_clusters": os.path.join(root, "test_spike_clusters.npy"),
            "kslabels": os.path.join(root, "test_cluster_KSLabel.tsv"),
            "behavior_csv": os.path.join(
                root, "test_events_with_indices.csv"
            ),
            "swr_csv": os.path.join(
                root, "test_SWRs_ca2.csv"
            ),
        }

    # files from full_data: like 7742 or 7744
    stimuli = stimuli
    extra_tag = "_sleepyvole" if individual == "7742" else ""
    root = os.path.join(base, individual, stimuli)

    return {
        "spike_times": os.path.join(root, "spike_times.npy"),
        "spike_clusters": os.path.join(root, "spike_clusters.npy"),
        "kslabels": os.path.join(root, "cluster_KSLabel.tsv"),
        "behavior_csv": os.path.join(
            root,
            f"{individual}_{stimuli}{extra_tag}_events_with_indices.csv",
        ),
        "swr_csv": os.path.join(
            root,
            f"{individual}_{stimuli}{extra_tag}_SWRs_ca2.csv",
        ),
    }


# load data
def load_data(paths):
    """
    Load spike, behavior, and SWR data.

    Parameters
    ----------
    paths :: dict

    Returns
    -------
    (DataFrame, DataFrame, DataFrame)
    """
    try:
        spike_df = load_spike_data(
            time_dir=paths["spike_times"],
            cluster_dir=paths["spike_clusters"],
            label_dir=paths["kslabels"],
        )
    except FileNotFoundError:
        print("Error: Spike data files not found.")
        sys.exit(1)

    spike_df = filter_dataframe(spike_df, {"KSLabel": ["good"]})
    spike_df = spike_df.reset_index(drop=True)

    try:
        behavior_df = pd.read_csv(paths["behavior_csv"])
    except FileNotFoundError:
        print("Error: Behavior CSV not found.")
        sys.exit(1)

    try:
        swr_df = pd.read_csv(paths["swr_csv"])
    except FileNotFoundError:
        print("Error: SWR CSV not found.")
        sys.exit(1)

    return spike_df, behavior_df, swr_df


# filter behavior dataframe for relevant events
def preprocess_behavior_df(df):
    """
    Filter and rename time columns in behavior dataframe.

    Parameters
    ----------
    df :: pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
    """
    df = filter_dataframe(
        df,
        {"EventType": ["social interaction", "cup interaction"]},
    )
    df = df.reset_index(drop=True)

    df[["indexStart", "indexEnd"]] /= 2500.0
    df = df.rename(columns={"indexStart": "Start", "indexEnd": "Stop"})
    return df


def compute_overlap_matrix(behavior_lists, swr_lists):
    """
    Compute normalized LCCS overlap matrix.

    Parameters
    ----------
    behavior_lists :: list
    swr_lists :: list

    Returns
    -------
    numpy.ndarray
    """
    overlap = np.zeros((len(swr_lists), len(behavior_lists)))

    for bx, bseq in enumerate(tqdm(behavior_lists)):
        if not bseq:
            continue

        bset = set(bseq)

        for sy, sseq in enumerate(swr_lists):
            if not sseq:
                continue

            if bset.isdisjoint(sseq):
                continue

            overlap[sy, bx] = lccs_rabin_karp(sseq, bseq, normalize=True)

    return overlap


def plot_overlap_matrix(matrix):
    """
    Visualize the SWR behavior replay of behavior activity.

    Parameters
    ----------
    matrix :: numpy.ndarray
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(
        matrix,
        cmap="inferno",
        aspect="auto",
        interpolation="none",
        origin="lower",
    )
    plt.xlabel("Behavior Window Index")
    plt.ylabel("SWR Window Index")
    plt.colorbar(label="Normalized Overlap")
    plt.tight_layout()
    plt.show()


def parse_arguments():

    parser = argparse.ArgumentParser(
        description="Compute SWR–behavior overlap matrices."
    )

    parser.add_argument(
        "--individual",
        type=str,
        required=True,
        help="Individual ID ('7742', '7744', or 'test').",
    )

    parser.add_argument(
        "--stimuli",
        type=str,
        required=True,
        help="Social stimuli folder name for full data "
        "(default: PartnerIntro).",
    )

    args = parser.parse_args()

    if args.individual not in ["7742", "7744", "test"]:
        print("Error: individual must be '7742', '7744', or 'test'.")
        sys.exit(1)

    if args.stimuli not in ["PartnerIntro", "SSIntro"]:
        print("Error: stimuli must be 'PartnerIntro' or 'SSIntro'.")
        sys.exit(1)

    return args


def main():
    args = parse_arguments()
    paths = build_file_paths(args.individual, args.stimuli)

    spike_df, behavior_df, swr_df = load_data(paths)
    behavior_df = preprocess_behavior_df(behavior_df)

    behavior_clusters = group_dataframes_by_time(
        window_df=behavior_df,
        event_df=spike_df,
        event_time_column="Time",
        keep_event_columns=["Cluster ID"],
        time_interval_columns=["Start", "Stop"],
        progress=True,
    )

    swr_clusters = group_dataframes_by_time(
        window_df=swr_df,
        event_df=spike_df,
        event_time_column="Time",
        keep_event_columns=["Cluster ID"],
        time_interval_columns=["Start", "Stop"],
        progress=True,
    )

    overlap = compute_overlap_matrix(
        behavior_clusters["Event Cluster IDs"].tolist(),
        swr_clusters["Event Cluster IDs"].tolist(),
    )

    plot_overlap_matrix(overlap)


if __name__ == "__main__":
    main()
