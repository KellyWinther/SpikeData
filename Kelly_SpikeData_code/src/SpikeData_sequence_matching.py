
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

    For datesets from full_data, the expected structure is:
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
            # Default location for the cluster -> neuron-type table.
            "neuron_types": os.path.join(root, "test_neuron_types.csv"),
        }

    # files from full_data:
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
        # Expected table with one row per cluster and columns identifying
        # cluster ID and neuron type. This can be overridden from the CLI.
        "neuron_types": os.path.join(root, "neuron_types.csv"),
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


def filter_spikes_by_cluster_type(spike_df, cluster_type_csv, classification):
    """Keep spikes belonging to clusters of one requested neuron type.

    The cluster_type.csv file should contain one row per cluster
    and the first two columns are ClusterID and Classification:

        ClusterID,Classification
        12,Pyramidal Neuron
        18,FSI
        27,Other

    document must be .csv like pyh2 outputs
    """
    if not os.path.exists(cluster_type_csv):
        print(f"Error: neuron-type file not found: {cluster_type_csv}")
        sys.exit(1)

    type_df = pd.read_csv(cluster_type_csv)

    cluster_candidates = ["ClusterID"]
    type_candidates = ["Classification"]

    cluster_col = next((c for c in cluster_candidates if c in type_df.columns), None)
    type_col = next((c for c in type_candidates if c in type_df.columns), None)

    if cluster_col is None or type_col is None:
        print(
            "Error: neuron-type file must contain a cluster-ID column "
            f"({cluster_candidates}) and a classification column ({type_candidates})."
        )
        print(f"Columns found: {list(type_df.columns)}")
        sys.exit(1)

    selected_clusters = type_df.loc[
        type_df[type_col].astype(str) == classification, cluster_col
    ]

    if selected_clusters.empty:
        print(f"Error: no clusters labeled '{classification}' were found.")
        sys.exit(1)

    selected_clusters = set(pd.to_numeric(selected_clusters, errors="raise").astype(int))

    spike_df = spike_df[spike_df["Cluster ID"].astype(int).isin(selected_clusters)].copy()
    spike_df = spike_df.reset_index(drop=True)

    clusters = spike_df["Cluster ID"].nunique()

    print(
        f"Neuron-type filter: {classification} | "
        f"{spike_df['Cluster ID'].nunique()}/{clusters} clusters | "
    )

    return spike_df


# filter behavior dataframe for relevant events
def preprocess_behavior_df(boris_df):
    """
    Filter and rename time columns in behavior dataframe.

    Parameters
    ----------
    boris_df :: pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
    """

    # filter Boris labeled frames for soical interaction while awake (i.e. not a rest period)
    df = filter_dataframe(boris_df, {"EventType": ["social interaction", "cup interaction"]},)
    df = df.reset_index(drop=True)

    # convert index to seconds and rename columns 
    # (TO DO - is this needed, becuase the behavior alignment code 
    # gets the frames to match to the recoring timestamp... recheck this againt thebehvaior alignment code in jupyter notebook
    # and remove conversion if not needed)

    df[["indexStart", "indexEnd"]] /= 2500.0
    df = df.rename(columns={"indexStart": "Start", "indexEnd": "Stop"})
    return df


## This is the attempt of checking which spikes from my spike_df are in the behavior_df and also in a ripple
# from the swr_df time windows. 
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
    plt.colorbar(label="Overlap")
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
        help="Individual ID ('7742', '7744', '9490', or 'test').",
    )

    parser.add_argument(
        "--stimuli",
        type=str,
        required=True,
        help="Recording folder name for data "
        "(default: OSIntro).",
    )

    parser.add_argument(
        "--cluster-type",
        type=str,
        default="Pyramidal Neuron",
        choices=["Pyramidal Neuron", "FSI", "Other"],
        help="Cluster type to include (default: Pyramidal Neuron).",
    )

    parser.add_argument(
        "--cluster-type-file",
        type=str,
        default=None,
        help=(
            "CSV mapping cluster IDs to cluster types. "
            "Default: <recording folder>/cluster_type.csv."
        ),
    )

    args = parser.parse_args()

    if args.individual not in ["7742", "7744", "9490" "test"]:
        print("Error: individual must be '7742', '7744', '9490', or 'test'.")
        sys.exit(1)

    if args.stimuli not in ["OSIntro", "OS_Choice_d1"]:
        print("Error: stimuli must be 'OSIntro' or 'OS_Choice_d1'.")
        sys.exit(1)

    return args


def main():
    args = parse_arguments()
    paths = build_file_paths(args.individual, args.stimuli)

    spike_df, behavior_df, swr_df = load_data(paths)

    cluster_type_file = args.cluster_type_file or paths["cluster_types"]
    spike_df = filter_spikes_by_cluster_type(
        spike_df,
        cluster_type_file=cluster_type_file,
        cluster_type=args.cluster_type,
    )

    behavior_df = preprocess_behavior_df(behavior_df)

    behavior_clusters = group_dataframes_by_time(
        window_df=behavior_df,
        event_df=spike_df,
        event_time_column="Time",
        keep_event_columns=["ClusterID"],
        time_interval_columns=["Start", "Stop"],
        progress=True,
    )

    swr_clusters = group_dataframes_by_time(
        window_df=swr_df,
        event_df=spike_df,
        event_time_column="Time",
        keep_event_columns=["ClusterID"],
        time_interval_columns=["Start", "Stop"],
        progress=True,
    )

    overlap = compute_overlap_matrix(
        behavior_clusters["Event ClusterIDs"].tolist(),
        swr_clusters["Event ClusterIDs"].tolist(),
    )

    plot_overlap_matrix(overlap)


if __name__ == "__main__":
    main()
