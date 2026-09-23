import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


def make_correlation_dictionary(
    df: pd.DataFrame,
    id_column_name: str = "Event Cluster IDs",
    normalize: bool = True,
) -> dict:
    """
    Generate a nested dictionary describing co-firing frequencies
    between neuron clusters.

    corr_dict[a][b] returns how many times cluster b fired in SWRs
    where cluster a also fired.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame where the ID column contains a list of cluster IDs
        firing in each SWR event.

    id_column_name : str
        Name of the column that stores cluster ID lists.

    normalize : bool
        If True, divide counts by the total SWR count for that baseline
        cluster.

    Returns
    -------
    dict
        Nested dictionary mapping cluster → cluster → count/probability.
    """
    all_ids = np.array(np.sum(df[id_column_name])).flatten()
    unique_ids = np.unique(all_ids)

    corr_dict = {
        key: {key: 0 for key in unique_ids} for key in unique_ids
    }

    for baseline_id in tqdm(unique_ids, desc="Building correlation dict."):
        for compared_id in unique_ids:
            for row in df[id_column_name]:
                if (baseline_id in row) and (compared_id in row):
                    corr_dict[baseline_id][compared_id] += 1

        if normalize:
            baseline_value = corr_dict[baseline_id][baseline_id]
            for compared_id in unique_ids:
                try:
                    corr_dict[baseline_id][compared_id] /= baseline_value
                except ZeroDivisionError:
                    corr_dict[baseline_id][compared_id] = 0

    return corr_dict


def visualize_correlation_dictionary(
    corr_matrix: dict,
    save_directory: str = None,
):
    """
    Visualize a correlation dictionary as a matrix heatmap.

    Parameters
    ----------
    corr_matrix : dict
        Nested dictionary of cluster → cluster → value.

    save_directory : str, optional
        Path (directory + filename) to save figure. If None, figure
        is shown instead.
    """
    shape = (len(corr_matrix), len(corr_matrix))
    grid = np.zeros(shape)

    keys = list(corr_matrix.keys())

    for i in range(len(grid)):
        for j in range(len(grid)):
            grid[j][i] = corr_matrix[keys[i]][keys[j]]

    plt.rcParams["figure.figsize"] = (9, 7)
    plt.imshow(grid, origin="lower", interpolation="none", cmap="inferno")
    plt.colorbar(label="Probability of Co-Activity")

    plt.title("Correlation Matrix for Neuron Clusters")
    plt.xlabel("Cluster ID (baseline)")
    plt.ylabel("Cluster ID (compared)")

    plt.xticks(range(len(keys)), labels=keys, rotation=90, fontsize=7)
    plt.yticks(range(len(keys)), labels=keys, fontsize=7)

    if save_directory:
        plt.savefig(save_directory, bbox_inches="tight")
        plt.close()
        plt.clf()
    else:
        plt.show()


def make_spike_df(
    time_dir: str,
    cluster_dir: str,
    label_dir: str,
):
    """
    Load spike times, cluster IDs, and KS labels into a DataFrame.

    Parameters
    ----------
    time_dir :: str
        Path to spike_times.npy file.
    cluster_dir :: str
        Path to spike_clusters.npy file.
    label_dir :: str
        Path to cluster_KSLabel.tsv file.

    Returns
    -------
    spike_df :: pd.DataFrame
        DataFrame with columns: Time, Cluster ID, KSLabel.
        filtered for "good" clusters
    """
    try:
        spike_times = np.load(time_dir).flatten() / 30000  # convert to seconds
    except FileNotFoundError:
        print(f"File not found: {time_dir}")
        sys.exit(1)

    try:
        spike_clusters = np.load(cluster_dir).flatten()
    except FileNotFoundError:
        print(f"File not found: {cluster_dir}")
        sys.exit(1)

    try:
        cluster_labels = pd.read_csv(label_dir, sep="\t")
    except FileNotFoundError:
        print(f"File not found: {label_dir}")
        sys.exit(1)

    # Standardize column names
    cluster_labels = cluster_labels.rename(
        columns={"cluster_id": "Cluster ID"}
    )

    # Merge to add KSLabel to every spike event
    spike_df = pd.DataFrame(
        {"Time": spike_times, "Cluster ID": spike_clusters}
    )
    spike_df = spike_df.merge(cluster_labels, on="Cluster ID", how="left")

    # Filter for good clusters only
    spike_df = spike_df[spike_df["KSLabel"] == "good"]

    return spike_df


def match_up(df,
             swr_dir=None,
             swr_df=None,
             only_keep_good=True,
             progress=True):
    """
    Assign spike times and cluster IDs to each SWR interval.

    One of:
      • swr_dir — path to SWR CSV
      • swr_df  — already-loaded DataFrame

    Parameters
    ----------
    df :: pd.DataFrame
        DataFrame of spike times and cluster IDs.
    swr_dir :: str
        Path to SWR CSV file.
    swr_df :: pd.DataFrame
        DataFrame of SWR events.
    only_keep_good :: bool
        If True, only consider spikes from clusters labeled "good".
    progress :: bool
        If True, show progress bar.

    Returns
    -------
    swr_df :: pd.DataFrame
        DataFrame of SWR events with assigned spike times and
        cluster IDs.
    """
    if swr_df is None and swr_dir is None:
        raise ValueError("Provide either swr_dir or swr_df")

    spike_df = df.copy()

    if swr_df is None:
        try:
            swr_df = pd.read_csv(swr_dir)
        except FileNotFoundError:
            print(f"File not found: {swr_dir}")
            sys.exit(1)
    else:
        swr_df = swr_df.copy()

    if only_keep_good:
        spike_df = spike_df[spike_df["KSLabel"] == "good"]

    swr_df["Spike Times (s)"] = None
    swr_df["Cluster IDs"] = None
    swr_df = swr_df.astype(
        {"Spike Times (s)": "object", "Cluster IDs": "object"}
    )

    spike_times = np.array(spike_df["Time"])
    cluster_ids = np.array(spike_df["Cluster ID"])

    for idx in tqdm(
        range(len(swr_df)),
        desc="Checking SWR Data",
        disable=not progress,
    ):
        mask = (
            (swr_df["Start"][idx] <= spike_times) &
            (spike_times <= swr_df["Stop"][idx])
        )
        swr_df.at[idx, "Spike Times (s)"] = list(spike_times[mask])
        swr_df.at[idx, "Cluster IDs"] = list(cluster_ids[mask])

    return swr_df


def count_spikes(swr_df: pd.DataFrame, mode="all"):
    """
    Count spikes per cluster across SWRs.
    Parameters
    ----------
    swr_df :: pd.DataFrame
        DataFrame of SWR events with assigned spikes.
    mode :: str
        mode="first": count only the first spike per SWR.
        mode="all": count all spikes.

    Returns
    -------
    counts :: dict
        Dictionary mapping cluster ID to spike count.
    """
    if mode not in ("first", "all"):
        raise ValueError("mode must be 'first' or 'all'")

    counts = {}

    for _, row in swr_df.iterrows():
        times = row["Spike Times (s)"]
        clusters = row["Cluster IDs"]

        if times is None or len(times) == 0:
            continue

        times = np.array(times, dtype=float)
        clusters = np.array(clusters, dtype=float)

        if mode == "first":
            first_cluster = clusters[np.argmin(times)]
            counts[first_cluster] = counts.get(first_cluster, 0) + 1

        else:
            unique, freqs = np.unique(clusters, return_counts=True)
            for cid, f in zip(unique, freqs):
                counts[cid] = counts.get(cid, 0) + f

    return counts


def compute_mean_firing_rates(spike_df: pd.DataFrame,
                              total_duration: float = None):
    """
    Compute mean firing rate (Hz) for each cluster.

    Parameters
    ----------
    spike_df :: pd.DataFrame
        DataFrame of spike times and cluster IDs.
    total_duration :: float
        Total recording duration (seconds). If None, inferred from
        spike_df.

    Returns
    -------
    firing_rates :: dict
        Dictionary mapping cluster ID to mean firing rate (Hz).
    """
    if total_duration is None:
        total_duration = spike_df["Time"].max()

    counts = spike_df["Cluster ID"].value_counts()
    firing_rates = counts / total_duration

    return firing_rates.to_dict()


def circular_permutation_test_with_firing_rate(
    swr_df: pd.DataFrame,
    spike_df: pd.DataFrame,
    tail: str = "two",
    total_duration: float = None,
    n_permutations: int = 1000,
    shift_range_seconds: float = 3.0,
    progress: bool = True,
    save_path: str = None,
    mode: str = "all",
):
    """
    Perform circular permutation tests of SWR windows with firing-rate
    normalization.
    Parameters
    ----------
    swr_df :: pd.DataFrame
        DataFrame of SWR events with assigned spikes.
    spike_df :: pd.DataFrame
        DataFrame of spike times and cluster IDs.
    tail :: str
        "one" for one-tailed test, "two" for two-tailed test.
    total_duration :: float
        Total recording duration (seconds). If None, inferred from
        spike_df.
    n_permutations :: int
        Number of permutations to run.
    shift_range_seconds :: float
        Maximum absolute time shift (seconds) for circular permutation.
    progress :: bool
        If True, show progress bar.
    save_path :: str
        If provided, save results DataFrame to this path.
    mode :: str
        "first" to count only first spike per SWR, "all" to count all
        spikes.
    Returns
    -------
    pd.DataFrame:
        results DataFrame, with columns:
        Cluster ID
        Firing Rate (Hz)
        True Count
        Mean Permuted
        Normalized True
        Normalized Mean
        Z-Score
        p-value (one- or two-tailed)
    true counts :: dict
        true observed counts
    all_permuted_counts :: dict
        permuted counts
    """
    if total_duration is None:
        total_duration = spike_df["Time"].max()

    firing_rates = compute_mean_firing_rates(spike_df, total_duration)
    true_counts = count_spikes(swr_df, mode)

    all_permuted_counts = {cid: [] for cid in true_counts.keys()}

    lower_bound = -shift_range_seconds
    upper_bound = shift_range_seconds

    shifts = np.random.uniform(
        low=lower_bound,
        high=upper_bound,
        size=n_permutations,
    )

    for shift in tqdm(
        shifts,
        desc="Running permutations",
        disable=not progress,
    ):
        shifted_swr = swr_df.copy()
        shifted_swr["Start"] = (
            shifted_swr["Start"] + shift
        ) % total_duration
        shifted_swr["Stop"] = (
            shifted_swr["Stop"] + shift
        ) % total_duration

        shifted_swr = match_up(
            spike_df,
            swr_df=shifted_swr,
            only_keep_good=False,
            progress=False,
        )
        perm_counts = count_spikes(shifted_swr)

        for cid in all_permuted_counts.keys():
            all_permuted_counts[cid].append(
                perm_counts.get(cid, 0)
            )

    results = []

    for cid, true_val in true_counts.items():
        perm_vals = np.array(all_permuted_counts[cid])
        n_perm = len(perm_vals)

        mean_perm = perm_vals.mean()
        std_perm = perm_vals.std(ddof=1) + 1e-6
        rate = firing_rates.get(cid, np.nan)

        if tail == "one":
            p_value = (
                np.sum(perm_vals >= true_val) + 1
            ) / (n_perm + 1)

        elif tail == "two":
            p_value = (
                np.sum(
                    np.abs(perm_vals - mean_perm) >=
                    np.abs(true_val - mean_perm)
                ) + 1
            ) / (n_perm + 1)

        else:
            raise ValueError(
                "Parameter 'tail' must be 'one' or 'two'."
            )

        normalized_true = (
            true_val / rate if rate > 0 else np.nan
        )
        normalized_mean = (
            mean_perm / rate if rate > 0 else np.nan
        )
        z_score = (
            (true_val - mean_perm) / std_perm
            if len(perm_vals) > 1 else np.nan
        )

        results.append(
            {
                "Cluster ID": cid,
                "Firing Rate (Hz)": rate,
                "True Count": true_val,
                "Mean Permuted": mean_perm,
                "Normalized True": normalized_true,
                "Normalized Mean": normalized_mean,
                "Z-Score": z_score,
                f"p-value ({tail}-tailed)": p_value,
            }
        )

        result_df = pd.DataFrame(results)

        if save_path is not None:
            os.makedirs(
                os.path.dirname(save_path), exist_ok=True
            )
            result_df.to_csv(save_path, index=False)

    print(f"\n Results saved to: {save_path}")

    return result_df, true_counts, all_permuted_counts


def plot_permutation_histogram(
    cid: int,
    all_permuted_counts,
    true_counts,
    binsize=20,
    two_tailed=True,
):
    """
    Plot permutation distribution and overlay the observed spike count.
    Parameters
    ----------
    cid :: int
        Cluster ID to plot.
    all_permuted_counts :: dict
        Dictionary mapping cluster ID to list of permuted counts.
    true_counts :: dict
        Dictionary mapping cluster ID to true observed count.
    binsize :: int
        Number of bins in the histogram.
    two_tailed :: bool
        If True, compute two-tailed p-value; else one-tailed.

    Returns
    -------
    Plot histogram and print p-value and z-score.
    """
    perm_vals = np.array(all_permuted_counts[cid])
    true_val = true_counts[cid]
    mean_perm = np.mean(perm_vals)

    if two_tailed:
        diff_true = abs(true_val - mean_perm)
        diff_perm = abs(perm_vals - mean_perm)
        p_value = (
            np.sum(diff_perm >= diff_true) + 1
        ) / (len(perm_vals) + 1)
    else:
        p_value = (
            np.sum(perm_vals >= true_val) + 1
        ) / (len(perm_vals) + 1)

    std_perm = perm_vals.std(ddof=1) + 1e-6
    z_score = (true_val - mean_perm) / std_perm

    plt.figure(figsize=(6, 4))
    plt.hist(
        perm_vals,
        bins=binsize,
        color="lightblue",
        edgecolor="black",
    )
    plt.axvline(
        true_val,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"True = {true_val}",
    )
    plt.title(
        f"Cluster {cid} | "
        f"{'Two' if two_tailed else 'One'}-tailed "
        f"p = {p_value:.3f}, z = {z_score:.2f}"
    )
    plt.xlabel("Spike count (permutation)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(
        f"Cluster {cid}: true = {true_val}, "
        f"mean_perm = {mean_perm:.2f}, "
        f"p = {p_value:.3f}, "
        f"z = {z_score:.3f}"
    )
