"""
sequence_matching_utils.py
==========================
Detects neuron-firing *replay*.

Scientific question
-------------------
While the vole is awake and interacting socially, its neurons fire in some
particular order — call that sequence S_awake. Later, during a sharp-wave
ripple (SWR) in rest/sleep, neurons fire again. We want to know:

    Does the awake sequence reappear inside the ripple?

We operationalize "reappear" as: the longest contiguous run of cluster IDs
shared between the two sequences, normalized by the shorter list's length.
If that ratio is high, lots of the awake firing order is being replayed.

Algorithm
---------
Naive longest-common-substring on two integer lists is O(n·m) memory.
We use Rabin–Karp rolling hashes + binary search on the answer length L:

    for each candidate L (binary-searched in [0, n_shorter]):
        compute all length-L hashes from list 1 → put in a set
        scan list 2 length-L hashes → check membership in the set
        if any hits → L is achievable, try larger; else try smaller

Each step is O(n), and binary search adds a log(n) factor. Far cheaper
than dynamic programming for the sizes we deal with.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import sys

# We reuse the data loaders rather than duplicate them. This keeps both
# modules in sync if the data format ever changes.
from loading_utils import load_spike_data, filter_dataframe


# ===========================================================================
# Private helpers (leading underscore = "implementation detail; don't
# import me from outside the module")
# ===========================================================================

def _preprocess_lists(
    l1: list,
    l2: list,
) -> tuple:
    """
    Two jobs:
      1. Coerce every element to Python int (hashing math is brittle on
         numpy.int64 mixed with Python ints).
      2. Swap so the SHORTER list is first. Several routines below assume
         |l1| <= |l2|; enforcing it once at the entry simplifies the rest.
    """
    l1 = list(map(int, l1))
    l2 = list(map(int, l2))
    return (l1, l2) if len(l1) <= len(l2) else (l2, l1)


def _compute_pow_base(
    n: int,
    base: int,
    mod: int,
) -> list:
    """
    Precompute base^i mod `mod` for i in [0, n].

    Rolling-hash math requires base^L on the fly; computing it inside the
    inner loop would dominate runtime. Caching a table is a classic trick.
    """
    pow_base = [1] * (n + 1)
    for i in range(1, n + 1):
        # Multiply previous power by base; mod keeps numbers bounded.
        pow_base[i] = (pow_base[i - 1] * base) % mod
    return pow_base


def _prefix_hash(
    arr: list,
    base: int,
    mod: int,
) -> list:
    """
    Compute the polynomial prefix-hash array.

    Definition: H[i] is the hash of arr[0:i]. With this array we can pull
    the hash of any sub-array arr[i:i+L] in O(1) (see _get_hash).

    Recurrence:  H[i+1] = (H[i] * base + (value + 1)) % mod
    The (+1) shifts values so that a zero element doesn't collapse into
    the empty-string hash.
    """
    H = [0] * (len(arr) + 1)
    for i, v in enumerate(arr):
        H[i + 1] = (H[i] * base + (v + 1)) % mod
    return H


def _get_hash(
    H: list,
    i: int,
    L: int,
    pow_base: list,
    mod: int,
) -> int:
    """
    Hash of arr[i : i+L] in O(1) using the prefix table.

    Derivation: H[i+L] = H[i] * base^L + hash(arr[i:i+L]), so
                hash(arr[i:i+L]) = H[i+L] - H[i] * base^L  (all mod p).
    """
    return (H[i + L] - (H[i] * pow_base[L]) % mod) % mod


def _has_match(
    L: int,
    n1: int,
    n2: int,
    H1: list,
    H2: list,
    pow_base: list,
    mod: int,
) -> bool:
    """
    "Is there ANY contiguous match of length L common to both lists?"

    Strategy:
      - Bag every length-L hash from list 1 into a set.
      - Scan list 2's length-L hashes; if any is in the set → match.
    O(n1 + n2). Constant collision risk because we use a 61-bit Mersenne
    prime modulus — empirically negligible for our small lists.
    """
    if L == 0:
        return True            # empty match trivially exists
    if L > n1:
        return False           # can't fit length L in the shorter list

    # All length-L window hashes from list 1, deduplicated.
    seen = {
        _get_hash(H1, i, L, pow_base, mod)
        for i in range(n1 - L + 1)
    }

    # Scan list 2 for any window whose hash is in the set.
    for j in range(n2 - L + 1):
        if _get_hash(H2, j, L, pow_base, mod) in seen:
            return True

    return False


def _recover_sequence(
    L: int,
    l1: list,
    l2: list,
    H1: list,
    H2: list,
    n1: int,
    n2: int,
    pow_base: list,
    mod: int,
) -> list:
    """
    Same scan as _has_match, but when we find a hash collision we double-
    check element-by-element and return the actual matching sub-list.

    Verifying with `l1[i:i+L] == l2[j:j+L]` rules out the (astronomically
    rare) hash collision case where two different sequences happen to
    share a hash.
    """
    if L == 0:
        return []

    # Build a hash → list-of-starting-indices map for list 1, so we can
    # check every candidate start that matches a given hash.
    seen = {}
    for i in range(n1 - L + 1):
        h = _get_hash(H1, i, L, pow_base, mod)
        seen.setdefault(h, []).append(i)

    # Walk list 2; on a hash hit verify by equality and return.
    for j in range(n2 - L + 1):
        h = _get_hash(H2, j, L, pow_base, mod)
        if h in seen:
            for i in seen[h]:
                if l1[i:i + L] == l2[j:j + L]:
                    return l1[i:i + L]

    return []  # only reached if every hash hit was a collision (~0% chance)


# ===========================================================================
# Public API
# ===========================================================================

def lccs_rabin_karp(
    l1: list,
    l2: list,
    normalize: bool = True,
    return_sequence: bool = False
) -> tuple[int | float, list]:
    """
    Longest common contiguous subsequence (LCCS), in O((n1+n2) log n1)
    expected time, via binary search on the answer length.
    """

    # Coerce and ensure |l1| <= |l2| so subsequent length checks are easy.
    l1, l2 = _preprocess_lists(l1, l2)
    n1, n2 = len(l1), len(l2)

    # Edge case: shorter list is empty → trivial answer.
    if n1 == 0:
        return (0, []) if return_sequence else 0

    # Hash parameters.
    #   base = 257   → larger than any single byte; collision risk low.
    #   mod  = 2^61 − 1 → Mersenne prime; fast modular multiplication and
    #                     huge collision space.
    base = 257
    mod = (1 << 61) - 1

    pow_base = _compute_pow_base(n1, base, mod)
    H1 = _prefix_hash(l1, base, mod)
    H2 = _prefix_hash(l2, base, mod)

    # Binary search the maximum L for which a match still exists.
    # Invariant: lo = currently achievable, hi = current upper bound.
    lo, hi = 0, n1
    while lo < hi:
        mid = (lo + hi + 1) // 2     # upper midpoint avoids infinite loop
        if _has_match(mid, n1, n2, H1, H2, pow_base, mod):
            lo = mid                  # can do at least mid → search higher
        else:
            hi = mid - 1              # mid impossible → search lower

    L = lo

    # If the caller only wants the LENGTH, we can return now.
    if not return_sequence:
        return L / n1 if normalize else L

    # Otherwise, recover and return the actual matching sequence too.
    seq = _recover_sequence(L, l1, l2, H1, H2, n1, n2, pow_base, mod)
    length_out = L / n1 if normalize else L
    return length_out, seq


# ===========================================================================
# Path / data helpers (used by example_sequence_matching.py)
# ===========================================================================

def get_base_directory(individual):
    """
    Test data and real data live in different folders. This little helper
    keeps the path-building functions below from sprinkling if/else
    everywhere.
    """
    if individual == "test":
        return "data/test_data"
    return "data/full_data"


def build_file_paths(individual, stimuli):
    """
    Construct the dictionary of file paths for a given animal/session.

    The real datasets have an inconsistent naming convention: animal 7742's
    files include `_sleepyvole_` in the filename, while 7744's do not. We
    encode that exception with `extra_tag`. (See the data-folder README
    for the historical reason.)
    """
    base = get_base_directory(individual)

    # Test fixtures use a flat layout — short names, no animal subdir.
    if individual == "test":
        root = base
        return {
            "spike_times":    os.path.join(root, "test_spike_times.npy"),
            "spike_clusters": os.path.join(root, "test_spike_clusters.npy"),
            "kslabels":       os.path.join(root, "test_cluster_KSLabel.tsv"),
            "behavior_csv":   os.path.join(root, "test_events_with_indices.csv"),
            "swr_csv":        os.path.join(root, "test_SWRs_ca2.csv"),
        }

    # Real data — nested under <animal>/<stimuli>/
    extra_tag = "_sleepyvole" if individual == "7742" else ""
    root = os.path.join(base, individual, stimuli)

    return {
        "spike_times":    os.path.join(root, "spike_times.npy"),
        "spike_clusters": os.path.join(root, "spike_clusters.npy"),
        "kslabels":       os.path.join(root, "cluster_KSLabel.tsv"),
        "behavior_csv":   os.path.join(
            root, f"{individual}_{stimuli}{extra_tag}_events_with_indices.csv"
        ),
        "swr_csv":        os.path.join(
            root, f"{individual}_{stimuli}{extra_tag}_SWRs_ca2.csv"
        ),
    }


def load_data(paths):
    """
    Load all three CSV/NPY families for a dataset and return them as
    (spike_df, behavior_df, swr_df). All sys.exit on missing files so
    downstream code never receives a partial dataset.
    """
    # --- spikes (uses the shared loader, then filters to 'good') ----------
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
    # Reset index so later positional operations are predictable.
    spike_df = spike_df.reset_index(drop=True)

    # --- behavior log -----------------------------------------------------
    try:
        behavior_df = pd.read_csv(paths["behavior_csv"])
    except FileNotFoundError:
        print("Error: Behavior CSV not found.")
        sys.exit(1)

    # --- SWR table --------------------------------------------------------
    try:
        swr_df = pd.read_csv(paths["swr_csv"])
    except FileNotFoundError:
        print("Error: SWR CSV not found.")
        sys.exit(1)

    return spike_df, behavior_df, swr_df


def preprocess_behavior_df(
    df,
    event_types=("social interaction", "cup interaction"),
):
    """
    Clean up the BORIS behavior-log CSV so its time columns line up with
    spike times.

    Two transformations:
      1. Keep only the requested behavior events. (BORIS logs many other
         categories we don't care about here.) By default we keep both
         'social interaction' and 'cup interaction' so this stays
         backward-compatible with example_sequence_matching.py. Pass a
         single-element list/tuple (e.g. ("cup interaction",)) to isolate
         one condition — this is how the awake social-vs-cup comparison
         pulls each condition separately.
      2. Convert indexStart/indexEnd from camera samples → seconds.
         The camera sampled at 2500 Hz, so divide by 2500. Also rename
         these to Start/Stop so downstream code can treat behavior bouts
         and SWRs interchangeably as "time windows".
    """
    # `list(event_types)` so callers may pass a tuple (immutable default)
    # or a list interchangeably; filter_dataframe expects a list value.
    df = filter_dataframe(
        df,
        {"EventType": list(event_types)},
    )
    df = df.reset_index(drop=True)

    # Vectorized divide on two columns at once.
    df[["indexStart", "indexEnd"]] /= 2500.0

    df = df.rename(columns={"indexStart": "Start", "indexEnd": "Stop"})
    return df


def compute_overlap_matrix(behavior_lists, swr_lists):
    """
    For every (SWR, behavior) pair compute the normalized LCCS and fill in
    an overlap matrix. Rows = SWRs; columns = behavior windows.

    Optimization: if the two cluster sets are disjoint (`bset.isdisjoint`),
    no contiguous match can possibly exist, so skip the expensive Rabin–
    Karp call entirely. This is a huge win because most pairs have zero
    overlap.
    """
    overlap = np.zeros((len(swr_lists), len(behavior_lists)))

    for bx, bseq in enumerate(tqdm(behavior_lists)):
        if not bseq:
            continue                        # empty behavior list → zeros

        bset = set(bseq)                    # used for fast disjoint test

        for sy, sseq in enumerate(swr_lists):
            if not sseq:
                continue                    # empty SWR list → zero
            if bset.isdisjoint(sseq):
                continue                    # no shared clusters → zero

            overlap[sy, bx] = lccs_rabin_karp(sseq, bseq, normalize=True)

    return overlap


def plot_overlap_matrix(matrix):
    """
    Heatmap of the (SWR × behavior) overlap matrix. Bright pixels mean
    a SWR replays a long contiguous chunk of a behavior bout's firing
    sequence.
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


# ===========================================================================
# Awake analysis helpers
# ---------------------------------------------------------------------------
# The functions below are the *minimal* additions needed to reuse the original
# SWR pipeline for an awake, behavior-conditioned comparison (social vs cup)
# instead of the original sleep-SWR-vs-behavior overlap. Two ideas:
#
#   1. filter_swrs_by_behavior() — keep only the SWRs that occur *inside*
#      an awake behavior bout. This is the literal "awake SWRs" subset.
#      (On the provided datasets this returns ~0 SWRs, because every
#      detected ripple falls during sleep — see build_behavior_windows for
#      the alternative that actually yields awake firing data.)
#
#   2. build_behavior_windows() — turn the awake behavior bouts themselves
#      into "window" rows shaped exactly like an SWR table (Start/Stop/Peak),
#      so prep_raster / plot_raster / make_correlation_dictionary can run on
#      them unchanged. Here "Peak" is defined as the bout onset, so the
#      raster's t_rel becomes "time since the interaction began".
# ===========================================================================

def build_behavior_windows(
    behavior_df,
    event_types,
    peak="start",
):
    """
    Convert preprocessed behavior bouts into an SWR-shaped window table.

    The original pipeline aligns spikes to a ripple's 'Peak'. Awake behavior
    bouts have no ripple peak, so we synthesize one:
        peak="start"  → Peak = Start  (align to bout onset; t_rel = time
                        since the interaction began)
        peak="mid"    → Peak = midpoint of the bout

    Parameters
    ----------
    behavior_df : pandas.DataFrame
        Output of preprocess_behavior_df — already has numeric Start/Stop
        columns (seconds) and an EventType column.
    event_types : list/tuple of str
        Which behavior label(s) to keep (e.g. ("social interaction",)).
    peak : {"start", "mid"}
        How to define the alignment reference column.

    Returns
    -------
    pandas.DataFrame
        Copy of the filtered bouts with a guaranteed 'Peak' column, ready
        to hand to match_times-style grouping and prep_raster.
    """
    # Restrict to the requested condition(s). We reuse filter_dataframe for
    # identical semantics/warnings as the rest of the codebase.
    windows = filter_dataframe(behavior_df, {"EventType": list(event_types)})
    windows = windows.reset_index(drop=True).copy()

    # Synthesize the alignment reference the raster code expects.
    if peak == "start":
        windows["Peak"] = windows["Start"]
    elif peak == "mid":
        windows["Peak"] = (windows["Start"] + windows["Stop"]) / 2.0
    else:
        raise ValueError("peak must be 'start' or 'mid'")

    return windows


def filter_swrs_by_behavior(
    swr_df,
    behavior_df,
    event_types,
    reference_col="Peak",
):
    """
    Keep only SWRs whose reference time falls inside an awake behavior bout.

    This is the "awake SWRs" selector: given the full SWR table and the
    preprocessed behavior bouts, return the subset of ripples that happened
    while the animal was engaged in the requested behavior (social / cup).

    Parameters
    ----------
    swr_df : pandas.DataFrame
        Raw SWR table with Start/Stop/Peak columns (seconds).
    behavior_df : pandas.DataFrame
        Output of preprocess_behavior_df (numeric Start/Stop in seconds).
    event_types : list/tuple of str
        Behavior label(s) that define "awake" here.
    reference_col : str
        Which SWR column decides membership. 'Peak' (default) asks
        "did the ripple peak occur during the behavior?".

    Returns
    -------
    pandas.DataFrame
        The filtered SWR rows (index reset). May be empty if no ripple
        overlaps the behavior — that is itself a meaningful result.
    """
    # Bouts of interest, as parallel numpy arrays for fast masking.
    bouts = filter_dataframe(behavior_df, {"EventType": list(event_types)})
    starts = np.asarray(bouts["Start"], dtype=float)
    stops = np.asarray(bouts["Stop"], dtype=float)

    ref = np.asarray(swr_df[reference_col], dtype=float)

    # For each SWR, True if its reference time lies inside ANY bout. We OR
    # a per-bout mask across all bouts — vectorized over SWRs, looped over
    # the (few) bouts.
    keep = np.zeros(len(ref), dtype=bool)
    for s, e in zip(starts, stops):
        keep |= (ref >= s) & (ref <= e)

    return swr_df.loc[keep].reset_index(drop=True)


def summarize_firing_order(cluster_id_lists):
    """
    Quantify how *consistent* the firing order is within one set of windows.

    Given a list of per-window cluster-ID sequences (in spike-time order),
    compute the pairwise normalized longest-common-contiguous-subsequence
    (LCCS) between every pair of windows and return summary statistics.
    A higher mean means windows tend to replay the same neuron order.

    Reuses compute_overlap_matrix (same LCCS engine as the original
    SWR-vs-behavior analysis) by comparing the set against itself.

    Parameters
    ----------
    cluster_id_lists : list of list
        One cluster-ID sequence per window (e.g. the 'Event Cluster IDs'
        column of a grouped DataFrame, as a Python list).

    Returns
    -------
    dict
        {
          "n_windows": int,
          "mean_overlap": float,   # mean off-diagonal normalized LCCS
          "median_overlap": float,
          "matrix": np.ndarray,    # full pairwise matrix (diagonal = 1)
        }
    """
    n = len(cluster_id_lists)
    # Self-comparison: rows and columns are the same window set.
    matrix = compute_overlap_matrix(cluster_id_lists, cluster_id_lists)

    if n < 2:
        # Not enough windows to have any pair — return zeros gracefully.
        return {
            "n_windows": n,
            "mean_overlap": 0.0,
            "median_overlap": 0.0,
            "matrix": matrix,
        }

    # Exclude the diagonal (a window vs itself is trivially 1.0) when
    # summarizing, so the statistic reflects *between-window* consistency.
    off_diag = matrix[~np.eye(n, dtype=bool)]
    return {
        "n_windows": n,
        "mean_overlap": float(np.mean(off_diag)),
        "median_overlap": float(np.median(off_diag)),
        "matrix": matrix,
    }
