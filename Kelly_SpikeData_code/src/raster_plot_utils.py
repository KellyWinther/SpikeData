import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def prep_raster(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Prepare long-form spike ripple aligned data for raster plotting.
    Works directly on the output of match_times(), which includes:
        - 'Peak'
        - 'Event Times'
        - 'Event Cluster IDs'

    Adds:
        - ripple_idx  (for selecting ripples later)
    """

    if df is None:
        raise ValueError("No DataFrame provided.")

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df)}.")

    required = ["Peak", "Event Times", "Event Cluster IDs"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input dataframe is missing columns: {missing}")

    # Add ripple index before exploding
    df = df.copy()
    df["ripple_idx"] = df.index

    # Rename match_times output to standard names
    df = df.rename(columns={
        "Event Times": "Spike Times (s)",
        "Event Cluster IDs": "Cluster IDs"
    })

    # Convert stringified lists if needed
    for col in ["Spike Times (s)", "Cluster IDs"]:
        if isinstance(df[col].iloc[0], str):
            from ast import literal_eval
            df[col] = df[col].apply(literal_eval)

    # Expand each spike entry
    exp_df = df.explode(["Spike Times (s)", "Cluster IDs"], ignore_index=True)
    exp_df = exp_df.dropna(subset=["Spike Times (s)", "Cluster IDs"])

    # Compute time relative to peak
    exp_df["t_rel"] = exp_df["Spike Times (s)"] - exp_df["Peak"]

    print(f"prep_raster: {len(exp_df)} spikes from {df.shape[0]} ripple(s)")

    return exp_df


def select_ripples_to_plot(
    exp_df: pd.DataFrame,
    ripple_index: int = None,
) -> pd.DataFrame:
    """
    Select specific ripples from the exploded raster dataframe
    for making a simple raster plot

    Parameters
    ----------
    exp_df : pandas.DataFrame
        The exploded dataframe produced by prep_raster()

    ripple_index : None, int, or list-like
        None = return all ripples
        int = return that ripple
        list = return multiple ripples
    """
    if ripple_index is None:
        return exp_df.copy()

    # Normalize input type
    if isinstance(ripple_index, int):
        ripple_index = [ripple_index]

    if not hasattr(ripple_index, "__iter__"):
        raise TypeError(
            "ripple_index must be None, an int, or an iterable of ints.")

    ripple_index = list(ripple_index)

    if any(not isinstance(i, int) for i in ripple_index):
        raise ValueError("All ripple indices must be integers.")

    # Ensure indices exist in the dataset
    valid_indices = set(exp_df["ripple_idx"].unique())
    invalid = [i for i in ripple_index if i not in valid_indices]

    if invalid:
        raise ValueError(
            f"Invalid ripple index/indices {invalid}. "
            f"Choose between: {min(valid_indices)} and {max(valid_indices)}."
        )

    # Subset the data for ripple_index
    sub = exp_df[exp_df["ripple_idx"].isin(ripple_index)].copy()

    return sub


def plot_raster(
    exp_df: pd.DataFrame,
    height: float = 7.0,
    width: float = 9.0,
    color: str = "black",
    tick_width: float = 10.0,
    window: float = 0.1,
    ripple_index: int = None,
    save_path: str = None
):
    """
    Plot a spike raster aligned to ripple peaks, using evenly spaced rows
    for each cluster while labeling the y-axis with the actual cluster IDs.

    Parameters
    ----------
    exp_df : pandas.DataFrame
        Long-form exploded DataFrame produced by prep_raster().
        Must contain:
            - 't_rel' : float
                Spike time relative to ripple peak (seconds).
            - 'Cluster IDs' : int
                Cluster identity for each spike event.
            - 'ripple_idx' : int
                Index of the ripple each spike belongs to.

    height : float, optional (default=9)
        Height of the matplotlib figure (in inches).

    width : float, optional (default=6)
        Width of the matplotlib figure (in inches).

    color : str, optional (default="black")
        Color of spike markers in the raster plot.

    tick_width : float, optional (default=3)
        Marker size for each spike event.

    window : float, optional (default=0.25)
        Time window (in seconds) used for filtering spikes.
        Spikes with -window <= t_rel <= window will be plotted.

        Example:
            window=0.25 → shows spikes within ±250 ms of ripple peak.

    ripple_index : None, int, or list of int, optional (default=None)
        Controls which ripple(s) the title refers to.
            - None:
                Title shows "All SWR Spiking Activity..."
                (exp_df is assumed already filtered if selection was applied)
            - int:
                Title reflects a single ripple index.
            - list of int:
                Title shows all selected ripple indices.

    Notes
    -----
    - Y-axis spacing is categorical:
        cluster IDs are mapped to evenly spaced row positions (0,1,2,...).
    - Y-axis tick labels show the actual cluster IDs from the recording.
    - The function does not modify exp_df.
    """

    if "t_rel" not in exp_df.columns:
        raise ValueError("Expected 't_rel' column. Run prep_raster() first.")

    # Filter spike events to the chosen time window
    mask = (exp_df["t_rel"] >= -window) & (exp_df["t_rel"] <= window)
    raster_df = exp_df.loc[mask].copy()

    # Identify unique clusters present in this subset
    # clusters = sorted(raster_df["Cluster IDs"].unique())

    spike_counts = raster_df["Cluster IDs"].value_counts().sort_values()
    clusters = spike_counts.index.tolist()

    # Map each cluster ID to an evenly spaced row index
    cluster_to_row = {cid: i for i, cid in enumerate(clusters)}
    raster_df["cluster_row"] = raster_df["Cluster IDs"].map(cluster_to_row)

    # Title Formatting
    if ripple_index is None:
        title = f"All SWR Spiking Activity at Peak (±{window*1000:.0f} ms)"
    else:
        # Convert int → list
        if isinstance(ripple_index, int):
            ripple_index = [ripple_index]

        ripple_str = ", ".join(str(i) for i in ripple_index)
        title = (
            f"Ripple {ripple_str} Spiking Activity at Peak "
            f"(±{window*1000:.0f} ms)"
        )

    # Create the figure
    _, ax = plt.subplots(figsize=(width, height))

    # Vertical line marking ripple peak (t_rel = 0)
    ax.axvline(
        0,
        color="red",
        linestyle="--",
        linewidth=1,
        zorder=0
        )

    # Scatter plot of spikes
    ax.scatter(
        raster_df["t_rel"],
        raster_df["cluster_row"],
        marker="|",
        s=tick_width,
        color=color,
        alpha=0.7,
        zorder=2
    )

    # Axis formatting
    ax.set_xlim(-window, window)
    ax.set_xlabel("Time (s) relative to ripple peak")
    ax.set_ylabel("Cluster ID")
    ax.set_title(title)

    # Y-axis tick labeling (evenly spaced rows labeled by cluster ID)
    ax.set_yticks(range(len(clusters)))
    ax.set_yticklabels(clusters)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
