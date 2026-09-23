# Necessary imports
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
from ast import literal_eval


def time_in_range(
    t_start: float,
    t_end: float,
    t: float,
) -> bool:
    """
    Checks if a time lies between a given
    start and end time (inclusive).

    Parameters:
    -----------
    t_start :: float
        Time (seconds) marking the start of
        the window.
    t_end :: float
        Time (seconds) marking the stop of
        the window.
    t :: float
        Time (seconds) marking the time being
        compared to the provided window.

    Returns:
    --------
    in_range :: bool
        True iff t is contained within the window.
    """

    # NOTE: TypeError can be triggered by 'None'
    try:
        in_range = t_start <= t <= t_end
    except TypeError:
        print("Non-numeric entry provided, defaulting to 'False'")
        in_range = False

    return in_range


def load_spike_data(
    time_dir: str,
    cluster_dir: str,
    label_dir: str,
) -> pd.DataFrame:
    """
    Loads the time and clusters recorded
    for each spike and joins both datasets
    into a single Pandas DataFrame.

    Parameters:
    -----------
    time_dir :: str
        Path to time data (should include the file name)
    cluster_dir :: str
        Path to cluster ID data (should include the file name)
    label_dir :: str
        Path to the KSLabel data (should include the file name)

    Returns:
    --------
    spike_df :: pd.DataFrame
        Dataframe containing spike times / IDs
    """

    # Factor of 30,000 accounts for sampling rate
    try:
        spike_times = np.load(time_dir).flatten() / 30000
    except FileNotFoundError:
        print(f"Filename '{time_dir}' not found")
        sys.exit(1)

    # The '.flatten()' command fixes the data loading as a column
    try:
        spike_clusters = np.load(cluster_dir).flatten()
    except FileNotFoundError:
        print(f"Filename '{cluster_dir}' not found")
        sys.exit(1)

    try:
        cluster_labels = pd.read_csv(label_dir, sep="\t")
    except FileNotFoundError:
        print(f"Filename '{label_dir}' not found")
        sys.exit(1)

    # Loads cluster labels ('cluster_id' renamed to match later convention)
    cluster_labels = cluster_labels.rename(
        columns={"cluster_id": "Cluster ID"}
    )

    # Joins time, id, and label data into a single array
    spike_df = pd.DataFrame(
        {"Time": spike_times, "Cluster ID": spike_clusters}
    )
    spike_df = pd.merge(spike_df, cluster_labels, on="Cluster ID", how="left")

    return spike_df


def filter_dataframe(
    df: pd.DataFrame,
    filter_dictionary: dict,
) -> pd.DataFrame:
    """
    Attempts to filter down a DataFrame for a handful
    of columns with user-specified "allowed values".
    For example, if you wanted to filter a DataFrame
    based on the "KSLabel" column, the 'filter_dictionary'
    may look like...

        {"KSLabel": ["good",]}

    If an empty dictionary is provided, then no filter will
    be applied.

    Parameters:
    -----------
    df :: pd.DataFrame
        A DataFrame to filter down based on the value(s) of
        a specified column(s).
    filter_dictionary :: dict
        Specifies which column to filter by (should be a
        key in the dictionary) and what values should be
        kept (should be a list of entries).

    Returns:
    --------
    df :: pd.DataFrame
        The final, filtered DataFrame.
    """

    for column_name, valid_values in filter_dictionary.items():

        try:
            df = df[df[column_name].isin(valid_values)]

        # Should only trigger if a user-provided column is not in df
        except KeyError:
            print(
                f"Column '{column_name}' not found",
                "moving on without filtering.",
            )

        # Should only trigger if the value of filter dictionary is not a list
        except TypeError:
            print(
                f"Valid values must be a list, not '{type(valid_values)}'",
                "Moving on without filtering.",
            )

    return df


# def group_dataframes_by_time(
#     window_df: pd.DataFrame,
#     event_df: pd.DataFrame,
#     event_time_column: str,
#     keep_event_columns: list,
#     time_interval_columns: list,
#     progress: bool,
# ):
#     """
#     Checks if events contained in a user-provided DataFrame
#     contain times that fall within a window found in a
#     separate datafile. For consistency, the user-provided
#     DataFrame is referred to as the 'event dataframe' and
#     should only contain a single time per row. The directory
#     should point towards the 'window dataframe' which should
#     contain two times per row (a start and stop time).

#     Parameters:
#     -----------
#     window_df :: pd.DataFrame
#         Dataframe containing at least one column
#         labeled 'Time' (or the string assigned to
#         'event_time_column'; assumes units of seconds).
#         This function will look for time ranges that
#         these times fall between.
#     event_df :: pd.DataFrame
#         Dataframe containing at least one column
#         labeled 'Time' (or the string assigned to
#         'event_time_column'; assumes units of seconds).
#         This function will look for time ranges that
#         these times fall between.
#     filter_event_data :: dict
#         Indicates which values to filter by in the 'event
#         dataframe.' Every key-value pair should be a column
#         name in the 'event dataframe' (key) and a list of
#         valid entries (value). If no arguments are provided,
#         then no filtering is applied.
#     filter_window_data :: dict
#         Indicates which values to filter by in the 'window
#         dataframe.' Every key-value pair should be a column
#         name in the 'window dataframe' (key) and a list of
#         valid entries (value). If no arguments are provided,
#         then no filtering is applied.
#     event_time_column :: str
#         The name of a column in the 'event dataframe' containing
#         time data (assumes units of seconds).
#     time_interval_column :: list
#         The names of two columns in the 'window dataframe'
#         containing the start and stop (both in seconds) of
#         a valid window. Events with a time falling between these
#         two values will be associated with the respective window.
#     keep_event_columns :: list
#         A list of columns to keep from the 'event dataframe'. For
#         example, if provided ['A', 'B'], then the values of 'A' and
#         'B' for all matched events will be saved in a list in the
#         returned dataframe.
#     progress :: bool
#         Disables / enables progress bar.

#     Returns:
#     --------
#     grouped_df :: pd.DataFramed
#     """

#     # Prevents us from replacing initial DataFrame
#     grouped_df = window_df.copy()

#     # Renames columns to be grammatically correct
#     renamed_event_columns = [
#         f"Event {string}s" for string in keep_event_columns
#     ]

#     # Initialize new columns properly (object dtype) to prevent Pandas errors
#     for col in renamed_event_columns:
#         grouped_df[col] = None
#         grouped_df[col] = grouped_df[col].astype("object")

#     # Extract all relevant time data (units of seconds)
#     event_times = np.array(event_df[event_time_column])

#     for idx in tqdm(
#         range(len(grouped_df)),
#         desc="Checking Event Data",
#         disable=not progress,
#     ):

#         # Masking allows us to avoid excessive looping
#         start_time = np.array(grouped_df[time_interval_columns[0]])[idx]
#         end_time = np.array(grouped_df[time_interval_columns[1]])[idx]
#         mask = (start_time <= event_times) & (event_times <= end_time)

#         for column, renamed_column in zip(
#             keep_event_columns, renamed_event_columns
#         ):
#             data = list(event_df[column][mask])
#             grouped_df.at[idx, renamed_column] = data

#     return grouped_df


# def match_times(
#     dataframe: pd.DataFrame,
#     directory: str,
#     filter_event_data: dict = {},
#     filter_window_data: dict = {},
#     event_time_column: str = "Time",
#     time_interval_columns: list = ["Start", "Stop"],
#     keep_event_columns: list = [],
#     progress: bool = True,
# ) -> pd.DataFrame:
#     """
#     Checks if events contained in a user-provided DataFrame
#     contain times that fall within a window found in a
#     separate datafile. For consistency, the user-provided
#     DataFrame is referred to as the 'event dataframe' and
#     should only contain a single time per row. The directory
#     should point towards the 'window dataframe' which should
#     contain two times per row (a start and stop time).

#     Parameters:
#     -----------
#     dataframe :: pd.DataFrame
#         Dataframe containing at least one column
#         labeled 'Time' (or the string assigned to
#         'event_time_column'; assumes units of seconds).
#         This function will look for time ranges that
#         these times fall between.
#     directory :: str
#         Path (directory + filename) containing data with
#         times ranges (found in columns 'Start' and 'Stop'
#         by default).
#     filter_event_data :: dict
#         Indicates which values to filter by in the 'event
#         dataframe.' Every key-value pair should be a column
#         name in the 'event dataframe' (key) and a list of
#         valid entries (value). If no arguments are provided,
#         then no filtering is applied.
#     filter_window_data :: dict
#         Indicates which values to filter by in the 'window
#         dataframe.' Every key-value pair should be a column
#         name in the 'window dataframe' (key) and a list of
#         valid entries (value). If no arguments are provided,
#         then no filtering is applied.
#     event_time_column :: str
#         The name of a column in the 'event dataframe' containing
#         time data (assumes units of seconds).
#     time_interval_column :: list
#         The names of two columns in the 'window dataframe'
#         containing the start and stop (both in seconds) of
#         a valid window. Events with a time falling between these
#         two values will be associated with the respective window.
#     keep_event_columns :: list
#         A list of columns to keep from the 'event dataframe'. For
#         example, if provided ['A', 'B'], then the values of 'A' and
#         'B' for all matched events will be saved in a list in the
#         returned dataframe.
#     progress :: bool
#         Disables / enables progress bar.

#     Returns:
#     --------
#     return_df :: pd.DataFrame
#         A copy of the initial DataFrame, but with
#         two new columns containing the SWR start
#         and stop times. If no matching SWR data
#         was found, both columns should default
#         to NaN values.
#     """

#     # This prevents us from accidentally modifying the original DataFrame
#     try:
#         event_df = dataframe.copy()
#     except AttributeError:
#         print("Could not copy the provided DataFrame")
#         sys.exit(1)

#     try:
#         window_df = pd.read_csv(directory)
#     except FileNotFoundError:
#         assert FileNotFoundError(f"Filename '{directory}' not found")
#         sys.exit(1)

#     # If filter dictionary is empty, DataFrame is not modified
#     event_df = filter_dataframe(event_df, filter_event_data)
#     window_df = filter_dataframe(window_df, filter_window_data)

#     grouped_df = group_dataframes_by_time(
#         window_df=window_df,
#         event_df=event_df,
#         event_time_column=event_time_column,
#         keep_event_columns=keep_event_columns,
#         time_interval_columns=time_interval_columns,
#         progress=progress,
#     )

#     return grouped_df


## TEST this for adding unique cluster and sorted cluster columns to matched_times df

def group_dataframes_by_time(
    window_df: pd.DataFrame,
    event_df: pd.DataFrame,
    event_time_column: str,
    keep_event_columns: list,
    time_interval_columns: list,
    progress: bool,
    add_unique_sorted: bool = True,
):
    """
    Checks if events contained in a user-provided DataFrame
    contain times that fall within a window found in a
    separate datafile. For consistency, the user-provided
    DataFrame is referred to as the 'event dataframe' and
    should only contain a single time per row. The directory
    should point towards the 'window dataframe' which should
    contain two times per row (a start and stop time).

    Parameters:
    -----------
    window_df :: pd.DataFrame
        Dataframe containing at least one column
        labeled 'Time' (or the string assigned to
        'event_time_column'; assumes units of seconds).
        This function will look for time ranges that
        these times fall between.
    event_df :: pd.DataFrame
        Dataframe containing at least one column
        labeled 'Time' (or the string assigned to
        'event_time_column'; assumes units of seconds).
        This function will look for time ranges that
        these times fall between.
    filter_event_data :: dict
        Indicates which values to filter by in the 'event
        dataframe.' Every key-value pair should be a column
        name in the 'event dataframe' (key) and a list of
        valid entries (value). If no arguments are provided,
        then no filtering is applied.
    filter_window_data :: dict
        Indicates which values to filter by in the 'window
        dataframe.' Every key-value pair should be a column
        name in the 'window dataframe' (key) and a list of
        valid entries (value). If no arguments are provided,
        then no filtering is applied.
    event_time_column :: str
        The name of a column in the 'event dataframe' containing
        time data (assumes units of seconds).
    time_interval_column :: list
        The names of two columns in the 'window dataframe'
        containing the start and stop (both in seconds) of
        a valid window. Events with a time falling between these
        two values will be associated with the respective window.
    keep_event_columns :: list
        A list of columns to keep from the 'event dataframe'. For
        example, if provided ['A', 'B'], then the values of 'A' and
        'B' for all matched events will be saved in a list in the
        returned dataframe.
    progress :: bool
        Disables / enables progress bar.
    add_unique_sorted :: bool
        If True, adds 'Unique Event Xs' and 'Sorted Event Xs' columns
        for any column X that contains 'Cluster ID' in its name.
        Default is True.

    Returns:
    --------
    grouped_df :: pd.DataFramed
    """

    # Prevents us from replacing initial DataFrame
    grouped_df = window_df.copy()

    # Renames columns to be grammatically correct
    renamed_event_columns = [
        f"Event {string}s" for string in keep_event_columns
    ]

    # Initialize new columns properly (object dtype) to prevent Pandas errors
    for col in renamed_event_columns:
        grouped_df[col] = None
        grouped_df[col] = grouped_df[col].astype("object")

    # Extract all relevant time data (units of seconds)
    event_times = np.array(event_df[event_time_column])

    for idx in tqdm(
        range(len(grouped_df)),
        desc="Checking Event Data",
        disable=not progress,
    ):

        # Masking allows us to avoid excessive looping
        start_time = np.array(grouped_df[time_interval_columns[0]])[idx]
        end_time = np.array(grouped_df[time_interval_columns[1]])[idx]
        mask = (start_time <= event_times) & (event_times <= end_time)

        for column, renamed_column in zip(
            keep_event_columns, renamed_event_columns
        ):
            data = list(event_df[column][mask])
            grouped_df.at[idx, renamed_column] = data

    # Add unique and sorted versions for Cluster ID columns
    if add_unique_sorted:
        for col in renamed_event_columns:
            if "Cluster ID" in col:
                # Create unique column
                unique_col = col.replace("Event ", "Unique Event ")
                grouped_df[unique_col] = grouped_df[col].apply(_get_unique_preserving_order)
                
                # Create sorted column
                sorted_col = col.replace("Event ", "Sorted Event ")
                grouped_df[sorted_col] = grouped_df[unique_col].apply(lambda lst: sorted(lst) if lst else [])
                
                if progress:
                    print(f"Added '{unique_col}' and '{sorted_col}' columns")

    return grouped_df


def _get_unique_preserving_order(seq):
    """
    Helper function to get unique values from a sequence
    while preserving order.
    
    Parameters:
    -----------
    seq :: list or array-like
        Sequence to extract unique values from
        
    Returns:
    --------
    list : Unique values in order of first appearance
    """
    if not seq or seq is None:
        return []
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]


def match_times(
    dataframe: pd.DataFrame,
    directory: str,
    filter_event_data: dict = {},
    filter_window_data: dict = {},
    event_time_column: str = "Time",
    time_interval_columns: list = ["Start", "Stop"],
    keep_event_columns: list = [],
    progress: bool = True,
    add_unique_sorted: bool = True,
) -> pd.DataFrame:
    """
    Checks if events contained in a user-provided DataFrame
    contain times that fall within a window found in a
    separate datafile. For consistency, the user-provided
    DataFrame is referred to as the 'event dataframe' and
    should only contain a single time per row. The directory
    should point towards the 'window dataframe' which should
    contain two times per row (a start and stop time).

    Parameters:
    -----------
    dataframe :: pd.DataFrame
        Dataframe containing at least one column
        labeled 'Time' (or the string assigned to
        'event_time_column'; assumes units of seconds).
        This function will look for time ranges that
        these times fall between.
    directory :: str
        Path (directory + filename) containing data with
        times ranges (found in columns 'Start' and 'Stop'
        by default).
    filter_event_data :: dict
        Indicates which values to filter by in the 'event
        dataframe.' Every key-value pair should be a column
        name in the 'event dataframe' (key) and a list of
        valid entries (value). If no arguments are provided,
        then no filtering is applied.
    filter_window_data :: dict
        Indicates which values to filter by in the 'window
        dataframe.' Every key-value pair should be a column
        name in the 'window dataframe' (key) and a list of
        valid entries (value). If no arguments are provided,
        then no filtering is applied.
    event_time_column :: str
        The name of a column in the 'event dataframe' containing
        time data (assumes units of seconds).
    time_interval_column :: list
        The names of two columns in the 'window dataframe'
        containing the start and stop (both in seconds) of
        a valid window. Events with a time falling between these
        two values will be associated with the respective window.
    keep_event_columns :: list
        A list of columns to keep from the 'event dataframe'. For
        example, if provided ['A', 'B'], then the values of 'A' and
        'B' for all matched events will be saved in a list in the
        returned dataframe.
    progress :: bool
        Disables / enables progress bar.
    add_unique_sorted :: bool
        If True, adds 'Unique Event Xs' and 'Sorted Event Xs' columns
        for any column X that contains 'Cluster ID' in its name.
        Default is True.

    Returns:
    --------
    return_df :: pd.DataFrame
        A copy of the initial DataFrame, but with
        two new columns containing the SWR start
        and stop times. If no matching SWR data
        was found, both columns should default
        to NaN values.
    """

    # This prevents us from accidentally modifying the original DataFrame
    try:
        event_df = dataframe.copy()
    except AttributeError:
        print("Could not copy the provided DataFrame")
        sys.exit(1)

    try:
        window_df = pd.read_csv(directory)
    except FileNotFoundError:
        assert FileNotFoundError(f"Filename '{directory}' not found")
        sys.exit(1)

    # If filter dictionary is empty, DataFrame is not modified
    event_df = filter_dataframe(event_df, filter_event_data)
    window_df = filter_dataframe(window_df, filter_window_data)

    grouped_df = group_dataframes_by_time(
        window_df=window_df,
        event_df=event_df,
        event_time_column=event_time_column,
        keep_event_columns=keep_event_columns,
        time_interval_columns=time_interval_columns,
        progress=progress,
        add_unique_sorted=add_unique_sorted,
    )

    return grouped_df


def filter_swr_by_channels(
    matched_swr_df: pd.DataFrame,
    cluster_info_df: pd.DataFrame,
    channels: list,
    cluster_id_col: str = "cluster_id",
    channel_col: str = "ch",
    event_times_col: str = "Event Times",
    event_cluster_col: str = "Event Cluster IDs"
) -> pd.DataFrame:
    """
    Filter matched SWR data to only include spike times and cluster IDs
    for clusters that are on specific channels.
    
    Parameters:
    -----------
    matched_swr_df : pd.DataFrame
        DataFrame output from match_times() containing:
            - 'Event Times': list of spike times per SWR
            - 'Event Cluster IDs': list of cluster IDs per SWR
            
    cluster_info_df : pd.DataFrame
        DataFrame containing cluster information with at least:
            - cluster_id column (or as specified in cluster_id_col)
            - channel column (or as specified in channel_col)
            
    channels : list of int
        List of channel numbers to filter by
        
    cluster_id_col : str, optional (default="cluster_id")
        Name of the cluster ID column in cluster_info_df
        
    channel_col : str, optional (default="ch")
        Name of the channel column in cluster_info_df
        
    event_times_col : str, optional (default="Event Times")
        Name of the event times column in matched_swr_df
        
    event_cluster_col : str, optional (default="Event Cluster IDs")
        Name of the event cluster IDs column in matched_swr_df
        
    Returns:
    --------
    filtered_df : pd.DataFrame
        DataFrame with same structure as input but with only spikes
        from clusters on the specified channels. Contains:
            - All original columns from matched_swr_df
            - Filtered 'Event Times'
            - Filtered 'Event Cluster IDs'
            - Updated 'Unique Event Cluster IDs' (if present)
            - Updated 'Sorted Event Cluster IDs' (if present)
            
    Examples:
    ---------
    >>> # Filter for clusters on channels 35 and 29
    >>> filtered = filter_swr_by_channels(
    ...     matched_swr_df=swr_data,
    ...     cluster_info_df=cluster_info,
    ...     channels=[35, 29]
    ... )
    
    >>> # Using different column names
    >>> filtered = filter_swr_by_channels(
    ...     matched_swr_df=swr_data,
    ...     cluster_info_df=cluster_info,
    ...     channels=[35, 29],
    ...     cluster_id_col="Cluster ID",
    ...     channel_col="Channel"
    ... )
    """
    
    # Validate inputs
    if not isinstance(channels, (list, np.ndarray)):
        raise TypeError("channels must be a list or numpy array")
    
    if cluster_id_col not in cluster_info_df.columns:
        raise ValueError(
            f"Column '{cluster_id_col}' not found in cluster_info_df. "
            f"Available columns: {list(cluster_info_df.columns)}"
        )
    
    if channel_col not in cluster_info_df.columns:
        raise ValueError(
            f"Column '{channel_col}' not found in cluster_info_df. "
            f"Available columns: {list(cluster_info_df.columns)}"
        )
    
    if event_times_col not in matched_swr_df.columns:
        raise ValueError(
            f"Column '{event_times_col}' not found in matched_swr_df"
        )
    
    if event_cluster_col not in matched_swr_df.columns:
        raise ValueError(
            f"Column '{event_cluster_col}' not found in matched_swr_df"
        )
    
    # Get clusters on the specified channels
    clusters_on_channels = cluster_info_df[
        cluster_info_df[channel_col].isin(channels)
    ][cluster_id_col].values
    
    if len(clusters_on_channels) == 0:
        print(f"Warning: No clusters found on channels {channels}")
        # Return empty dataframe with same structure
        filtered_df = matched_swr_df.copy()
        filtered_df[event_times_col] = filtered_df[event_times_col].apply(lambda x: [])
        filtered_df[event_cluster_col] = filtered_df[event_cluster_col].apply(lambda x: [])
        return filtered_df
    
    print(f"Found {len(clusters_on_channels)} clusters on channels {channels}")
    print(f"Cluster IDs: {sorted(clusters_on_channels)}")
    
    # Create a copy to avoid modifying the original
    filtered_df = matched_swr_df.copy()
    
    # Helper function to parse string lists if needed
    def parse_list(val):
        if isinstance(val, str):
            return literal_eval(val)
        return val
    
    # Filter function for each row
    def filter_row(row):
        times = parse_list(row[event_times_col])
        cluster_ids = parse_list(row[event_cluster_col])
        
        # Create mask for clusters on specified channels
        mask = [cid in clusters_on_channels for cid in cluster_ids]
        
        # Apply mask
        filtered_times = [t for t, m in zip(times, mask) if m]
        filtered_clusters = [c for c, m in zip(cluster_ids, mask) if m]
        
        return filtered_times, filtered_clusters
    
    # Apply filtering
    filtered_data = filtered_df.apply(filter_row, axis=1, result_type='expand')
    filtered_df[event_times_col] = filtered_data[0]
    filtered_df[event_cluster_col] = filtered_data[1]
    
    # Update unique and sorted columns if they exist
    def get_unique_preserving_order(seq):
        """Get unique values preserving order"""
        if not seq or seq is None:
            return []
        seen = set()
        return [x for x in seq if not (x in seen or seen.add(x))]
    
    if "Unique Event Cluster IDs" in filtered_df.columns:
        filtered_df["Unique Event Cluster IDs"] = filtered_df[event_cluster_col].apply(
            get_unique_preserving_order
        )
    
    if "Sorted Event Cluster IDs" in filtered_df.columns:
        filtered_df["Sorted Event Cluster IDs"] = filtered_df[event_cluster_col].apply(
            lambda x: sorted(get_unique_preserving_order(x)) if x else []
        )
    
    # Print summary statistics
    total_spikes_before = sum(len(parse_list(x)) for x in matched_swr_df[event_cluster_col])
    total_spikes_after = sum(len(x) for x in filtered_df[event_cluster_col])
    
    print(f"\nFiltering summary:")
    print(f"  Total spikes before: {total_spikes_before}")
    print(f"  Total spikes after: {total_spikes_after}")
    print(f"  Spikes removed: {total_spikes_before - total_spikes_after}")
    print(f"  Percentage retained: {100 * total_spikes_after / total_spikes_before:.1f}%")
    
    return filtered_df