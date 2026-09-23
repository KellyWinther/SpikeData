"""
make_raster.py

Generate ripple-aligned raster plots from spike and SWR data.

Example usage: Test data
---------------
python src/example_make_raster.py \
    --spike_time "data/test_data/test_spike_times.npy" \
    --clusters "data/test_data/test_spike_clusters.npy" \
    --kslabels "data/test_data/test_cluster_KSLabel.tsv" \
    --swr_csv "data/test_data/test_SWRs_ca2.csv" \
    --window 3 \
    --color black \
    --tick_width 100 \
    --height 5 \
    --width 7 \
    --ripple_index 0 1

Example usage: for all ripples
---------------
python src/example_make_raster.py \
    --spike_time "data/full_data/7744/PartnerIntro/spike_times.npy" \
    --clusters "data/full_data/7744/PartnerIntro/spike_clusters.npy" \
    --kslabels "data/full_data/7744/PartnerIntro/cluster_KSLabel.tsv" \
    --swr_csv \
        "data/full_data/7744/PartnerIntro/7744_Partnerintro_SWRs_ca2.csv" \
    --window 0.1 \
    --color blue \
    --tick_width 20 \
    --height 7 \
    --width 9

Example usage: for specific ripple
---------------
python src/example_make_raster.py \
    --spike_time "data/full_data/7744/PartnerIntro/spike_times.npy" \
    --clusters "data/full_data/7744/PartnerIntro/spike_clusters.npy" \
    --kslabels "data/full_data/7744/PartnerIntro/cluster_KSLabel.tsv" \
    --swr_csv \
        "data/full_data/7744/PartnerIntro/7744_Partnerintro_SWRs_ca2.csv" \
    --window 0.1 \
    --color blue \
    --tick_width 20 \
    --height 7 \
    --width 9 \
    --ripple_index 110 \
    --save_path "data/full_data/7744/PartnerIntro/raster_ripple110.png"
"""

import argparse

from loading_utils import load_spike_data, match_times
from raster_plot_utils import (
    prep_raster,
    select_ripples_to_plot,
    plot_raster,
)


def build_parser():
    """
    Construct and return the command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser supporting file paths and plotting options for the raster.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate ripple-aligned raster plots from spike time and "
            "SWR interval data."
        )
    )

    parser.add_argument(
        "--spike_time",
        type=str,
        required=True,
        help="Path to spike_times.npy.",
    )

    parser.add_argument(
        "--clusters",
        type=str,
        required=True,
        help="Path to spike_clusters.npy.",
    )

    parser.add_argument(
        "--kslabels",
        type=str,
        required=True,
        help="Path to cluster_KSLabel.tsv.",
    )

    parser.add_argument(
        "--swr_csv",
        type=str,
        required=True,
        help="Path to SWR CSV with Peak, Start, and Stop columns.",
    )

    parser.add_argument(
        "--ripple_index",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Optional ripple index or multiple indices (space-separated). "
            "If omitted, all ripples in the dataframe are plotted."
        ),
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help=(
            "Optional path to save the raster plot image. "
            "If omitted, the plot is displayed instead."
        ),
    )

    parser.add_argument(
        "--window",
        type=float,
        default=0.1,
        help=(
            "Window size in seconds (±window around ripple peak). "
            "Default = 0.1 (100 ms)."
        ),
    )

    parser.add_argument(
        "--height",
        type=float,
        default=9,
        help="Height of raster plot (in inches).",
    )

    parser.add_argument(
        "--width",
        type=float,
        default=6,
        help="Width of raster plot (in inches).",
    )

    parser.add_argument(
        "--tick_width",
        type=float,
        default=3,
        help="Marker size for raster spike dots.",
    )

    parser.add_argument(
        "--color",
        type=str,
        default="black",
        help="Color for spike markers.",
    )

    return parser


def main():
    """
    Main workflow to load spike data, align spikes with SWRs,
    prepare the raster DataFrame, filter by ripple index,
    and generate a raster plot.
    """
    parser = build_parser()
    args = parser.parse_args()

    # 1. Load spike data
    print("\n 1. Loading spike data…")
    spike_df = load_spike_data(
        time_dir=args.spike_time,
        cluster_dir=args.clusters,
        label_dir=args.kslabels,
    )

    # 2. Match spikes to SWRs
    print(" 2. Matching spikes to SWRs…")
    swr_df = match_times(
        spike_df,
        args.swr_csv,
        filter_event_data={"KSLabel": ["good"]},
        keep_event_columns=["Time", "Cluster ID"],
        progress=True,
    )
    head = swr_df.head(10)
    print(" First ten rows of matched spike-SWR data:")
    print(head)

    # 3. Construct exploded raster dataframe
    print("\n 3. Preparing raster data (exploding spikes)…")
    exp_df = prep_raster(swr_df)

    # 4. Apply ripple selection (if provided)
    print(" 4. Selecting ripples to plot…")
    exp_df_sel = select_ripples_to_plot(
        exp_df,
        ripple_index=args.ripple_index,
    )

    # 5. Plot raster
    print(" 5. Creating raster… ")
    plot_raster(
        exp_df_sel,
        height=args.height,
        width=args.width,
        color=args.color,
        tick_width=args.tick_width,
        window=args.window,
        ripple_index=args.ripple_index,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()
