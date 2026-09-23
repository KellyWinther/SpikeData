import os
import ast
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")

# import functions from src/
sys.path.append("src/")  # noqa

from loading_utils import load_spike_data, match_times # noqa
from raster_plot_utils import (
    prep_raster, 
    select_ripples_to_plot, 
    plot_raster
)
from analysis_utils import make_correlation_dictionary, visualize_correlation_dictionary  # noqa

# define name for datasets based on filepath
DATASETS = [
    "7742/PartnerIntro",
    "7742/SSIntro",
    "7744/PartnerIntro",
    "7744/SSIntro",
]

# quick helper function to get all datasets filenames
def get_swr_csv_name(dataset):
    """ uses dataset string to get swr csv filename """
    parts = dataset.split('/')
    animal_id = parts[0]
    session = parts[1]
    
    # handle special case for sleepyvole
    if dataset == "7742/PartnerIntro":
        return f"{animal_id}_{session}_sleepyvole_SWRs_ca2.csv"
    
    # handle special case for sleepyvole
    if dataset == "7742/SSIntro":
        return f"{animal_id}_{session}_sleepyvole_SWRs_ca2.csv"
    
    # all other cases
    return f"{animal_id}_{session}_SWRs_ca2.csv"

# create outputs directory
try:
    os.mkdir("outputs")
except FileExistsError:
    pass

rule all:
    input:
        # 1. All Raster Plots
        expand("outputs/{dataset}/ripple_raster.png", dataset=DATASETS),
        # 2. All Correlation Matrices
        expand("outputs/{dataset}/correlation_matrix.png", dataset=DATASETS)

rule load_spike_data:
    input:
        time_dir = "data/full_data/{dataset}/spike_times.npy",
        cluster_dir  = "data/full_data/{dataset}/spike_clusters.npy",
        label_dir = "data/full_data/{dataset}/cluster_KSLabel.tsv",
    output:
        csv="outputs/{dataset}/spike_data.csv"
    run:
        os.makedirs(os.path.dirname(output.csv), exist_ok=True)
        
        df = load_spike_data(
                time_dir = input.time_dir,
                cluster_dir  = input.cluster_dir,
                label_dir = input.label_dir,
        )
        df.to_csv(output.csv, index=False)

rule match_spikes_with_SWRs:
    input:
        spike_data="outputs/{dataset}/spike_data.csv"
    output:
        matched_data="outputs/{dataset}/matched_SWR_data.csv"
    params:
        swr_csv_name=lambda wildcards: get_swr_csv_name(wildcards.dataset)
    run:
        spike_df = pd.read_csv(input.spike_data)

        # Build path to SWR CSV file
        swr_path = f"data/full_data/{wildcards.dataset}/{params.swr_csv_name}"

        # Use match_times from loading_utils
        df = match_times(
            dataframe=spike_df,
            directory=swr_path,
            filter_event_data={"KSLabel": ["good"]},
            keep_event_columns=["Time", "Cluster ID"],
        )

        df.to_csv(output.matched_data, index=False)

rule generate_correlation_matrix:
    input:
        matched_data="outputs/{dataset}/matched_SWR_data.csv"
    output:
        matrix="outputs/{dataset}/correlation_matrix.png"
    run:
        # Load the matched SWR data
        df = pd.read_csv(input.matched_data)

        # Converts 'Event Cluster IDs' strings to Python lists
        df["Event Cluster IDs"] = df["Event Cluster IDs"].apply(
            lambda s: [int(x) for x in ast.literal_eval(s)]
        )

        # Generate the correlation dictionary
        corr_dictionary = make_correlation_dictionary(
            df,
            normalize=False,
        )        

        # Visualize and save the correlation matrix
        visualize_correlation_dictionary(
            corr_dictionary,
            save_directory=output.matrix,
        )

rule generate_raster_plot:
    """Generates a ripple-aligned raster plot for a single dataset."""
    input:
        matched_data="outputs/{dataset}/matched_SWR_data.csv"
    output:
        raster="outputs/{dataset}/ripple_raster.png"
    params:
        window=0.1,  
        color="black",
        tick_width=20,
        height=7,
        width=9,
        ripple_index=None
    run:
        swr_df = pd.read_csv(input.matched_data)
        
        print("Preparing raster data (exploding spikes)...")
        exp_df = prep_raster(swr_df)

        exp_df_sel = select_ripples_to_plot(exp_df, ripple_index=None)
        
        print(f"Plotting raster to {output.raster}")
        # NOTE: This assumes 'plot_raster' in your utility script 
        # has been updated to save the figure using the 'save_path' argument.
        plot_raster(
            exp_df_sel,
            height=params.height,
            width=params.width,
            color=params.color,
            tick_width=params.tick_width,
            window=params.window,
            ripple_index=params.ripple_index,
            save_path=output.raster
        )
