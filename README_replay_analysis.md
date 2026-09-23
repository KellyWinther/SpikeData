# Ordered behavior → SWR replay analysis

## Definition used
The analysis filters spikes first, sorts them by spike time, and then reduces each behavior bout and SWR to **unique cluster IDs in order of first appearance**. Replay is the longest ordered sequence shared by a behavior bout and an SWR; extra/interspersed clusters are allowed.

Example: behavior `[1,2,3,4,5]`, raw SWR `[1,1,2,3,4]` → unique SWR `[1,2,3,4]` → matched sequence `[1,2,3,4]` → **80% of the behavior sequence replayed**.

## Flexible filtering
In `run_replay_analysis.py`:

- Disable cell-type filtering: `"filter_neuron_type": False`
- Pyramidal only: `"neuron_types_to_keep": ("Pyramidal Neuron",)`
- FSI only: `("FSI",)`
- Pyramidal + FSI: `("Pyramidal Neuron", "FSI")`
- Other: `("Other",)`
- Non-somatic: `("Non-somatic",)`
- Disable Kilosort-good filtering: `"filter_kslabel": False`

Filtering happens before grouping spikes into behavior/SWR windows to reduce memory and computation.

## Inputs
- `spike_times.npy`
- `spike_clusters.npy`
- `cluster_KSLabel.tsv`
- `cluster_type.csv`
- BORIS behavior CSV with `EventType`, `indexStart`, `indexEnd`
- SWR CSV with `Start`, `Peak`, `Stop`

## Outputs
- `behavior_unique_clusters.csv`
- `swr_unique_clusters.csv`
- `behavior_swr_pairwise_replay.csv`
- `best_replay_per_swr.csv`
- `replay_percent_histogram.png`

`best_replay_per_swr.csv` gives one histogram datapoint per SWR. The full pairwise CSV preserves the exact matched ordered cluster IDs for every behavior-bout × SWR comparison.
