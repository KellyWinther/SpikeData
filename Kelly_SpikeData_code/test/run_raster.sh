#!/bin/bash
test -e ssshtest || wget -q https://raw.githubusercontent.com/ryanlayer/ssshtest/master/ssshtest
. ssshtest
set -euo pipefail

run valid_call python src/example_make_raster.py \
    --spike_time data/test_data/TEST_spike_times.npy \
    --clusters data/test_data/TEST_spike_clusters.npy \
    --kslabels data/test_data/TEST_cluster_KSLabel.tsv \
    --swr_csv data/test_data/TEST_SWRs_ca2.csv \
    --window 3 \
    --color black \
    --tick_width 100 \
    --height 5 \
    --width 7 \
    --ripple_index 0 1 \
    --output_csv True

echo "Comparing output file to expected output file..."
diff test/exploded_raster_data.csv data/test_data/expected_raster_data.csv
assert_exit_code 0