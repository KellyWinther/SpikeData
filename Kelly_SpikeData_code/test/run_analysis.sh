#!/bin/bash
test -e ssshtest || wget -q https://raw.githubusercontent.com/ryanlayer/ssshtest/master/ssshtest
. ssshtest
set -euo pipefail

run valid_call python src/example_call.py \
    --spike_time_filename "data/full_data/7742/PartnerIntro/spike_times.npy" \
    --cluster_filename "data/full_data/7742/PartnerIntro/spike_clusters.npy" \
    --KSlabel_filename "data/full_data/7742/PartnerIntro/cluster_KSLabel.tsv" \
    --swr_filename "data/full_data/7742/PartnerIntro/7742_Partnerintro_sleepyvole_SWRs_ca2.csv" \
    --output_csv True

echo "Comparing output file to expected output file..."
diff test/correlation_matrix.csv data/test_data/expected_correlation_matrix.csv
assert_exit_code 0
