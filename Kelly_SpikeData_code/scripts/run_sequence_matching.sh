#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_sequence_matching.sh              # defaults to Pyramidal
#   ./run_sequence_matching.sh Pyramidal
#   ./run_sequence_matching.sh FSI
#   ./run_sequence_matching.sh Other
#
# Any arguments after the neuron type are passed through to the Python script.

NEURON_TYPE="${1:-Pyramidal Neuron}"

case "$NEURON_TYPE" in
    Pyramidal|FSI|Other)
        ;;
    *)
        echo "Error: invalid neuron type '$NEURON_TYPE'." >&2
        echo "Allowed values: Pyramidal Neuron, FSI, Other" >&2
        exit 2
        ;;
esac

# Remove the neuron-type positional argument if the user supplied one.
if [[ $# -gt 0 ]]; then
    shift
fi

echo "Running sequence matching for neuron type: $NEURON_TYPE"

python example_sequence_matching.py \
    --neuron-type "$NEURON_TYPE" \
    "$@"
