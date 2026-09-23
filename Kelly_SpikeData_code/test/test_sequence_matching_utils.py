import sys
import unittest
import numpy as np
import os
import tempfile
import shutil
import pandas as pd

sys.path.append("src/")  # noqa

import sequence_matching_utils as utils  # noqa
import example_sequence_matching as matrix  # noqa

# Default parameters used in 'lccs_rabin_karp'
BASE = 257
MOD = (1 << 61) - 1


class TestSequence(unittest.TestCase):
    def test_preprocess_lists(self):

        for _ in range(20):

            # Two lists with random lengths & random int values
            original_l1 = np.random.uniform(
                low=0,
                high=10,
                size=np.random.randint(low=1, high=5)
            ).astype(int)
            original_l2 = np.random.uniform(
                low=0,
                high=10,
                size=np.random.randint(low=1, high=5)
            ).astype(int)

            l1, l2 = utils._preprocess_lists(
                original_l1,
                original_l2,
            )

            # Checks that _preprocess_lists() reorders lists correctly
            self.assertTrue(
                len(l1) <= len(l2)
            )

    def test_compute_pow_base(self):

        # Sanity check for a small number powers
        pow_base = utils._compute_pow_base(
            n=3,
            base=BASE,
            mod=MOD,
        )
        self.assertEqual(
            pow_base, [1, 257, 66049, 16974593],
        )

        # Ensures that all n < 0 return an empty list
        self.assertEqual(
            utils._compute_pow_base(-1, BASE, MOD),
            utils._compute_pow_base(-2, BASE, MOD),
        )

        # Ensures that all n = 0 returns [1]
        self.assertEqual(
            utils._compute_pow_base(0, BASE, MOD),
            [1],
        )

    def test_prefix_hash(self):

        valid_array = [0, 4, 20, 200, 3]

        # Ensures that '_prefix_hash' works with lists and np.arrays
        self.assertTrue(
            utils._prefix_hash(valid_array, BASE, MOD),
            utils._prefix_hash(np.array(valid_array), BASE, MOD),
        )

        # Ensures that an empty list does return a hash (0)
        self.assertEqual(utils._prefix_hash([], BASE, MOD), [0])

    def test_lccs_rabin_karp(self):

        # Overlap of 4 entries [1, 2, 3, 4]
        l1 = [0, 0, 1, 2, 3, 4, 0, 0]
        l2 = [1, 2, 3, 4, 5]

        # Sanity check for a valid call
        overlap, sequence = utils.lccs_rabin_karp(
            l1=l1, l2=l2, normalize=True, return_sequence=True,
        )
        self.assertEqual(overlap, 0.8)
        self.assertEqual(sequence, [1, 2, 3, 4])

        # Ensures that 'return_sequence' behaves as expected
        overlap = utils.lccs_rabin_karp(
            l1=l1, l2=l2, normalize=True, return_sequence=False,
        )
        self.assertEqual(overlap, 0.8)

        # Checks that 'normalize=False' works as intended
        overlap = utils.lccs_rabin_karp(
            l1=l1, l2=l2, normalize=False, return_sequence=False,
        )
        self.assertEqual(overlap, 4)


# testing example_make_raster.py functions
def mock_load_spike_data(time_dir, cluster_dir, label_dir):
    """Return simple DataFrame for testing."""
    return pd.DataFrame({
        "Time": [0.1, 0.2, 0.3],
        "Cluster ID": [1, 2, 3],
        "KSLabel": ["good", "good", "mua"],
    })


def mock_filter_dataframe(df, dictionary):
    """Behavior of filter_dataframe from your utils."""
    key, values = next(iter(dictionary.items()))
    return df[df[key].isin(values)].copy()


def mock_group_dataframes_by_time(window_df, event_df, event_time_column,
                                  keep_event_columns, time_interval_columns,
                                  progress=False):
    """Simulate grouping by time for overlap matrix."""
    return pd.DataFrame({
        "Event Cluster IDs": [event_df["Cluster ID"].tolist()]
    })


class TestPipelineFunctions(unittest.TestCase):
    def test_get_base_directory(self):
        self.assertEqual(matrix.get_base_directory("test"), "data/test_data")
        self.assertEqual(matrix.get_base_directory("7742"), "data/full_data")
        self.assertEqual(matrix.get_base_directory("7744"), "data/full_data")

    def test_build_file_paths_test(self):
        paths = matrix.build_file_paths("test", "PartnerIntro")
        self.assertIn("test_spike_times.npy", paths["spike_times"])
        self.assertIn("test_events_with_indices.csv", paths["behavior_csv"])
        self.assertIn("test_SWRs_ca2.csv", paths["swr_csv"])
        self.assertTrue(paths["spike_times"].startswith("data/test_data"))

    def test_build_file_paths_fulldata_7742(self):
        paths = matrix.build_file_paths("7742", "PartnerIntro")
        # Check directory structure
        for key in paths:
            self.assertIn("data/full_data/7742/PartnerIntro", paths[key])
        # Check special tag applied
        self.assertIn("7742_PartnerIntro_sleepyvole", paths["behavior_csv"])

    def test_build_file_paths_fulldata_7744(self):
        paths = matrix.build_file_paths("7744", "SSIntro")
        for key in paths:
            self.assertIn("data/full_data/7744/SSIntro", paths[key])
        self.assertIn("7744_SSIntro_events_with_indices.csv",
                      paths["behavior_csv"])

    def test_preprocess_behavior_df(self):
        df = pd.DataFrame({
            "EventType": ["social interaction", "ignore"],
            "indexStart": [2500, 5000],
            "indexEnd": [5000, 7500],
        })

        # Patch filter_dataframe
        original_filter = sys.modules["loading_utils"].filter_dataframe
        sys.modules["loading_utils"].filter_dataframe = mock_filter_dataframe

        out = matrix.preprocess_behavior_df(df)
        sys.modules["loading_utils"].filter_dataframe = original_filter

        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["Start"], 1.0)
        self.assertEqual(out.iloc[0]["Stop"], 2.0)

    def test_compute_overlap_matrix(self):
        behavior_lists = [[1, 2, 3], [4, 5], []]
        swr_lists = [[2, 3], [10], [1, 2, 3]]

        # Expected:
        # First SWR row overlaps with first behavior seq → lccs = 2/3
        # Third SWR row overlaps with first behavior seq → lccs = 3/3 = 1.0

        mat = matrix.compute_overlap_matrix(behavior_lists, swr_lists)

        self.assertEqual(mat.shape, (3, 3))
        self.assertGreater(mat[0, 0], 0)
        self.assertAlmostEqual(mat[2, 0], 1.0)


if __name__ == "__main__":
    unittest.main()
