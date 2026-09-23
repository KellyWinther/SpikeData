import unittest
import os
from pathlib import Path
import numpy as np
import pandas as pd
import sys

sys.path.append("src/")  # noqa

import analysis_utils as au
import loading_utils as lu

DATA_DIR = "data/test_data"

PATH_SPIKE_TS = os.path.join(DATA_DIR, "test_spike_times.npy")
PATH_SPIKE_CL = os.path.join(DATA_DIR, "test_spike_clusters.npy")
PATH_KSLABEL = os.path.join(DATA_DIR, "test_cluster_KSLabel.tsv")
PATH_SWR = os.path.join(DATA_DIR, "TEST_SWRs_ca2.csv")

BAD_PATH = os.path.join(DATA_DIR, "nonexistent_file.npy")


class SharedTestData:

    spike_df = None
    swr_df = None
    event_df_loading = None
    event_df_analysis = None
    ready = False

    @classmethod
    def build(cls):
        if cls.ready:
            return

        # Load Spike data (shared)
        cls.spike_df = lu.load_spike_data(
            time_dir=PATH_SPIKE_TS,
            cluster_dir=PATH_SPIKE_CL,
            label_dir=PATH_KSLABEL,
        )

        # Load SWR data (shared)
        cls.swr_df = pd.read_csv(PATH_SWR)

        # Build event_df for loading_utils tests
        # Event Times, Event Cluster IDs
        cls.event_df_loading = lu.match_times(
            dataframe=cls.spike_df,
            directory=PATH_SWR,
            filter_event_data={"KSLabel": ["good"]},
            keep_event_columns=["Time", "Cluster ID"],
            progress=False,
        )

        # Build event_df for analysis_utils tests
        # Spike Times (s), Cluster IDs
        cls.event_df_analysis = au.match_up(
            df=cls.spike_df,
            swr_df=cls.swr_df,
            only_keep_good=True,
            progress=False,
        )

        cls.ready = True


#  TEST CLASS 1: loading_utils
class TestLoadingUtils(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        SharedTestData.build()
        cls.spike_df = SharedTestData.spike_df
        cls.swr_df = SharedTestData.swr_df
        cls.event_df_loading = SharedTestData.event_df_loading

    # load_spike_data — real paths
    def test_load_spike_data_structure(self):
        df = self.spike_df
        self.assertIn("Time", df.columns)
        self.assertIn("Cluster ID", df.columns)
        self.assertIn("KSLabel", df.columns)
        self.assertEqual(len(df), 50)

    # load_spike_data — Bad paths
    def test_load_spike_data_bad_path(self):
        with self.assertRaises(SystemExit):
            lu.load_spike_data(
                time_dir=BAD_PATH,
                cluster_dir=PATH_SPIKE_CL,
                label_dir=PATH_KSLABEL,
            )

    # filter_dataframe with valid column
    def test_filter_dataframe_valid(self):
        df = lu.filter_dataframe(self.spike_df, {"KSLabel": ["good"]})
        self.assertTrue(all(df["KSLabel"] == "good"))

    # filter_dataframe with invalid column
    def test_filter_dataframe_missing_column(self):
        df = lu.filter_dataframe(self.spike_df, {"NOPE": ["x"]})
        self.assertEqual(len(df), len(self.spike_df))

    # match_times and ensure output columns exist
    def test_match_times_event_columns(self):
        df = self.event_df_loading
        self.assertIn("Event Times", df.columns)
        self.assertIn("Event Cluster IDs", df.columns)

    # match_times — test no matches (empty SWR window)
    def test_match_times_empty_window(self):
        # Make SWR file with bad times
        fake_swr = pd.DataFrame({"Start": [99999], "Stop": [100000]})
        out = lu.group_dataframes_by_time(
            window_df=fake_swr,
            event_df=self.spike_df,
            event_time_column="Time",
            keep_event_columns=["Time"],
            time_interval_columns=["Start", "Stop"],
            progress=False
        )
        self.assertEqual(out.iloc[0]["Event Times"], [])


#  TEST CLASS 2: analysis_utils
class TestAnalysisUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        SharedTestData.build()
        cls.spike_df = SharedTestData.spike_df
        cls.swr_df = SharedTestData.swr_df
        cls.event_df_loading = SharedTestData.event_df_loading
        cls.event_df_analysis = SharedTestData.event_df_analysis

    # make_correlation_dictionary valid path
    def test_make_correlation_dictionary(self):
        df = self.event_df_loading.rename(
            columns={"Event Cluster IDs": "Event Cluster IDs"}
        )

        corr = au.make_correlation_dictionary(
            df,
            id_column_name="Event Cluster IDs",
            normalize=False,
        )

        self.assertIsInstance(corr, dict)
        keys = list(corr.keys())
        self.assertGreater(len(keys), 0)

        for k in keys:
            self.assertEqual(len(corr[k]), len(keys))

    # make_correlation_dictionary with empty event lists
    def test_make_correlation_empty_lists(self):
        df = pd.DataFrame({"Event Cluster IDs": [[]]})
        corr = au.make_correlation_dictionary(df, "Event Cluster IDs")
        # should return dict with one entry mapping to itself
        self.assertEqual(list(corr.keys()), [])

    # analysis_utils.match_times with correct structure
    def test_analysis_utils_match_times_structure(self):
        df = self.event_df_analysis
        self.assertIn("Spike Times (s)", df.columns)
        self.assertIn("Cluster IDs", df.columns)

    # count_spikes with failure on missing columns
    def test_count_spikes_missing_column(self):
        bad_df = pd.DataFrame({"Spike Times (s)": [[1, 2]]})
        with self.assertRaises(KeyError):
            au.count_spikes(bad_df)

    # circular_permutation_test_with_firing_rate
    def test_circular_permutation_pipeline(self):

        result_df, true_counts, perm_counts = (
            au.circular_permutation_test_with_firing_rate(
                swr_df=self.event_df_analysis,
                spike_df=self.spike_df,
                n_permutations=4,
                shift_range_seconds=0.5,
                progress=False,
                mode="all",
            )
        )

        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertGreater(len(result_df), 0)

        # perm_counts each has 4 entries
        for k, v in perm_counts.items():
            self.assertEqual(len(v), 4)

    # circular_permutation_test_with_firing_rate thats very short
    def test_circular_permutation_short_total_duration(self):
        # Create very short fake spike_df
        small = self.spike_df.copy()
        small["Time"] = small["Time"] * 0

        # avoids division by 0
        _, _, perm = au.circular_permutation_test_with_firing_rate(
            swr_df=self.event_df_analysis,
            spike_df=small,
            n_permutations=3,
            shift_range_seconds=0.2,
            progress=False,
            mode="all",
        )

        for k, v in perm.items():
            self.assertEqual(len(v), 3)


if __name__ == "__main__":
    unittest.main()
