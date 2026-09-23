import unittest
from pathlib import Path
import os
import numpy as np
import pandas as pd
import sys

sys.path.append("src/")  # noqa

import loading_utils as lu
import raster_plot_utils as rpu


DATA_DIR = "data/test_data"

PATH_SPIKE_TS = os.path.join(DATA_DIR, "test_spike_times.npy")
PATH_SPIKE_CL = os.path.join(DATA_DIR, "test_spike_clusters.npy")
PATH_KSLABEL = os.path.join(DATA_DIR, "test_cluster_KSLabel.tsv")
PATH_SWR = os.path.join(DATA_DIR, "TEST_SWRs_ca2.csv")

BAD_PATH = os.path.join(DATA_DIR, "nonexistent_file.npy")


class SharedRasterTestData:

    ready = False
    spike_df = None
    swr_df = None
    exp_df = None

    @classmethod
    def build(cls):
        if cls.ready:
            return

        # Load spike data (times, clusters, KSLabel)
        cls.spike_df = lu.load_spike_data(
            time_dir=str(PATH_SPIKE_TS),
            cluster_dir=str(PATH_SPIKE_CL),
            label_dir=str(PATH_KSLABEL),
        )

        # Match spikes to SWRs (loading_utils.match_times)
        cls.swr_df = lu.match_times(
            dataframe=cls.spike_df,
            directory=str(PATH_SWR),
            filter_event_data={"KSLabel": ["good"]},
            keep_event_columns=["Time", "Cluster ID"],
            progress=False,
        )

        # Prepare exploded raster dataframe
        cls.exp_df = rpu.prep_raster(cls.swr_df)

        cls.ready = True


# Tests using pipeline data
class TestPrepRaster(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        SharedRasterTestData.build()
        cls.spike_df = SharedRasterTestData.spike_df
        cls.swr_df = SharedRasterTestData.swr_df
        cls.exp_df = SharedRasterTestData.exp_df

    def test_prep_raster_structure(self):
        # prep_raster should return an exploded long-form dataframe
        exp_df = self.exp_df

        # Basic columns
        for col in ["Spike Times (s)", "Cluster IDs", "ripple_idx", "t_rel"]:
            self.assertIn(col, exp_df.columns)

        # Comes from original SWR df
        self.assertIn("Peak", exp_df.columns)

        self.assertEqual(len(exp_df), 7)
        self.assertSetEqual(set(exp_df["ripple_idx"]), {0, 1})

    def test_prep_raster_t_rel_values(self):
        exp_df = self.exp_df

        ripple0 = exp_df[exp_df["ripple_idx"] == 0]
        ripple1 = exp_df[exp_df["ripple_idx"] == 1]

        # Force numeric dtype for github action
        t0 = ripple0["t_rel"].astype(float)
        t1 = ripple1["t_rel"].astype(float)

        # Ripple 0, Peak = 2, SpikeTimes = 1,2,3, t_rel = -1,0,1
        self.assertSetEqual(set(np.round(t0, 6)), {-1.0, 0.0, 1.0})

        # Ripple 1, Peak = 43, SpikeTimes = 41,42,43,44, t_rel = -2,-1,0,1
        self.assertSetEqual(set(np.round(t1, 6)), {-2.0, -1.0, 0.0, 1.0})


class TestSelectRipples(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        SharedRasterTestData.build()
        cls.exp_df = SharedRasterTestData.exp_df

    def test_select_all_ripples(self):
        # ripple_index=None should return the full exploded df
        out = rpu.select_ripples_to_plot(self.exp_df, ripple_index=None)
        self.assertEqual(len(out), len(self.exp_df))

    def test_select_single_ripple(self):
        # Selecting a single ripple index returns only those rows
        out = rpu.select_ripples_to_plot(self.exp_df, ripple_index=0)
        self.assertEqual(set(out["ripple_idx"]), {0})
        # In test data ripple 0 has 3 spikes
        self.assertEqual(len(out), 3)

    def test_select_multiple_ripples(self):
        # Multiple ripple indices returns a union of their rows
        out = rpu.select_ripples_to_plot(self.exp_df, ripple_index=[0, 1])
        self.assertEqual(set(out["ripple_idx"]), {0, 1})
        self.assertEqual(len(out), len(self.exp_df))


# Edge-case tests (fake Dfs)
class TestPrepRasterEdgeCases(unittest.TestCase):
    # Edge cases and error handling for prep_raster

    def test_prep_raster_none(self):
        with self.assertRaises(ValueError):
            rpu.prep_raster(None)

    def test_prep_raster_wrong_type(self):
        with self.assertRaises(TypeError):
            rpu.prep_raster("not a dataframe")

    def test_prep_raster_missing_columns(self):
        bad_df = pd.DataFrame({"Peak": [1.0], "Event Times": [[0.9, 1.1]]})
        with self.assertRaises(ValueError):
            rpu.prep_raster(bad_df)

    def test_prep_raster_string_lists(self):
        df = pd.DataFrame({
            "Peak": [1.0],
            "Event Times": ["[0.9, 1.1]"],
            "Event Cluster IDs": ["[10, 10]"],
        })
        exp_df = rpu.prep_raster(df)

        self.assertEqual(len(exp_df), 2)
        self.assertIsInstance(exp_df["Spike Times (s)"].iloc[0], float)
        self.assertIsInstance(exp_df["Cluster IDs"].iloc[0], int)


class TestSelectRipplesEdgeCases(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Small fake dfs for testing
        cls.exp_df = pd.DataFrame({
            "Spike Times (s)": [0.9, 1.1, 1.9, 2.2],
            "Cluster IDs": [10, 10, 11, 12],
            "Peak": [1.0, 1.0, 2.0, 2.0],
            "ripple_idx": [0, 0, 1, 1],
            "t_rel": [-0.1, 0.1, -0.1, 0.2],
        })

    def test_ripple_index_wrong_type(self):
        with self.assertRaises(TypeError):
            rpu.select_ripples_to_plot(self.exp_df, ripple_index=3.14)

    def test_ripple_index_wrong_element_type(self):
        with self.assertRaises(ValueError):
            rpu.select_ripples_to_plot(self.exp_df, ripple_index=["0"])

    def test_ripple_index_invalid_value(self):
        with self.assertRaises(ValueError):
            rpu.select_ripples_to_plot(self.exp_df, ripple_index=[99])

    def test_ripple_index_mixed_invalid(self):
        with self.assertRaises(ValueError):
            rpu.select_ripples_to_plot(self.exp_df, ripple_index=[0, 5])


if __name__ == "__main__":
    unittest.main()
