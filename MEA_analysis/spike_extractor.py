# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 12:25:57 2021

@author: Marvin
"""
import numpy as np
import h5py
import pandas as pd
from ipywidgets import widgets
import plotly
from plotly import graph_objs as go


class Recording_spikes:
    """
    Spike class

    This class stores methods to read the .hdf5 file which is the output of the spikesorting algorithm Herdingspikes2
    It allows for loading either all the spikes in the file or just a subset either based on number of cells or with a
    maximal nr of spikes threshold.
    """

    spikes = {}

    def __init__(self, file):

        """
        Initialize function. Opens the .hdf5 file that contains results of the
        Herdingspikes2 spikesorting algorithm and assignes them to the object.

        Parameters
        ----------
        file: str: The location of the hdf5 results file

        Returns
        -------

        """
        self.file = file
        with h5py.File(file, "r") as f:
            self.spikes["centres"] = np.array(f["/centres"], dtype=float)
            self.spikes["cluster_id"] = np.array(f["/cluster_id"], dtype=int)
            self.spikes["times"] = np.array(f["/times"], dtype=int)
            self.spikes["sampling"] = np.array(f["/Sampling"], dtype=float)
            self.spikes["channels"] = np.array(f["/ch"], dtype=float)
            self.spikes["spike_freq"] = np.array(
                np.unique(self.spikes["cluster_id"], return_counts=True)
            )
            self.spikes["nr_cells"] = np.max(self.spikes["spike_freq"][0, :]) + 1
            self.spikes["cell_indices"] = np.linspace(
                1, self.spikes["nr_cells"], self.spikes["nr_cells"], dtype=int
            )
            self.spikes["max_spikes"] = np.max(self.spikes["spike_freq"][1, :])
            self.define_subset()
            self.spikes["shapes"] = np.array(f["/shapes"], dtype=float)

    def define_subset(self, subset_left=None, subset_right=None):
        """
        Defines subset of cells in the recording from which all spikes data will
        be picked

        Parameters
        ----------
        subset_left: int: First cluster in the results hdf5 file to be considered
        subset_right: int: Last cluster in the results hdf5 file to be considered

        Returns
        -------
        self.spikes_subset: np.array(dtype=bool) Index of cells in the subset == True
        cells not in the subset == False
        """

        # Check which cells should be loaded depending on subset nr
        spikes_subset = np.ones(self.spikes["nr_cells"], dtype=bool)
        if not subset_left and not subset_right:
            self.spikes_subset = spikes_subset
        else:
            spikes_subset[0:subset_left] = False
            spikes_subset[subset_right:] = False
            self.spikes_subset = spikes_subset

    def define_thr(self, max_spikes=None, min_spikes=None):
        """
        Defines a subset of cells in the hdf5 results file which contain at least
        x number of spikes and maximal y number of spikes.

        Parameters
        ----------
        max_spikes: int: = maximal number of spikes in a cluster
        min_spikes: int = minimal number of spikes in a cluster

        Returns
        -------
        self.spikes_threshold: Bool array of cells that fall into the subset and
        those that do not, similar as in "define_subset" function.
        """

        over_threshold = np.array(
            self.spikes["spike_freq"][1, :] < max_spikes, dtype=bool
        )
        under_threshold = np.array(
            self.spikes["spike_freq"][1, :] > min_spikes, dtype=bool
        )
        threshold_pass = np.multiply(over_threshold, under_threshold)
        self.spikes_threshold = threshold_pass

        # if any(max_spikes)

    def get_spikes(self, subset=False, thr=False):

        """
        Collects the spikes from the hdf5 file. Optional only from a subset of cells.

        Parameters
        ----------
        subset: bool: If true, function will use self.spikes_subset to get spikes
        from a subset of clusters
        thr: bool: If true, function will use self.spikes_threshold to get spikes
        from a subset of clusters

        Returns
        -------
        spikes: dictonary: containing all information for the clusters in the
        results .hdf5 file.

        spiketimestamps: np.array: Only the spiketimestamps for each cell.

        spikes_df: pandas.DataFrame containing the information about each loaded
        cell (including the spiketimes as arrays in pandas field)

        """

        if subset and thr:
            subset_subset = np.multiply(self.spikes_subset, self.spikes_threshold)
            # print(subset_subset)
            spikes = {}
            spikes["centres"] = self.spikes["centres"][subset_subset, :]
            spikes["spike_freq"] = self.spikes["spike_freq"][:, subset_subset]
            spikes["max_spikes"] = np.max(spikes["spike_freq"][1, :])
            spikes["nr_cells"] = np.count_nonzero(subset_subset)
            spikes["sampling"] = self.spikes["sampling"]
            spikes["cell_indices"] = self.spikes["cell_indices"][subset_subset]
            spikes["spiketimestamps"] = np.zeros(
                (spikes["max_spikes"], spikes["nr_cells"])
            )
            for i in range(spikes["nr_cells"]):
                spikes["spiketimestamps"][
                    0 : spikes["spike_freq"][1, i], i
                ] = self.spikes["times"][
                    self.spikes["cluster_id"] == -1 + spikes["cell_indices"][i]
                ]

        elif subset:
            spikes = {}
            spikes["centres"] = self.spikes["centres"][0, self.spikes_subset]
            spikes["spike_freq"] = self.spikes["spike_freq"][:, self.spikes_subset]
            spikes["max_spikes"] = np.max(spikes["spike_freq"][1, :])
            spikes["nr_cells"] = np.count_nonzero(self.spikes_subset)
            spikes["sampling"] = self.spikes["sampling"]
            spikes["cell_indices"] = self.spikes["cell_indices"][self.spikes_subset]
            spikes["spiketimestamps"] = np.zeros(
                (spikes["max_spikes"], spikes["nr_cells"])
            )
            for i in range(spikes["nr_cells"]):
                spikes["spiketimestamps"][
                    0 : spikes["spike_freq"][1, i], i
                ] = self.spikes["times"][
                    self.spikes["cluster_id"] == -1 + spikes["cell_indices"][i]
                ]

        elif thr:
            spikes = {}
            spikes["centres"] = self.spikes["centres"][0, self.spikes_threshold]
            spikes["spike_freq"] = self.spikes["spike_freq"][:, self.spikes_threshold]
            spikes["max_spikes"] = np.max(spikes["spike_freq"][1, :])
            spikes["nr_cells"] = np.count_nonzero(self.spikes_threshold)
            spikes["sampling"] = self.spikes["sampling"]
            spikes["cell_indices"] = self.spikes["cell_indices"][self.spikes_threshold]
            spikes["spiketimestamps"] = np.zeros(
                (spikes["max_spikes"], spikes["nr_cells"])
            )
            for i in range(spikes["nr_cells"]):
                spikes["spiketimestamps"][
                    0 : spikes["spike_freq"][1, i], i
                ] = self.spikes["times"][
                    self.spikes["cluster_id"] == -1 + spikes["cell_indices"][i]
                ]

        else:
            spikes = {}
            spikes["centres"] = self.spikes["centres"]
            spikes["spike_freq"] = self.spikes["spike_freq"]
            spikes["max_spikes"] = np.max(spikes["spike_freq"][1, :])
            spikes["nr_cells"] = self.spikes["nr_cells"]
            spikes["sampling"] = self.spikes["sampling"]
            spikes["cell_indices"] = self.spikes["cell_indices"]
            spikes["spiketimestamps"] = np.zeros(
                (spikes["max_spikes"], spikes["nr_cells"])
            )
            for i in range(spikes["nr_cells"]):
                spikes["spiketimestamps"][
                    0 : spikes["spike_freq"][1, i], i
                ] = self.spikes["times"][
                    self.spikes["cluster_id"] == -1 + spikes["cell_indices"][i]
                ]

        spikes_df = pd.DataFrame(
            columns=(
                "Cell index",
                "Centres x",
                "Centres y",
                "Nr of spikes",
                "Area",
                "Spiketimestamps",
            )
        )
        for cell in range(spikes["nr_cells"]):
            # print(spikes['centres'][0])
            spikes_df.loc[cell] = [
                spikes["cell_indices"][cell],
                spikes["centres"][cell, 0],
                spikes["centres"][cell, 1],
                spikes["spike_freq"][1, cell],
                spikes["spike_freq"][1, cell]
                / np.max(spikes["spike_freq"][1, :])
                * 100,
                spikes["spiketimestamps"][:, cell],
            ]
        return spikes, spikes["spiketimestamps"], spikes_df

    def get_all_spikes(self):
        """
        Function which extracts all spikes from the .hdf5 results file.

        """
        all_spikes, spiketimestamps, spikes_df = self.get_spikes()
        return all_spikes, spiketimestamps, spikes_df

    def get_spikes_in_s(self, subset=False, thr=False):
        """
        Function that returns spikes from the .hdf5 results file, either from all
        cells or just a subset. Importantly, returns spikes as spikes in seconds.

        Parameters
        ----------
        subset: bool: If true, function will use self.spikes_subset to get spikes
        from a subset of clusters
        thr: bool: If true, function will use self.spikes_threshold to get spikes
        from a subset of clusters

        Returns
        -------
        spikes: dictonary: containing all information for the clusters in the
        results .hdf5 file.

        spiketimestamps: np.array: Only the spiketimestamps for each cell.

        spikes_df: pandas.DataFrame containing the information about each loaded
        cell (including the spiketimes as arrays in pandas field)

        """
        spikes, spiketimestamps, spikes_df = self.get_spikes(subset, thr)
        spikes["spiketimestamps"] = np.divide(
            spikes["spiketimestamps"], spikes["sampling"]
        )
        return spikes, spikes["spiketimestamps"], spikes_df

    def get_unit_waveforms(self, units, window, subset=100):
        waveforms_plot = plotly.subplots.make_subplots(
            rows=len(units),
            cols=1,
            subplot_titles=("Spike_waveforms"),
            shared_xaxes=False,
        )

        for unit, loc in zip(units, range(len(units))):

            selection_array = np.where(self.spikes["cluster_id"] == unit)[0]
            shape = self.spikes["shapes"][:, selection_array]
            waveforms = {}
            waveforms["mean"] = np.mean(shape, axis=1)
            bin_nr = np.shape(waveforms["mean"])[0]
            bins = np.arange(0, bin_nr, 1)
            waveforms["raw"] = np.empty((subset, bin_nr), dtype=float)
            nr_waves = np.shape(shape)[0]
            selection = np.linspace(0, nr_waves, subset, dtype=int)
            idx = 0
            for wave in selection:
                waveforms["raw"][idx, :] = shape[:, wave]
                idx = idx + 1
                waveforms_plot.add_trace(
                    go.Scattergl(x=bins, y=waveforms["raw"], mode="lines"),
                    row=loc + 1,
                    col=1,
                )

            waveforms_plot.add_trace(
                go.Scattergl(x=bins, y=waveforms["mean"], mode="lines"),
                row=loc + 1,
                col=1,
            )
        with window:
            display(waveforms_plot)


class Thresholds:
    """
    Threshold class
    define the different thresholds which are used to extract all or a subset of
    spikes from the hdf5 file. For this, different widgets are created which allow
    the user to interactively choose the parameters.
    """

    def __init__(self, Spikes, Stimuli):
        """
        Initialize function. Creates 6 widgets that allow to set thresholds.
        Then links these widgets together, so that values are expressed in different
        units.

        Parameters
        ----------
        Spikes: np.array spiketimestamps of the cells
        Stimuli: pandas.DataFrame: Contains the information for each stimulus

        Returns
        -------
        self.threshold_low_widget
        self.threshold_low_widget_min
        self.threshold_up_widget
        self.threshold_up_widget_min
        self.threshold_left_widget
        self.threshold_right_widget

        """

        self.Stimuli = Stimuli

        # Lower threshold
        self.threshold_low_widget = widgets.IntText(
            value=np.quantile(Spikes["spike_freq"][1, :], 0.05),
            description="Lower threshold:",
            disabled=False,
            style={"description_width": "initial"},
        )

        self.threshold_low_widget_min = widgets.FloatText(
            value=(
                np.quantile(Spikes["spike_freq"][1, :], 0.05)
                / (len(Stimuli.channel["Frame"]) / Stimuli.sampling_frequency)[0]
            )
            * 60,
            description="Lower threshold in spikes per minute:",
            disabled=False,
            style={"description_width": "initial"},
        )

        # Higher threshold

        self.threshold_up_widget = widgets.IntText(
            value=np.quantile(Spikes["spike_freq"][1, :], 0.95),
            description="Upper threshold:",
            disabled=False,
            style={"description_width": "initial"},
        )

        self.threshold_up_widget_min = widgets.FloatText(
            value=(
                np.quantile(Spikes["spike_freq"][1, :], 0.95)
                / (len(Stimuli.channel["Frame"]) / Stimuli.sampling_frequency)[0]
            )
            * 60,
            description="Upper threshold in spikes per minute:",
            disabled=False,
            style={"description_width": "initial"},
        )

        # Left Threshold
        self.threshold_left_widget = widgets.IntText(
            value=0,
            description="Left Threshold",
            disabled=False,
            style={"description_width": "initial"},
        )

        # Right Threshold
        self.threshold_right_widget = widgets.IntText(
            value=Spikes["nr_cells"],
            description="Right Threshold",
            disabled=False,
            style={"description_width": "initial"},
        )

        # Observations
        self.threshold_low_widget.observe(
            self.threshold_low_value_change_min, names="value"
        )
        self.threshold_up_widget.observe(
            self.threshold_up_value_change_min, names="value"
        )
        self.threshold_low_widget_min.observe(
            self.threshold_low_value_change, names="value"
        )
        self.threshold_up_widget_min.observe(
            self.threshold_up_value_change, names="value"
        )

    def threshold_low_value_change_min(self, change):

        new_a = change["new"]
        self.threshold_low_widget_min.value = (
            new_a
            / (len(self.Stimuli.channel["Frame"]) / self.Stimuli.sampling_frequency)[0]
            * 60
        )

    def threshold_up_value_change_min(self, change):
        new_a = change["new"]
        self.threshold_up_widget_min.value = (
            new_a
            / (len(self.Stimuli.channel["Frame"]) / self.Stimuli.sampling_frequency)[0]
            * 60
        )

    def threshold_low_value_change(self, change):
        new_a = change["new"]

        new_value = int(
            new_a
            * (len(self.Stimuli.channel["Frame"]) / self.Stimuli.sampling_frequency)[0]
            / 60
        )

        if (
            new_value - 1 < self.threshold_low_widget.value
            or new_value + 1 > self.threshold_low_widget.value
        ):
            self.threshold_low_widget.value = new_value

    def threshold_up_value_change(self, change):
        new_a = change["new"]
        new_value = int(
            new_a
            * (len(self.Stimuli.channel["Frame"]) / self.Stimuli.sampling_frequency)[0]
            / 60
        )

        if (
            new_value - 1 < self.threshold_up_widget.value
            or new_value + 1 > self.threshold_up_widget.value
        ):
            self.threshold_up_widget.value = new_value
