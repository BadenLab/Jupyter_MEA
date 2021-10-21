# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 16:15:05 2021
Single stimulus analysis functions
@author: Marvin
"""
import numpy as np
import qgrid
import pyspike as spk
from MEA_analysis import stimulus_and_spikes as sp
from MEA_analysis import spike_plotly
import sys
from IPython.display import display
import ipywidgets as widgets
from scipy import signal
import math
import matplotlib.pyplot as plt
from importlib import reload

reload(spike_plotly)


class Single_stimulus_spikes:
    def __init__(self, cells_df, stimulus_df, sampling_freq=17852.767845719834):
        self.cells_df = cells_df
        self.stimulus_df = stimulus_df
        self.Colours = []
        self.sampling_freq = sampling_freq

    def load_spikes_for_stimulus(self, stimulus_id):
        self.spikes_stimulus = self.cells_df.xs(stimulus_id, level="Stimulus ID").copy()
        self.stimulus_info = self.stimulus_df.stimuli.loc[stimulus_id]
        self.trigger_complete = np.array(
            self.stimulus_df.stimuli["Trigger_Fr_relative"][stimulus_id]
        )

        return qgrid.show_grid(self.spikes_stimulus)

    def load_spikes_for_stimulus_name(self, name):
        self.spikes_stimulus = self.cells_df.xs(name, level="Stimulus name").copy()
        self.stimulus_info = self.stimulus_df.xs(name, level="Stimulus_name")

        return qgrid.show_grid(self.spikes_stimulus)

    def plot_raster_whole_stimulus(self, cell_idx=0):

        spikes, spiketrains = sp.get_spikes_whole_stimulus(
            self.spikes_stimulus,
            self.trigger_complete,
            cell_idx,
            int(self.stimulus_info["Stimulus_repeat_logic"]),
            self.sampling_freq,
        )
        raster_plot = spike_plotly.plot_raster_whole_stimulus(
            spiketrains,
            int(self.stimulus_info["Stimulus_repeat_logic"]),
            int(self.stimulus_info["Stimulus_repeat_sublogic"]),
            self.Colours.axcolours,
            self.Colours.LED_names,
        )
        return raster_plot

    def plot_raster_whole_stimulus_new(self, cell_idx=0):

        spikes, spiketrains = sp.get_spikes_whole_stimulus(
            self.spikes_stimulus,
            self.trigger_complete,
            cell_idx,
            int(self.stimulus_info["Stimulus_repeat_logic"]),
            self.sampling_freq,
        )
        cell_df = self.spikes_stimulus.loc[cell_idx]
        raster_plot = spike_plotly.plot_raster_whole_stimulus_new(
            cell_df,
            spiketrains,
            int(self.stimulus_info["Stimulus_repeat_logic"]),
            int(self.stimulus_info["Stimulus_repeat_sublogic"]),
            self.Colours.axcolours,
            self.Colours.LED_names,
        )

        return raster_plot

    def define_output_window(self, output_window):
        self.out_window = output_window

    def plot_raster_whole_stimulus_from_grid(self, event, qgrid):
        index = event["new"][0]
        cell_idx = self.spikes_stimulus[index : index + 1].index[0][0]
        raster_plot = self.plot_raster_whole_stimulus(cell_idx)
        self.out_window.clear_output()
        with self.out_window:
            display(raster_plot)

    def plot_raster_whole_stimulus_from_grid_new(self, event, qgrid):

        index = event["new"][0]
        cell_idx = self.spikes_stimulus[index : index + 1].index[0][0]
        raster_plot = self.plot_raster_whole_stimulus_new(cell_idx)
        self.out_window.clear_output()
        with self.out_window:
            display(raster_plot)

    def spikes_conv_gauss_all(
        self, resolution=0.001, penalty=1, sigma=10, plot_gaussian=False
    ):
        self.spikes_stimulus = spikes_conv_gauss_all(
            self.spikes_stimulus,
            self.trigger_complete,
            int(self.stimulus_info["Stimulus_repeat_logic"]),
            self.sampling_freq,
            resolution,
            penalty,
            sigma,
            plot_gaussian,
        )

    def spikes_psth_all(self, bin_size=0.05):
        self.spikes_stimulus = spikes_psth_all(
            self.spikes_stimulus,
            self.trigger_complete,
            int(self.stimulus_info["Stimulus_repeat_logic"]),
            self.sampling_freq,
            bin_size,
        )

    def spikes_isi_all(self):
        self.spikes_stimulus = spikes_isi_all(
            self.spikes_stimulus,
            self.trigger_complete,
            int(self.stimulus_info["Stimulus_repeat_logic"]),
            self.sampling_freq,
        )

    def spikes_sync_all(self):
        self.spikes_stimulus = spikes_sync_all(
            self.spikes_stimulus,
            self.trigger_complete,
            int(self.stimulus_info["Stimulus_repeat_logic"]),
            self.sampling_freq,
        )


def calculate_quality_index(spike_df, trigger_complete, repeat_logic, sampling_freq):
    cells = np.array(spike_df.index.get_level_values(0))
    nr_cells = np.shape(cells)[0]
    # Initialize results dataframe
    # results_dataframe = pd.DataFrame(columns=('Cell index', 'max_isi', 'std_sync', 'max_psth', 'total_qc', 'nr_spikes'))
    row = 0
    max_isi = []
    std_sync = []
    max_psth = []
    total_qc = []
    stimulus_spikes = []
    histograms = []
    bins = []
    mean_isi = []
    mean_sync = []
    total_qc_new = []

    for cell in cells:
        # print(cell)

        try:
            spikes, spiketrains = sp.get_spikes_whole_stimulus(
                spike_df, trigger_complete, cell, repeat_logic, sampling_freq
            )
            avrg_isi_profile = spk.isi_profile(spiketrains)
            spike_sync_profile = spk.spike_sync_profile(spiketrains)
            psth = spk.psth(spiketrains, bin_size=0.05)
            x, y = avrg_isi_profile.get_plottable_data()
            x1, y1 = spike_sync_profile.get_plottable_data()
            x2, y2 = psth.get_plottable_data()
            max_isi.append(np.max(y))
            std_sync.append(np.std(y1))
            max_psth.append(np.max(y2))
            total_qc.append(np.max(y) * np.std(y1) * np.max(y2))
            # total_qc.append(np.max(y)*np.std(y1)*(np.max(y2)-np.quantile(y2, 0.25)))
            stimulus_spikes.append(np.sum(y2))
            histograms.append(y2)
            bins.append(x2)
            mean_isi.append(np.mean(y))
            mean_sync.append(np.mean(y1))
            total_qc_new.append(
                (np.max(y1) - np.mean(y) + (np.max(y2) - np.mean(y2))) * np.std(y1)
            )

            # results_dataframe.loc[row] = [cell, np.max(y), np.std(y1), np.max(y2), np.max(y)*np.std(y1)*np.max(y2),
            # np.sum(y2)]

        except:
            print("Unexpected error:", sys.exc_info()[0])
            # results_dataframe.loc[row] = [cell, 0, 0, 0, 0, 0]
            max_isi.append(0)
            std_sync.append(0)
            max_psth.append(0)
            total_qc.append(0)
            stimulus_spikes.append(0)
            histograms.append(0)
            bins.append(0)
            mean_isi.append(1)
            mean_sync.append(0)
            total_qc_new.append(0)
        # row = row+1

    spike_df["Max isi"] = max_isi
    spike_df["Std sync"] = std_sync
    spike_df["Max psth"] = max_psth
    spike_df["Total qc"] = total_qc
    spike_df["stimulus spikes"] = stimulus_spikes
    spike_df["Histogram"] = histograms
    spike_df["Bins"] = bins
    spike_df["ISI mean"] = mean_isi
    spike_df["mean sync"] = mean_sync
    spike_df["total qc new"] = total_qc_new
    return spike_df


def spikes_conv_gauss_all(
    spike_df,
    trigger_complete,
    repeat_logic,
    sampling_freq,
    resolution=0.0001,
    penalty=1,
    sigma=10,
    plot_gaussian=False,
):
    # Get indices of cells in the dataframe:
    cells = np.array(spike_df.index.get_level_values(0))
    nr_cells = np.shape(cells)[0]

    # Define lists that will store the output before integration into the dataframe
    gauss_bins_all = []
    gauss_x_all = []
    gauss_filter_all = []
    gauss_average_all = []

    # Define gaussian window:
    window_size = sigma * 3
    gauss_window = signal.windows.gaussian(window_size, std=sigma)
    if plot_gaussian:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(gauss_window)

    for cell in cells:
        try:
            spikes, spiketrains = sp.get_spikes_whole_stimulus(
                spike_df, trigger_complete, cell, repeat_logic, sampling_freq
            )

            # Define bining parameters
            nr_bins = math.ceil(spiketrains[0].t_end / resolution) * 2 - 2
            nr_bins_simple = math.ceil(spiketrains[0].t_end / resolution - 1)

            # Prepare arrays for loop over repeats
            repeats = len(spiketrains)
            gauss_bins = np.zeros(
                (nr_bins, repeats)
            )  # This stores the inital bined spikes
            gauss_x = np.zeros((nr_bins))  # This stores the x values for all repeats
            gauss_filter = np.zeros(
                (nr_bins, repeats)
            )  # Stores the convolved traces for all repeats
            # Loop
            for repeat in range(repeats):
                # Calculate histogram
                psth = spk.psth([spiketrains[repeat]], resolution)
                x, y = psth.get_plottable_data()
                gauss_bins[:, repeat] = y
                # Store x values
                if repeat == 0:
                    gauss_x = x
                # Convolve with gaussian kernel defined above
                gauss_filter[:, repeat] = np.convolve(y, gauss_window, mode="same") / 2

            gauss_filter_corrected = (
                gauss_filter + penalty
            )  # This adds the penalty value
            # Averaging, first we assigne the first repeat, the other will be multiplied with it
            gauss_average = gauss_filter_corrected[:, 0]
            for repeat in range(1, repeats):
                gauss_average = np.multiply(
                    gauss_average, gauss_filter_corrected[:, repeat]
                )
            if penalty >= 1:
                gauss_average = (
                    gauss_average - penalty
                )  # remove the penalty value from the averaged trace

            # Add the results to the output list
            gauss_bins_all.append(gauss_bins)
            gauss_x_all.append(gauss_x)
            gauss_filter_all.append(gauss_filter)
            gauss_average_all.append(gauss_average)

        # Add error handling
        except:
            print(str(cell) + " Failed")
            gauss_bins_all.append(0)
            gauss_x_all.append(0)
            gauss_filter_all.append(0)
            gauss_average_all.append(0)

    # Store into dataframe
    spike_df["Gauss_bins"] = gauss_bins_all
    spike_df["Gauss_x"] = gauss_x_all
    spike_df["Gauss_filter"] = gauss_filter_all
    spike_df["Gauss_average"] = gauss_average_all

    return spike_df


def spikes_psth_all(
    spike_df, trigger_complete, repeat_logic, sampling_freq, bin_size=0.05
):
    # Get indices of cells in the dataframe:
    cells = np.array(spike_df.index.get_level_values(0))
    # nr_cells = np.shape(cells)[0]

    # Define lists that will store the output before integration into the dataframe
    psth_all = []
    psth_x_all = []

    for cell in cells:
        try:
            spikes, spiketrains = sp.get_spikes_whole_stimulus(
                spike_df, trigger_complete, cell, repeat_logic, sampling_freq
            )

            psth = spk.psth(spiketrains, bin_size)
            x, y = psth.get_plottable_data()
            psth_all.append(y)
            psth_x_all.append(x)
        except:
            psth_all.append(0)
            psth_x_all.append(0)

    spike_df["PSTH"] = psth_all
    spike_df["PSTH_x"] = psth_x_all
    return spike_df


def spikes_isi_all(spike_df, trigger_complete, repeat_logic, sampling_freq):
    # Get indices of cells in the dataframe:
    cells = np.array(spike_df.index.get_level_values(0))

    # Define lists that will store the output before integration into the dataframe
    isi_all = []
    isi_x_all = []

    for cell in cells:
        try:
            spikes, spiketrains = sp.get_spikes_whole_stimulus(
                spike_df, trigger_complete, cell, repeat_logic, sampling_freq
            )

            isi = spk.isi_profile(spiketrains)
            x, y = isi.get_plottable_data()
            isi_all.append(y)
            isi_x_all.append(x)
        except:
            isi_all.append(0)
            isi_x_all.append(0)

    spike_df["ISI"] = isi_all
    spike_df["ISI_x"] = isi_x_all
    return spike_df


def spikes_sync_all(spike_df, trigger_complete, repeat_logic, sampling_freq):
    # Get indices of cells in the dataframe:
    cells = np.array(spike_df.index.get_level_values(0))

    # Define lists that will store the output before integration into the dataframe
    sync_all = []
    sync_x_all = []

    for cell in cells:
        try:
            spikes, spiketrains = sp.get_spikes_whole_stimulus(
                spike_df, trigger_complete, cell, repeat_logic, sampling_freq
            )

            sync = spk.spike_sync_profile(spiketrains)
            x, y = sync.get_plottable_data()
            sync_all.append(y)
            sync_x_all.append(x)
        except:
            sync_all.append(0)
            sync_x_all.append(0)

    spike_df["SYNC"] = sync_all
    spike_df["SYNC_x"] = sync_x_all
    return spike_df
