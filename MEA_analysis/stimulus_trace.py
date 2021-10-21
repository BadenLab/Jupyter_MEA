# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 12:23:16 2021
This file contains functions, and classes to deal and analyse the stimulus trace
obtained from 3Brain BioCam MEAs by exporting the trigger channel in .brw or .hdf
file formate.

@author: Marvin
"""
import MEA_analysis.backbone as backbone
import h5py
import pandas as pd
import numpy as np
from ipywidgets import widgets
import matplotlib.pyplot as plt
import scipy.signal as sg
import plotly.graph_objects as go
import pickle


class Stimulus_Extractor:
    """
    Stimulus trace class
    This class reads and handles the stimulus trace. The stimulus trace is plotted and
    than the user can select stimulus frames based on the plotted trigger signals.


    """

    def __init__(self, stimulus_file):
        """
        Initialize function. Opens the stimulus file and extracts the important
        data for further analysis. It also initializes object attributes which
        store information about how many stimuli have been defined, and which
        position these stimuli have in the stimulus channel.

        Parameters
        ----------
            stimulus_file: str: The file location of the stimulus trace.



        """

        # Check if file is .brw or .mat
        self.recording_folder = stimulus_file[: stimulus_file.rfind("/") + 1]
        format = backbone.get_file_ending(stimulus_file)
        if format == ".brw":
            with h5py.File(stimulus_file, "r") as f:
                self.channel = pd.DataFrame(
                    np.array([f["/3BData/Raw"]])[0, :], columns=["Voltage"]
                )
                self.max_Voltage = self.channel.Voltage.max(axis=0)
                self.min_Voltage = self.channel.Voltage.min(axis=0)
                self.half_Voltage = (
                    self.min_Voltage + (self.max_Voltage - self.min_Voltage) / 2
                )
                self.Frames = range(0, len(self.channel.index), 1)
                self.sampling_frequency = np.array(
                    f["/3BRecInfo/3BRecVars/SamplingRate"]
                )
                self.Time = self.Frames / self.sampling_frequency
                self.channel["Frame"] = self.Frames
                self.channel["Time_s"] = self.Time
                self.channel.Time_s = pd.to_timedelta(self.channel.Time_s, unit="s")
                self.channel.set_index("Time_s", inplace=True)

            # Stim nr input dialog:
            self.nr_stim_input = widgets.BoundedIntText(
                value=0,
                min=0,
                max=1000,
                step=1,
                description="Nr of Stimuli:",
                disabled=False,
            )
        self.switch = 0
        self.begins = []
        self.ends = []
        self.nr_stim = 0
        self.stimuli = pd.DataFrame(
            columns=[
                "Stimulus_name",
                "Begin_Fr",
                "End_Fr",
                "Trigger_Fr_relative",
                "Trigger_int",
                "Stimulus_index",
            ]
        )

    def plot_trigger_channel_new(self, dsf):
        """
        Plots trigger channel using plotly interactive plot and plotly widget.
        This allows for interactive stimulus selection.

        Parameters
        ----------
        dsf: str: example("200ms"). Downsample factor for downsampling the trigger
        channel.

        Returns
        -------
        self.f: The figure object of the plotly plot.

        """

        channel = self.channel
        channel["log"] = channel.Voltage > 3000
        channel = channel.resample(dsf).mean()
        # print(len(channel))
        self.f = go.FigureWidget(
            [
                go.Scattergl(
                    x=channel.Frame,
                    y=channel.log,
                    mode="lines+markers",
                    name="Trigger signal",
                )
            ]
        )
        self.f.update_traces(marker=dict(size=2, line=dict(width=0)))

        self.scatter = self.f.data[0]
        colors = ["#1f77b4"] * len(channel)
        self.scatter.marker.color = colors
        self.scatter.marker.size = [2] * len(channel)
        self.f.layout.hovermode = "closest"
        self.f.update_xaxes(title_text="Frames")
        self.f.update_yaxes(title_text="Trigger")
        self.scatter.on_click(self.update_point)
        return self.f

    # create our callback function
    def update_point(self, trace, points, selector):
        """
        Function to highlight selected points in the trigger channel plot

        Parameters
        ----------
        Not sure, copied from plotly interactive
        TODO: Look what are the input arguments here

        """

        c = list(self.scatter.marker.color)
        s = list(self.scatter.marker.size)

        for i in points.point_inds:
            if s[i] == 10:
                if self.switch == 0:
                    c[i] = "#1f77b4"
                    s[i] = 2
                    del self.ends[-1]
                    self.switch = 1

                else:
                    c[i] = "#1f77b4"
                    s[i] = 2
                    del self.begins[-1]
                    self.switch = 0

            elif self.switch == 0:
                c[i] = "#DFFF00"
                s[i] = 10
                self.begins.append(points.xs[0])
                self.switch = 1
            else:
                c[i] = "#ff2d00"
                s[i] = 10
                self.ends.append(points.xs[0])
                self.switch = 0

            with self.f.batch_update():
                self.scatter.marker.color = c
                self.scatter.marker.size = s

    def plot_trigger_channel(self, dsf):
        """
        Old version of trigger channel plotting, using matplotlib
        """
        channel = self.channel.resample(dsf).mean()
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(channel.Frame, channel.Voltage)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.xlabel("Frames")
        plt.ylabel("Voltage")
        plt.title("Stimulus channel complete")
        # fig = channel.plot.line(x='Frame', y='Voltage')
        # fig.show()
        return channel, fig, ax
        # channel.index.resample('3T').sum()

    def trigger_channel_for_selection(self, dsf):
        """
        Function that defines n number of stimuli and creates equal number of
        subplots for the matplot plot, so the user can select one stimulus per
        subplot. Deprecated.

        """

        if self.nr_stim_input.value == 0:
            print("Set number of stimulus to 0, no stimuli can be selected")
            return

        channel = self.channel.resample(dsf).mean()
        self.stim_select_fig, self.stim_select_axs = plt.subplots(
            nrows=self.nr_stim_input.value,
            ncols=1,
            figsize=(9, self.nr_stim_input.value * 2),
        )

        if self.nr_stim_input.value > 1:
            for i in range(self.nr_stim_input.value):
                self.stim_select_axs[i].plot(channel.Frame, channel.Voltage)
                self.stim_select_axs[i].spines["top"].set_visible(False)
                self.stim_select_axs[i].spines["right"].set_visible(False)
        else:
            self.stim_select_axs.plot(channel.Frame, channel.Voltage)

        plt.xlabel("Frames")
        plt.ylabel("Voltage")
        self.stim_select_fig.suptitle("Stimulus channel complete")

    def get_stim_range(self):
        """
        Old version of calculating the stimulus range and trigger times, when the
        stimulus trace was plotted with matplotlib. Deprecated.
        """
        self.stimuli = pd.DataFrame(
            columns=[
                "Stimulus_name",
                "Begin_Fr",
                "End_Fr",
                "Trigger_Fr_relative",
                "Trigger_int",
                "Stimulus_index",
            ]
        )
        for i in range(0, self.nr_stim_input.value):
            if self.nr_stim_input.value == 1:
                limits_temp = np.array(self.stim_select_axs.get_xlim())
            else:
                limits_temp = np.array(self.stim_select_axs[i].get_xlim())

            if limits_temp[0] < 0:
                limits_temp[0] = 0
            if limits_temp[1] < 0:
                limits_temp[1] = 0
            limits_int = limits_temp.astype(int)
            channel_cut = self.channel[limits_int[0] : limits_int[1]]
            channel_log = channel_cut.Voltage > self.half_Voltage
            peaks = sg.find_peaks(channel_log, height=1, plateau_size=2)

            peaks[0][:] = peaks[0][:] + limits_temp[0]
            peaks_left = peaks[1]["left_edges"] + limits_temp[0]
            stim_begin = int(peaks_left[0])
            trigger_interval = np.diff(peaks_left)
            min_trigger_interval = np.min(trigger_interval)

            stim_end = int(
                peaks_left[-1] + min_trigger_interval
            )  # Adds time after the last trigger, to get the whole stimulus
            # Failsafe: If last trigger was close to the end of the recording make last frame stimulus end
            if len(self.channel) < stim_end:
                stim_end = len(self.channel)

            nr_trigger = len(peaks_left)
            plot_ones = np.ones(nr_trigger) * self.max_Voltage

            df_temp = pd.DataFrame()
            df_temp["Stimulus_name"] = ""
            df_temp["Begin_Fr"] = [stim_begin]
            df_temp["End_Fr"] = [stim_end]
            df_temp["Trigger_Fr_relative"] = [peaks_left - peaks_left[0]]
            df_temp["Trigger_int"] = [trigger_interval]
            df_temp["Stimulus_index"] = [i]
            df_temp["Stimulus_repeat_logic"] = [0]
            df_temp["Stimulus_repeat_sublogic"] = [0]
            self.stimuli = self.stimuli.append(df_temp, ignore_index=True)

            # Update the figure from stimulus selection with trigger points
            if self.nr_stim_input.value == 1:
                self.stim_select_axs.scatter(peaks_left, plot_ones, color="hotpink")
                self.stim_select_axs.set_xlim(limits_int[0], limits_int[1])
            else:
                self.stim_select_axs[i].scatter(peaks_left, plot_ones, color="hotpink")
                self.stim_select_axs[i].set_xlim(limits_int[0], limits_int[1])

        self.stimuli.set_index("Stimulus_index", inplace=True)

    def get_stim_range_new(self):
        """
        Function that calculates the stimulus range and find the trigger times
        based on which stimulus borders were selected by the user in the interactive
        plotly graph beforehand.

        Parameters
        ----------
        Needs a plotted stimulus trigger plotly graph and selected stimulus borders
        for at least 1 stimulus.

        Returns
        -------
        returns to self a dataframe containing the stimulus information
        """

        print(self.nr_stim)
        for i in range(self.nr_stim, len(self.begins)):

            limits_temp = np.array(
                [self.begins[i] - 1000, self.ends[i] + 1000], dtype=int
            )

            if limits_temp[0] < 0:
                limits_temp[0] = 0
            if limits_temp[1] < 0:
                limits_temp[1] = 0
            limits_int = limits_temp.astype(int)
            channel_cut = self.channel[limits_int[0] : limits_int[1]]
            channel_log = channel_cut.Voltage > self.half_Voltage
            peaks = sg.find_peaks(channel_log, height=1, plateau_size=2)

            peaks[0][:] = peaks[0][:] + limits_temp[0]
            peaks_left = peaks[1]["left_edges"] + limits_temp[0]
            stim_begin = int(peaks_left[0])

            trigger_interval = np.diff(peaks_left)
            min_trigger_interval = np.min(trigger_interval)

            stim_end = int(
                peaks_left[-1] + min_trigger_interval
            )  # Adds time after the last trigger, to get the whole stimulus
            # Failsafe: If last trigger was close to the end of the recording make last frame stimulus end
            if len(self.channel) < stim_end:
                stim_end = len(self.channel)

            peaks_left = np.append(peaks_left, peaks_left[-1] + min_trigger_interval)
            nr_trigger = len(peaks_left)
            plot_ones = np.ones(nr_trigger)

            df_temp = pd.DataFrame()
            df_temp["Stimulus_name"] = ""
            df_temp["Begin_Fr"] = [stim_begin]
            df_temp["End_Fr"] = [stim_end]
            df_temp["Trigger_Fr_relative"] = [peaks_left - peaks_left[0]]
            df_temp["Trigger_int"] = [trigger_interval]
            df_temp["Stimulus_index"] = [i]
            df_temp["Stimulus_repeat_logic"] = [0]
            df_temp["Stimulus_repeat_sublogic"] = [0]
            self.stimuli = self.stimuli.append(df_temp, ignore_index=True)

            self.f.add_trace(
                go.Scattergl(x=peaks_left, y=plot_ones, name="Stimulus " + str(i))
            )
            # Update the figure from stimulus selection with trigger points

        self.nr_stim = len(self.ends)

    def get_stim_start_end(self, limits):
        """
        Function that cuts out one specific limit from the whole trigger channel

        Parameters
        ----------
        limits: np.array() size(2): Array containing the left and right limits
        of the stimulus window in frames

        Returns
        -------
        channel: The cut out trigger channel for one stimulus
        """
        for i in range(0, self.nr_stim_input.value):
            channel = self.channel.Voltage[limits[i, 0] : limits[i, 1]]
            return channel

    def get_changed_names(self, overview):
        """
        Adds information about stimulus names to the previously created plotly
        plot of the trigger channel.

        Parameters
        ----------
        overview: qgrid object: The qgrid overview over the stimulus data frame
        overview: dataframe: The stimulus dataframe can also be provided directly.

        Returns
        -------

        """
        try:
            self.stimuli = overview.get_changed_df()
            self.stimuli.set_index("Stimulus_index", inplace=True)
        except AttributeError:
            self.stimuli = overview

        for stimulus in range(len(self.stimuli)):
            self.f.data[stimulus + 1].name = self.stimuli["Stimulus_name"].loc[stimulus]

    def load_from_saved(self, savefile, show_plot=False):
        """
        Function that loads a pickled stimulus dataframe

        Parameters
        ----------
        savefile: The pickled stimulus dataframe
        show_plot=False: IF true, the complete trigger channel is plotted with
        respective stimulus ranges and names

        Returns
        -------
        Adds stimulus dataframe and trigger channel figure to self.
        """
        stimulus_df = pd.read_pickle(savefile, compression="zip")
        self.begins = stimulus_df["Begin_Fr"].to_list()
        self.ends = stimulus_df["End_Fr"].to_list()
        self.stimuli = stimulus_df
        self.plot_trigger_channel_new("200ms")
        self.get_stim_range_new()
        self.get_changed_names(stimulus_df)
        if show_plot:
            self.f.show()
