# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 15:09:02 2021
This file contains all the plotting functions for the analysis
(This might have to be changed if code becomes too long in future)
@author: Marvin
"""

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly
import pyspike as spk
from ipywidgets import interact, interact_manual, interactive
from IPython.display import display
import pandas as pd
from ipywidgets import widgets
import math
from sklearn.cluster import KMeans
from MEA_analysis import govardovskii as gvdski
from MEA_analysis import Colour_template as ct
from importlib import reload
import re

reload(ct)


class Recording_overview:
    """
    These functions plot an overview over how many spikes are found in each cluster
    and applies different, userdefined thresholds to the data, so a subset of spikes
    can be selected
    """

    def __init__(self, spikes):
        """
        Initialize function. Assignes the spikes data to the object.

        Parameters
        ----------
        spikes: dictonary containing the information about spiking for each cell

        Returns
        -------

        """

        self.spikes = spikes

    def plot_basic_recording_information(self, thresholds):
        """
        Plots an overview figure with interactively moveable thresholds for
        number of spikes and number of clusters.

        Parameters
        ----------
        thresholds: object: Object of the Threshold class from spike_extractor.

        Returns
        -------
        Shows figure
        """

        # Lets check how many clusters we have and how many spikes per cluster
        # Plot barplot with number of spikes per cluster
        self.spikes_overview_fig, self.overview_ax = plt.subplots(figsize=(15, 5))
        self.overview_ax.bar(
            self.spikes["spike_freq"][0, :], self.spikes["spike_freq"][1, :]
        )
        self.overview_ax.set_yscale("log")
        # Plot interactive threshold line Up Lower Thresholds
        xvalues = range(0, self.spikes["nr_cells"])
        y_value_low = np.ones(self.spikes["nr_cells"], dtype=int) * np.quantile(
            self.spikes["spike_freq"][1, :], 0.05
        )
        (self.threshold_line_low,) = self.overview_ax.plot(
            xvalues, y_value_low, color="red", label="Threshold Lower"
        )
        # y_value_up = np.ones(data['nr_cells'], dtype=int)*np.quantile(data['spike_freq'][1, :], 0.95)
        (self.threshold_line_up,) = self.overview_ax.plot(
            xvalues, y_value_low, color="green", label="Threshold Upper"
        )

        # Plot interactive threshold left right
        self.threshold_line_left = self.overview_ax.axvline(
            -1, 0, self.spikes["max_spikes"], color="black"
        )
        self.threshold_line_right = self.overview_ax.axvline(
            self.spikes["nr_cells"], 0, self.spikes["max_spikes"], color="black"
        )

        self.overview_ax.spines["top"].set_visible(False)
        self.overview_ax.spines["right"].set_visible(False)
        plt.legend()
        plt.xlabel("Cells")
        plt.ylabel("Nr_spikes")
        plt.title("Overview: Nr of spikes per cell in recording")

        interact(self.update_threshold_low, threshold=thresholds.threshold_low_widget)
        display(thresholds.threshold_low_widget_min)
        interact(self.update_threshold_up, threshold=thresholds.threshold_up_widget)
        display(thresholds.threshold_up_widget_min)
        interact(self.update_threshold_left, threshold=thresholds.threshold_left_widget)
        # display(threshold_left_widget)
        interact(
            self.update_threshold_right, threshold=thresholds.threshold_right_widget
        )

    def update_threshold_low(self, threshold):
        self.threshold_line_low.set_ydata(
            np.ones(self.spikes["nr_cells"], dtype=int) * threshold
        )
        plt.show()

    def update_threshold_up(self, threshold):
        self.threshold_line_up.set_ydata(
            np.ones(self.spikes["nr_cells"], dtype=int) * threshold
        )
        plt.show()

    def update_threshold_left(self, threshold):
        self.threshold_line_left.set_xdata(threshold)
        plt.show()

    def update_threshold_right(self, threshold):
        self.threshold_line_right.set_xdata(threshold)
        plt.show()


class ArrayFigure:
    cell_indices = []

    def __init__(self, spikes_df, spike_class):
        self.spikes_df = spikes_df
        self.spike_class = spike_class

    def plot_locations(self):
        """
        Plot an overview plot, which shows how many spikes exist for each cluster
        all over the array

        Parameters
        ----------
        spikes: dictonary: information of the cells in the recording

        Returns
        -------
        Plots cluster location plot

        """

        spikes = self.spikes_df
        try:

            plt.close(locations_plt)
        except:
            pass

        self.locations_plt = go.FigureWidget(
            [
                go.Scattergl(
                    x=spikes["Centres x"],
                    y=spikes["Centres y"],
                    hovertemplate="%{text}<extra></extra>",
                    text=[
                        "Cell index {}".format(i)
                        for i in spikes["Cell index"].to_numpy()
                    ],
                    showlegend=False,
                    mode="markers",
                    marker=dict(
                        color=spikes["Cell index"].astype(int), size=spikes["Area"],
                    ),
                )
            ]
        )

        # colors = np.random.rand(len(spikes))
        # locations_plt = px.scatter(
        #     spikes,
        #     x="Centres x",
        #     y="Centres y",
        #     color=spikes["Cell index"].astype(int),
        #     size="Area",
        #     hover_data=["Cell index"],
        # )

        self.locations_plt.update_layout(
            height=1000,
            width=1000,
            title_text="Cell locations and spike numbers per cluster",
            coloraxis_colorbar=dict(title="Cell index"),
        )

        self.scatter = self.locations_plt.data[0]
        self.scatter.on_click(self.get_point_cell)
        self.locations_plt.layout.on_change(self.handle_zoom, "mapbox_zoom")
        self.default_colors = self.scatter.marker.color
        self.default_sizes = self.scatter.marker.size
        return self.locations_plt

    def get_point_cell(self, trace, points, selector):

        if not selector.ctrl:
            print(len(self.locations_plt["data"]))
            for i in range(1, len(self.locations_plt["data"])):
                print(i)
                self.locations_plt["data"][i]["marker"]["color"] = "#ffffff"
            self.cell_indices = []
        # Transforming the string containing the cell index into integer
        cell_idx = int(
            re.findall(
                r"\d+", self.locations_plt["data"][0]["text"][points.point_inds[0]]
            )[0]
        )
        # append to the list of cell indices
        self.cell_indices.append(cell_idx)

        # create scatterplot to highlight selected points
        self.locations_plt.add_scatter(
            x=points.xs,
            y=points.ys,
            mode="markers",
            showlegend=False,
            hoverinfo="skip",
            marker=dict(
                color="#000000",
                size=self.scatter.marker.size[points.point_inds[0]] + 10,
            ),
        )
        # clears previous output
        self.window.clear_output()
        # Call function that provides waveforms
        self.spike_class.get_unit_waveforms(self.cell_indices, window=self.window)

    def handle_zoom(self, layout, mapbox_zoom):
        print("Shit")
        # print("new mapbox_zoom:", mapbox_zoom)


def plot_waveforms(waveforms):
    waveforms_plot = go.Figure()


def plot_raster_all_cells(spikes, spiketrain):

    """
    Plot an overview raster plot with all spikes from all cells in the dataset.
    OpenGL is used to compress data and plot faster.

    Parameters
    ----------
    spikes: dictonary containing the information for spikes of each cell

    spiketrain: pyspike.object: Pyspike object containing spiketrains for all cells
    over the whole recording.

    Returns
    -------
    raster_plot: plotly object: Figure showing the spiketrain of all cells
    in the recording

    """
    try:
        plt.close(raster_plot)
    except:
        pass

    avrg_isi_profile = spk.isi_profile(spiketrain)

    raster_plot = plotly.subplots.make_subplots(
        rows=2, cols=1, subplot_titles=("Spiketrains"), shared_xaxes=True
    )

    # First, plot spiketrain
    for cell in range(len(spiketrain)):

        spikes_temp = spiketrain[cell].spikes
        # spikes_temp = spikes['Spiketimestamps'].loc[cell][:spikes['Nr of Spikes'].loc[cell]]

        nr_spikes = np.shape(spikes_temp)[0]
        yvalue = np.ones(nr_spikes) * spikes["Cell index"].loc[cell]
        raster_plot.add_trace(
            go.Scattergl(
                mode="markers",
                x=spikes_temp,
                y=yvalue,
                name="Cell " + str(spikes["Cell index"].loc[cell]),
                marker=dict(color="Black", size=2),
            )
        )
        # raster_axis.set_ylabel('Cell ID')
        # raster_axis.spines['top'].set_visible(False)
        # raster_axis.spines['right'].set_visible(False)

        # PLot ISI distance average
    x, y = avrg_isi_profile.get_plottable_data()
    raster_plot.add_trace(
        go.Scatter(
            x=x, y=y, name="Average Isi distance", line=dict(color="Black", dash="dot")
        ),
        row=2,
        col=1,
    )
    # Set title labels
    raster_plot.update_xaxes(title_text="Time in Seconds", row=2, col=1)
    raster_plot.update_yaxes(title_text="Cell Index", row=1, col=1)
    raster_plot.update_yaxes(title_text="ISI", row=2, col=1)
    # raster_axis[0].set_title('Spiketrains in Recording')
    raster_plot.show()

    return raster_plot


# Color template
class Colour_template:

    nr_stimuli = 5
    stim_names = ["FFF_4_UV", "FFF_4", "FFF_6", "Silent_Substitution", "Contrast_Step"]
    indices = pd.Index(stim_names)
    plot_colour_dataframe = pd.DataFrame(
        columns=("Colours", "Description"), index=indices
    )

    # Colours
    Colours = []
    Colours.append(
        [
            "#fe7cfe",
            "#000000",
            "#7c86fe",
            "#000000",
            "#8afe7c",
            "#000000",
            "#fe7c7c",
            "#000000",
        ]
    )  # FFF_4_UV
    Colours.append(
        [
            "#7c86fe",
            "#000000",
            "#7cfcfe",
            "#000000",
            "#8afe7c",
            "#000000",
            "#fe7c7c",
            "#000000",
        ]
    )  # FFF_4
    # Colours.append(["#fe7cfe","#000000","#7c86fe","#000000","#7cfcfe","#000000","#8afe7c","#000000","#fafe7c","#000000","#fe7c7c","#000000"]) #FFF_6
    Colours.append(
        [
            "#fe7c7c",
            "#000000",
            "#fafe7c",
            "#000000",
            "#8afe7c",
            "#000000",
            "#7cfcfe",
            "#000000",
            "#7c86fe",
            "#000000",
            "#fe7cfe",
            "#000000",
        ]
    )  # FFF_6
    Colours.append(
        [
            "#E2E2E2",
            "#EF553B",
            "#E2E2E2",
            "#00CC96",
            "#E2E2E2",
            "#636EFA",
            "#E2E2E2",
            "#AB63FA",
            "#E2E2E2",
            "#7F7F7F",
            "#DC3912",
            "#7F7F7F",
            "#109618",
            "#7F7F7F",
            "#3366CC",
            "#7F7F7F",
            "#990099",
            "#7F7F7F",
        ]
    )  # Silent substitution
    Colours.append(
        [
            "#ffffff",
            "#000000",
            "#e6e6e6",
            "#000000",
            "#cdcdcd",
            "#000000",
            "#b4b4b4",
            "#000000",
            "#9b9b9b",
            "#000000",
            "#828282",
            "#000000",
            "#696969",
            "#000000",
            "#505050",
            "#000000",
            "#373737",
            "#000000",
            "#1e1e1e",
            "#000000",
        ]
    )  # Contrast Steps

    # Descriptions ("Names")
    Colours_names = []
    Colours_names.append(
        ["LED_630", "OFF", "LED_505", "OFF", "LED_420", "LED_360", "OFF"]
    )  # FFF_4_UV
    Colours_names.append(
        ["LED_630", "OFF", "LED_505", "OFF", "LED_480", "OFF", "LED_420", "OFF"]
    )  # FFF_4
    Colours_names.append(
        [
            "LED_630",
            "OFF",
            "LED_560",
            "OFF",
            "LED_505",
            "OFF",
            "LED_480",
            "OFF",
            "LED_420",
            "OFF",
            "LED_360",
            "OFF",
        ]
    )  # FFF_6
    Colours_names.append(
        [
            "Background_On",
            "Red Cone OFF",
            "Background_On",
            "Green Cone OFF",
            "Background_On",
            "Blue Cone OFF",
            "Background_On",
            "VS Cone OFF",
            "Background_On",
            "Background_Off",
            "Red Cone ON",
            "Background_Off",
            "Green Cone ON",
            "Background_Off",
            "Blue Cone ON",
            "Background_Off",
            "VS Cone ON",
            "Background_Off",
        ]
    )  # Silent substitution
    Colours_names.append(
        [
            "100%_On",
            "OFF",
            "90%_On",
            "OFF",
            "80%_On",
            "OFF",
            "70%_On",
            "OFF",
            "60%_On",
            "OFF",
            "50%_On",
            "OFF",
            "40%_On",
            "OFF",
            "30%_On",
            "OFF",
            "20%_On",
            "OFF",
            "10%_On",
            "OFF",
        ]
    )  # Contrast Steps

    # Populate dataframe
    def __init__(self):
        for stimulus in range(self.nr_stimuli):
            self.plot_colour_dataframe.loc[self.stim_names[stimulus]] = [
                self.Colours[stimulus],
                self.Colours_names[stimulus],
            ]
        self.plot_colour_dataframe

        self.stimulus_select = widgets.RadioButtons(
            options=list(self.plot_colour_dataframe.index),
            layout={"width": "max-content"},
            description="Colourset",
            disabled=False,
        )

    def select_preset_colour(self):
        return self.stimulus_select

    def pickstimcolour(self, selected_stimulus):
        self.selected_stimulus = selected_stimulus
        words = []
        nr_trigger = len(self.plot_colour_dataframe.loc[selected_stimulus]["Colours"])
        # print(nr_trigger)
        for trigger in range(nr_trigger):
            # print(trigger)
            selected_word = self.plot_colour_dataframe.loc[selected_stimulus][
                "Description"
            ][trigger]
            words.append(selected_word)

        items = [
            widgets.ColorPicker(
                description=w,
                value=self.plot_colour_dataframe.loc[selected_stimulus]["Colours"][
                    trigger
                ],
            )
            for w, trigger in zip(words, range(nr_trigger))
        ]
        first_box_items = []
        second_box_items = []
        third_box_items = []
        fourth_box_items = []

        a = 0
        for trigger in range(math.ceil(nr_trigger / 4)):
            try:
                first_box_items.append(items[a])
                a = a + 1
                second_box_items.append(items[a])
                a = a + 1
                third_box_items.append(items[a])
                a = a + 1
                fourth_box_items.append(items[a])
                a = a + 1
            except:
                break

        first_box = widgets.VBox(first_box_items)
        second_box = widgets.VBox(second_box_items)
        thrid_box = widgets.VBox(third_box_items)
        fourth_box = widgets.VBox(fourth_box_items)
        self.colour_select_box = widgets.HBox(
            [first_box, second_box, thrid_box, fourth_box]
        )
        return self.colour_select_box

    def changed_selection(self):
        self.axcolours = []
        self.LED_names = []
        nr_trigger = len(
            self.plot_colour_dataframe.loc[self.selected_stimulus]["Colours"]
        )
        for trigger in range(math.ceil(nr_trigger / 4)):
            try:
                self.axcolours.append(
                    self.colour_select_box.children[0].children[trigger].value
                )
                self.axcolours.append(
                    self.colour_select_box.children[1].children[trigger].value
                )
                self.axcolours.append(
                    self.colour_select_box.children[2].children[trigger].value
                )
                self.axcolours.append(
                    self.colour_select_box.children[3].children[trigger].value
                )

                self.LED_names = self.plot_colour_dataframe.loc[self.selected_stimulus][
                    "Description"
                ]
            except:

                break


def plot_qc_locations(spikes):
    qc_locations_plt = px.scatter(
        spikes,
        x="Centres x",
        y="Centres y",
        color=spikes["Total qc"],
        hover_data=[
            "Cell index",
            "Max isi",
            "Std sync",
            "Max psth",
            "stimulus spikes",
            "Total qc",
        ],
    )
    qc_locations_plt.update_traces(
        marker=dict(size=spikes["Area"]), selector=dict(mode="markers")
    )

    qc_locations_plt.update_layout(
        height=1000,
        width=1000,
        title_text="Cell locations quality index and nr of spikes per cluster",
        coloraxis_colorbar=dict(title="Total QI"),
    )

    return qc_locations_plt


def plot_heatmap(QC_df, stimulus_info, Colours, stimulus_trace=False):

    histogram_column = QC_df.loc[:, "Histogram"]
    histograms = histogram_column.values
    histogram_arr = np.zeros((len(QC_df), np.shape(histograms[0])[0]))
    bins_column = QC_df.loc[:, "Bins"]
    bins = bins_column.values
    bins = bins[0]
    nr_cells = np.shape(histograms)[0]
    cell_indices = np.linspace(0, nr_cells - 1, nr_cells)
    for cell in range(nr_cells):
        histogram_arr[cell, :] = histograms[cell] / np.max(histograms[cell])

    histogram_fig = plotly.subplots.make_subplots(rows=2, cols=1, row_width=[0.2, 0.9])
    histogram_fig.add_trace(go.Heatmap(x=bins, y=cell_indices, z=histogram_arr))
    histogram_fig.update_traces(showscale=False)
    nr_stim = int(stimulus_info["Stimulus_repeat_logic"]) * int(
        stimulus_info["Stimulus_repeat_sublogic"]
    )
    time_end = np.max(bins)
    trigger_dur = time_end / nr_stim

    if stimulus_trace == False:
        for i in range(nr_stim):
            # print(i)
            histogram_fig.add_trace(
                go.Scatter(
                    x=[
                        trigger_dur * i,
                        trigger_dur * i,
                        trigger_dur * (i + 1),
                        trigger_dur * (i + 1),
                        trigger_dur * i,
                    ],
                    y=[0, 1, 1, 0, 0],
                    fill="toself",
                    fillcolor=Colours.axcolours[i],
                    name=Colours.LED_names[i],
                    line=dict(width=0),
                    mode="lines",
                ),
                row=2,
                col=1,
            )

    else:
        histogram_fig.add_trace(
            go.Scatter(x=stimulus_trace[:, 0], y=stimulus_trace[:, 1], mode="lines"),
            row=2,
            col=1,
        )

    return histogram_fig.show()


def plot_heatmap_new(QC_df, stimulus_info, Colours, stimulus_trace=False):

    histogram_column = QC_df.loc[:, "PSTH"]
    histograms = histogram_column.values
    histogram_arr = np.zeros((len(QC_df), np.shape(histograms[0])[0]))
    bins_column = QC_df.loc[:, "PSTH_x"]
    bins = bins_column.values
    bins = bins[0]
    nr_cells = np.shape(histograms)[0]
    cell_indices = np.linspace(0, nr_cells - 1, nr_cells)
    for cell in range(nr_cells):
        histogram_arr[cell, :] = histograms[cell] / np.max(histograms[cell])

    histogram_fig = plotly.subplots.make_subplots(
        rows=3,
        cols=1,
        row_width=[0.05, 0.8, 0.2],
        vertical_spacing=0.01,
        shared_xaxes=True,
    )
    histogram_fig.add_trace(
        go.Scatter(
            x=bins,
            y=np.mean(histogram_arr, axis=0),
            mode="lines",
            name="Average PSTH",
            line=dict(color="#000000"),
            fill="tozeroy",
        ),
        row=1,
        col=1,
    )

    histogram_fig.add_trace(
        go.Heatmap(
            x=bins,
            y=cell_indices,
            z=histogram_arr,
            colorscale=[
                [0, "rgb(250, 250, 250)"],  # 0
                [0.2, "rgb(200, 200, 200)"],  # 10
                [0.4, "rgb(150, 150, 150)"],  # 100
                [0.6, "rgb(100, 100, 100)"],  # 1000
                [0.8, "rgb(50, 50, 50)"],  # 10000
                [1.0, "rgb(0, 0, 0)"],  # 100000
            ],
        ),
        row=2,
        col=1,
    )
    histogram_fig.update_traces(showscale=False, selector=dict(type="heatmap"))
    nr_stim = int(stimulus_info["Stimulus_repeat_logic"]) * int(
        stimulus_info["Stimulus_repeat_sublogic"]
    )
    time_end = np.max(bins)
    trigger_dur = time_end / nr_stim

    if stimulus_trace == False:
        for i in range(nr_stim):
            # print(i)
            histogram_fig.add_trace(
                go.Scatter(
                    x=[
                        trigger_dur * i,
                        trigger_dur * i,
                        trigger_dur * (i + 1),
                        trigger_dur * (i + 1),
                        trigger_dur * i,
                    ],
                    y=[0, 1, 1, 0, 0],
                    fill="toself",
                    fillcolor=Colours.axcolours[i],
                    name=Colours.LED_names[i],
                    line=dict(width=0),
                    mode="lines",
                ),
                row=3,
                col=1,
            )

    else:
        histogram_fig.add_trace(
            go.Scatter(x=stimulus_trace[:, 0], y=stimulus_trace[:, 1], mode="lines"),
            row=3,
            col=1,
        )

    histogram_fig.update_xaxes(showticklabels=False, row=1, col=1)
    histogram_fig.update_yaxes(showticklabels=False, row=3, col=1)
    histogram_fig.update_yaxes(showticklabels=False, row=1, col=1)
    histogram_fig.update_xaxes(showticklabels=False, row=2, col=1)
    histogram_fig.update_yaxes(showticklabels=False, row=2, col=1)
    histogram_fig.update_xaxes(title_text="Time in Seconds", row=3, col=1)

    histogram_fig.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)"}, showlegend=False)

    return histogram_fig.show()


def plot_raster_whole_stimulus(
    spiketrains, repeat_logic, sublogic, axcolors, LED_names
):

    """
    Plots the stimulus for a single cells. Plots spiketimes, ISI, SPIKE, SYNC, PSTH
    and stimulus trace
    """
    # print(spiketrains[0].spikes)
    whole_stim_raster = plotly.subplots.make_subplots(
        rows=6,
        cols=1,
        subplot_titles=(
            "Spiketrains",
            "Isi Distance",
            "Spike Profile",
            "Spike Synchronization",
            "PSTH",
        ),
        shared_xaxes=True,
    )

    stim_dur = spiketrains[0].t_end
    duration_trigger = stim_dur / repeat_logic
    trigger_dur = duration_trigger / sublogic
    nr_stim = repeat_logic * sublogic
    trigger_dur = trigger_dur

    for repeat in range(len(spiketrains)):

        spikes_temp = spiketrains[repeat].spikes
        # spikes_temp = spikes['Spiketimestamps'].loc[cell][:spikes['Nr of Spikes'].loc[cell]]

        nr_spikes = np.shape(spikes_temp)[0]
        yvalue = np.ones(nr_spikes) * repeat
        whole_stim_raster.add_trace(
            go.Scattergl(
                mode="markers",
                x=spikes_temp,
                y=yvalue,
                name=None,
                marker=dict(color="Black", size=2),
            )
        )

    avrg_isi_profile = spk.isi_profile(spiketrains)
    spike_profile = spk.spike_profile(spiketrains)
    spike_sync_profile = spk.spike_sync_profile(spiketrains)
    psth = spk.psth(spiketrains, bin_size=0.05)

    # PLot ISI distance average
    x, y = avrg_isi_profile.get_plottable_data()

    whole_stim_raster.add_trace(
        go.Scatter(x=x, y=y, name="Average Isi distance", line=dict(color="Black")),
        row=2,
        col=1,
    )

    x1, y1 = spike_profile.get_plottable_data()

    whole_stim_raster.add_trace(
        go.Scatter(x=x1, y=y1, name="Spike Profile", line=dict(color="Black")),
        row=3,
        col=1,
    )

    x2, y2 = spike_sync_profile.get_plottable_data()

    whole_stim_raster.add_trace(
        go.Scatter(x=x2, y=y2, name="Spike synchronization", line=dict(color="Black")),
        row=4,
        col=1,
    )

    x3, y3 = psth.get_plottable_data()

    whole_stim_raster.add_trace(
        go.Scatter(x=x3, y=y3, name="PSTH", line=dict(color="Black")), row=5, col=1
    )

    # Plot stimulus trace

    for i in range(nr_stim):
        # print(i)
        whole_stim_raster.add_trace(
            go.Scatter(
                x=[
                    trigger_dur * i,
                    trigger_dur * i,
                    trigger_dur * (i + 1),
                    trigger_dur * (i + 1),
                    trigger_dur * i,
                ],
                y=[0, 1, 1, 0, 0],
                fill="toself",
                fillcolor=axcolors[i],
                name=LED_names[i],
                line=dict(width=0),
                mode="lines",
            ),
            row=6,
            col=1,
        )

    # Set title labels
    whole_stim_raster.update_xaxes(title_text="Time in Seconds", row=6, col=1)
    whole_stim_raster.update_yaxes(title_text="Repeat", row=1, col=1)
    whole_stim_raster.update_yaxes(title_text="ISI", row=2, col=1, range=[0, 1])
    whole_stim_raster.update_yaxes(title_text="Spike", row=3, col=1, range=[0, 1])
    whole_stim_raster.update_yaxes(title_text="Spike Sync", row=4, col=1, range=[0, 1])
    whole_stim_raster.update_yaxes(title_text="Spike rate", row=5, col=1)

    # raster_axis[0].set_title('Spiketrains in Recording')

    return_data = {}
    return_data["isi"] = y
    return_data["sync"] = y2
    return_data["psth"] = y3
    return whole_stim_raster


def plot_raster_whole_stimulus_new(
    cell_df, spiketrains, repeat_logic, sublogic, axcolors, LED_names
):

    """
    Plots the stimulus for a single cells. Plots spiketimes, ISI, SPIKE, SYNC, PSTH
    and stimulus trace
    """
    # print(spiketrains[0].spikes)
    whole_stim_raster = plotly.subplots.make_subplots(
        rows=6,
        cols=1,
        subplot_titles=(
            "Spiketrains",
            "Isi Distance",
            "Spike Profile",
            "Spike Synchronization",
            "PSTH",
        ),
        shared_xaxes=True,
    )

    stim_dur = spiketrains[0].t_end
    duration_trigger = stim_dur / repeat_logic
    trigger_dur = duration_trigger / sublogic
    nr_stim = repeat_logic * sublogic
    trigger_dur = trigger_dur

    for repeat in range(len(spiketrains)):

        spikes_temp = spiketrains[repeat].spikes
        # spikes_temp = spikes['Spiketimestamps'].loc[cell][:spikes['Nr of Spikes'].loc[cell]]

        nr_spikes = np.shape(spikes_temp)[0]
        yvalue = np.ones(nr_spikes) * repeat
        whole_stim_raster.add_trace(
            go.Scattergl(
                mode="markers",
                x=spikes_temp,
                y=yvalue,
                name=None,
                marker=dict(color="Black", size=2),
            )
        )

    whole_stim_raster.add_trace(
        go.Scatter(
            x=cell_df["ISI_x"].to_numpy()[0],
            y=cell_df["ISI"].to_numpy()[0],
            name="Average Isi distance",
            line=dict(color="Black"),
        ),
        row=2,
        col=1,
    )

    whole_stim_raster.add_trace(
        go.Scatter(
            x=cell_df["SYNC_x"].to_numpy()[0],
            y=cell_df["SYNC"].to_numpy()[0],
            name="Spike synchronization",
            line=dict(color="Black"),
        ),
        row=3,
        col=1,
    )

    whole_stim_raster.add_trace(
        go.Scatter(
            x=cell_df["PSTH_x"].to_numpy()[0],
            y=cell_df["PSTH"].to_numpy()[0],
            name="PSTH",
            line=dict(color="Black"),
        ),
        row=4,
        col=1,
    )

    whole_stim_raster.add_trace(
        go.Scatter(
            x=cell_df["Gauss_x"].to_numpy()[0],
            y=cell_df["Gauss_average"].to_numpy()[0],
            name="Gaussian spike average",
            line=dict(color="Black"),
        ),
        row=5,
        col=1,
    )

    # Plot stimulus trace

    for i in range(nr_stim):
        # print(i)
        whole_stim_raster.add_trace(
            go.Scatter(
                x=[
                    trigger_dur * i,
                    trigger_dur * i,
                    trigger_dur * (i + 1),
                    trigger_dur * (i + 1),
                    trigger_dur * i,
                ],
                y=[0, 1, 1, 0, 0],
                fill="toself",
                fillcolor=axcolors[i],
                name=LED_names[i],
                line=dict(width=0),
                mode="lines",
            ),
            row=6,
            col=1,
        )

    # Set title labels

    whole_stim_raster.update_yaxes(title_text="Repeat", row=1, col=1)
    whole_stim_raster.update_yaxes(title_text="ISI", row=2, col=1, range=[0, 1])
    whole_stim_raster.update_yaxes(title_text="Spike Sync", row=3, col=1, range=[0, 1])
    whole_stim_raster.update_yaxes(
        title_text="Gaussian probability", row=4, col=1, range=[0, 1]
    )
    whole_stim_raster.update_xaxes(title_text="Time in Seconds", row=6, col=1)
    whole_stim_raster.update_yaxes(title_text="Spike rate", row=6, col=1)

    whole_stim_raster.update_layout(autosize=False, width=1500, height=750)
    whole_stim_raster["layout"]["yaxis4"].update(
        range=[0, np.max(cell_df["PSTH"].to_numpy()[0])]
    )

    # raster_axis[0].set_title('Spiketrains in Recording')

    return whole_stim_raster


def plot_FFF_tuning_curves(
    spikes_df, steps=12, only_on=False, only_off=False, clusnr=8
):

    # Condition for running the function:
    if not "PSTH" in spikes_df:
        print("PSTH has not been calculated, calculate first")
        return

    # Data orgaization:
    PSTH_data = spikes_df["PSTH"].to_numpy()
    PSTH_x = spikes_df["PSTH_x"].to_numpy()[0]
    # We only need the x for the first cell, as all the same
    nr_bins = int(np.shape(PSTH_x)[0] - 2)

    nr_bins_epoch = int(nr_bins / steps)
    bin_size = PSTH_x[1]
    borders_left = np.arange(0, nr_bins, nr_bins_epoch)
    borders_right = np.add(borders_left, int(nr_bins_epoch / 2))

    borders_control_r = np.add(borders_right, int(nr_bins_epoch / 2))

    nr_cells = np.shape(PSTH_data)[0]
    traces_max = np.zeros((nr_cells, steps))

    # Loop over all cells and calculate the max value of the response
    for cell in range(nr_cells):
        trace = np.divide(PSTH_data[cell], np.max(PSTH_data[cell]))
        for border_l, border_r, borders_c, idx in zip(
            borders_left, borders_right, borders_control_r, range(steps)
        ):
            traces_max[cell, idx] = np.max(trace[border_l:border_r])
            traces_max[cell, idx] = np.subtract(
                traces_max[cell, idx], np.max(trace[border_r:borders_c])
            )

    kmeans = KMeans(n_clusters=clusnr, random_state=0).fit(traces_max)
    nr_clusters = int(np.max(np.unique(kmeans.labels_)))

    cluster_names = []
    for cluster_nr in range(nr_clusters + 1):
        cluster_names.append(
            "Cluster nr: "
            + str(cluster_nr)
            + ", n= "
            + str(np.sum(kmeans.labels_ == cluster_nr))
        )
    # Do the plotting
    # Initialize colour template
    ctemplate = ct.Colour_template()

    wavelength = ctemplate.get_stimulus_wavelengths("FFF_6", only_off=True)

    opsins = np.array([570, 508, 455, 419])

    FFF_tuning_overview = plotly.subplots.make_subplots(
        rows=nr_clusters + 1, cols=1, subplot_titles=(cluster_names), shared_xaxes=True
    )

    if only_on:
        traces_max = traces_max[:, ::2]

    if only_off:
        traces_max = traces_max[:, 1::2]

    colours = ctemplate.get_stimulus_colors("FFF_6", only_on=True)
    ticktext = ctemplate.get_stimulus_names("FFF_6", only_off=only_off, only_on=only_on)

    for i in np.unique(kmeans.labels_):
        nr_traces = np.sum(kmeans.labels_ == i)
        cluster_data = traces_max[kmeans.labels_ == i, :]
        if np.max(np.unique(kmeans.labels_) != 0):
            for trace in range(nr_traces):
                FFF_tuning_overview.add_trace(
                    go.Scattergl(
                        x=wavelength,
                        y=cluster_data[trace, :],
                        line=dict(color="Gray", width=0.2),
                        mode="lines",
                    ),
                    row=i + 1,
                    col=1,
                )
        # Mean trace:
        FFF_tuning_overview.add_trace(
            go.Scattergl(
                x=wavelength,
                y=np.mean(cluster_data, axis=0),
                line=dict(color="Black", width=2),
            ),
            row=i + 1,
            col=1,
        )

        # Add opsin spectra for chicken
        opsins_val = gvdski.govardovskii(opsins, 4)

        for opsin in range(np.shape(opsins)[0]):
            print(i)
            FFF_tuning_overview.add_trace(
                go.Scattergl(
                    x=np.asarray(list(range(300, 700, 1))),
                    y=opsins_val[:, opsin],
                    line=dict(
                        color=f"rgba{(*hex_to_rgb(colours[opsin]), 0.2)}", width=2
                    ),
                    fill="tozeroy",
                    fillcolor=f"rgba{(*hex_to_rgb(colours[opsin]), 0.2)}",
                ),
                row=i + 1,
                col=1,
            )
    if clusnr == 1:
        height = 1000
    else:
        height = 5000

    FFF_tuning_overview.update_layout(
        xaxis8=dict(
            tickmode="array",
            tickvals=wavelength,
            ticktext=ticktext,
            showticklabels=True,
        ),
        showlegend=False,
        height=height,
        width=1500,
    )

    return FFF_tuning_overview


def plot_contrast_tuning_curves(
    spikes_df, steps=20, only_on=False, only_off=False, clusnr=8
):
    # Data orgaization:
    PSTH_data = spikes_df["PSTH"].to_numpy()
    PSTH_x = spikes_df["PSTH_x"].to_numpy()[0]
    # We only need the x for the first cell, as all the same
    nr_bins = int(np.shape(PSTH_x)[0] - 2)

    nr_bins_epoch = int(nr_bins / steps)
    bin_size = PSTH_x[1]
    borders_left = np.arange(0, nr_bins, nr_bins_epoch)
    borders_right = np.add(borders_left, int(nr_bins_epoch / 2))

    borders_control_r = np.add(borders_right, int(nr_bins_epoch / 2))

    nr_cells = np.shape(PSTH_data)[0]
    traces_max = np.zeros((nr_cells, steps))
    for cell in range(nr_cells):
        trace = np.divide(PSTH_data[cell], np.max(PSTH_data[cell]))
        for border_l, border_r, borders_c, idx in zip(
            borders_left, borders_right, borders_control_r, range(steps)
        ):
            traces_max[cell, idx] = np.max(trace[border_l:border_r])
            # traces_max[cell, idx] = np.subtract(
            #    traces_max[cell, idx], np.max(trace[border_r:borders_c])
            # )

    kmeans = KMeans(n_clusters=clusnr, random_state=0).fit(traces_max)
    nr_clusters = int(np.max(np.unique(kmeans.labels_)))

    cluster_names = []
    for cluster_nr in range(nr_clusters + 1):
        cluster_names.append(
            "Cluster nr: "
            + str(cluster_nr)
            + ", n= "
            + str(np.sum(kmeans.labels_ == cluster_nr))
        )

    ctemplate = ct.Colour_template()
    colours = ctemplate.get_stimulus_colors("Contrast_Step", only_on=True)
    contrasts = ctemplate.get_stimulus_wavelengths("Contrast_Step", only_on=True)
    contrast_tuning_overview = plotly.subplots.make_subplots(
        rows=nr_clusters + 1, cols=1, subplot_titles=(cluster_names), shared_xaxes=True
    )

    if only_on:
        traces_max = traces_max[:, ::2]

    if only_off:
        traces_max = traces_max[:, 1::2]

    ticktext = ctemplate.get_stimulus_names(
        "Contrast_Step", only_on=only_on, only_off=only_off
    )

    for i in np.unique(kmeans.labels_):
        nr_traces = np.sum(kmeans.labels_ == i)
        cluster_data = traces_max[kmeans.labels_ == i, :]
        if np.max(np.unique(kmeans.labels_) != 0):
            for trace in range(nr_traces):
                contrast_tuning_overview.add_trace(
                    go.Scattergl(
                        x=contrasts,
                        y=cluster_data[trace, :],
                        line=dict(color="Gray", width=0.2),
                        mode="lines",
                    ),
                    row=i + 1,
                    col=1,
                )

    contrast_tuning_overview.add_trace(
        go.Scattergl(
            x=contrasts,
            y=np.mean(cluster_data, axis=0),
            line=dict(color="Black", width=2),
        ),
        row=i + 1,
        col=1,
    )
    if clusnr == 1:
        height = 1500
    else:
        height = 5000

    contrast_tuning_overview.update_layout(
        xaxis8=dict(
            tickmode="array", tickvals=contrasts, ticktext=ticktext, showticklabels=True
        ),
        showlegend=False,
        height=height,
        width=1500,
    )

    return contrast_tuning_overview


def plot_silentsub_tuning_curves(
    spikes_df, steps=18, only_on=False, only_off=False, clusnr=8
):
    silent_sub_bg = np.array([0, 2, 4, 6, 8, 9, 11, 13, 15, 17])
    silent_sub_cone = np.array([1, 3, 5, 7, 10, 12, 14, 16])
    # Data orgaization:
    PSTH_data = spikes_df["PSTH"].to_numpy()
    PSTH_x = spikes_df["PSTH_x"].to_numpy()[0]
    # We only need the x for the first cell, as all the same
    nr_bins = int(np.shape(PSTH_x)[0] - 2)

    nr_bins_epoch = int(nr_bins / steps)
    bin_size = PSTH_x[1]
    borders_left = np.arange(0, nr_bins, nr_bins_epoch)
    borders_right = np.add(borders_left, int(nr_bins_epoch / 2))

    borders_control_r = np.add(borders_right, int(nr_bins_epoch / 2))

    nr_cells = np.shape(PSTH_data)[0]
    traces_max = np.zeros((nr_cells, steps))
    for cell in range(nr_cells):
        trace = np.divide(PSTH_data[cell], np.max(PSTH_data[cell]))
        for border_l, border_r, borders_c, idx in zip(
            borders_left, borders_right, borders_control_r, range(steps)
        ):
            traces_max[cell, idx] = np.max(trace[border_l:border_r])
            traces_max[cell, idx] = np.subtract(
                traces_max[cell, idx], np.max(trace[border_r:borders_c])
            )

    kmeans = KMeans(n_clusters=clusnr, random_state=0).fit(traces_max)
    nr_clusters = int(np.max(np.unique(kmeans.labels_)))

    cluster_names = []
    for cluster_nr in range(nr_clusters + 1):
        cluster_names.append(
            "Cluster nr: "
            + str(cluster_nr)
            + ", n= "
            + str(np.sum(kmeans.labels_ == cluster_nr))
        )

    ctemplate = ct.Colour_template()

    contrast_tuning_overview = plotly.subplots.make_subplots(
        rows=nr_clusters + 1, cols=1, subplot_titles=(cluster_names), shared_xaxes=True
    )

    if only_on:
        xvalues = np.arange(0, 8, 1)
        traces_max = traces_max[:, silent_sub_cone]

    if only_off:
        xvalues = np.arange(0, 10, 1)
        traces_max = traces_max[:, silent_sub_bg]

    ticktext = ctemplate.get_stimulus_names(
        "Silent_Substitution", only_on=only_on, only_off=only_off
    )

    for i in np.unique(kmeans.labels_):
        nr_traces = np.sum(kmeans.labels_ == i)
        cluster_data = traces_max[kmeans.labels_ == i, :]
        if np.max(np.unique(kmeans.labels_) != 0):
            for trace in range(nr_traces):
                contrast_tuning_overview.add_trace(
                    go.Scattergl(
                        x=xvalues,
                        y=cluster_data[trace, :],
                        line=dict(color="Gray", width=0.2),
                        mode="lines",
                    ),
                    row=i + 1,
                    col=1,
                )

        contrast_tuning_overview.add_trace(
            go.Scattergl(
                x=xvalues,
                y=np.mean(cluster_data, axis=0),
                line=dict(color="Black", width=2),
            ),
            row=i + 1,
            col=1,
        )
    if clusnr == 1:
        height = 1500
    else:
        height = 800 * clusnr

    for ax in contrast_tuning_overview["layout"]:
        if ax[:5] == "xaxis":
            contrast_tuning_overview["layout"][ax]["tickmode"] = "array"
            contrast_tuning_overview["layout"][ax]["tickvals"] = xvalues
            contrast_tuning_overview["layout"][ax]["ticktext"] = ticktext
            contrast_tuning_overview["layout"][ax]["showticklabels"] = True

    contrast_tuning_overview.update_layout(showlegend=False, height=height, width=1500)

    return contrast_tuning_overview


def hex_to_rgb(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = hex_color * 2
    return int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
