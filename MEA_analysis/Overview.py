# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 11:23:06 2021

@author: Marvin
"""
import pandas as pd
import numpy as np
import h5py
import qgrid
from MEA_analysis import plot_overview, spike_plotly
from importlib import reload
from MEA_analysis import stimulus_and_spikes as sp
from scipy import signal, io
import matplotlib.pyplot as plt
import math
import pyspike as spk
import multiprocessing as mp
from functools import partial
import traceback
from memory_profiler import profile
import pickle
import plotly
from ipywidgets import widgets
import plotly.graph_objects as go

reload(sp)
from sklearn.preprocessing import normalize, MinMaxScaler
import plotly.express as px
import os
from random import randint


class Dataframe:
    def __init__(self, path=None, name=None):
        self.recording_name = []
        self.nr_recordings = 1
        self.path = []
        self.recording_name.append(name)
        try:
            self.spikes_df = pd.read_pickle(
                path + "\\spikes_for_overview", compression="zip"
            )
        except:
            self.spikes_df = pd.read_csv(path + "\\spikes_for_overview")

        try:
            self.stimulus_df = pd.read_pickle(
                path + "\\stimulus_overview", compression="zip"
            )
        except:
            self.stimulus_df = pd.read_csv(path + "\\stimulus_overview")

        name = np.repeat(name, len(self.spikes_df))
        self.spikes_df.loc[:, "Recording"] = name.tolist()
        self.spikes_df.loc[:, "Idx"] = list(range(0, len(self.spikes_df)))
        nr_stimuli = len(self.spikes_df.index.unique(1))
        self.stimulus_df.loc[:, "Recording"] = name[:nr_stimuli].tolist()
        self.spikes_df.reset_index(inplace=True)
        self.spikes_df.set_index(
            ["Cell index", "Stimulus ID", "Recording"], inplace=True
        )
        # self.spikes_df.set_index("Recording", append=True, inplace=True)
        self.stimulus_df.set_index("Recording", append=True, inplace=True)
        self.path.append(path)
        self.nr_stimuli = len(
            self.stimulus_df.loc[
                self.stimulus_df.index.get_level_values("Recording") == name[0]
            ]
        )

    def add_recording(self, path=None, name=None):
        self.recording_name.append(name)
        try:
            spikes_df = pd.read_pickle(
                path + "\\spikes_for_overview", compression="zip"
            )
        except:
            spikes_df = pd.read_csv(path + "\\spikes_for_overview")

        try:
            stimulus_df = pd.read_pickle(
                path + "\\stimulus_overview", compression="zip"
            )
        except:
            stimulus_df = pd.read_csv(path + "\\stimulus_overview")

        name = np.repeat(name, len(spikes_df))
        spikes_df.loc[:, "Recording"] = name.tolist()
        nr_stimuli = len(spikes_df.index.unique(1))
        stimulus_df.loc[:, "Recording"] = name[:nr_stimuli].tolist()

        stimulus_df.set_index("Recording", append=True, inplace=True)

        spikes_df = spikes_df.reset_index()
        spikes_df = spikes_df.set_index(["Cell index", "Stimulus ID", "Recording"])
        self.spikes_df.drop(columns="Idx", inplace=True)
        self.spikes_df = pd.concat([self.spikes_df, spikes_df])
        self.stimulus_df = pd.concat([self.stimulus_df, stimulus_df])
        self.spikes_df.loc[:, "Idx"] = list(range(0, len(self.spikes_df)))
        self.nr_recordings = self.nr_recordings + 1
        self.path.append(path)

        print("Dataframe added to overview dataframe")

    def show_df(self, name="spikes", level=False, condition=False):
        if name == "spikes":
            if level and not condition:
                try:
                    view = qgrid.show_grid(self.spikes_df[level])
                except KeyError:
                    view = qgrid.show_grid(
                        self.spikes_df.index.show_level_values(level)
                    )

            elif level and condition:
                try:
                    view = qgrid.show_grid(
                        self.spikes_df[self.spikes_df[level] == condition]
                    )
                except KeyError:
                    view = qgrid.show_grid(
                        self.spikes_df[
                            self.spikes_df.index.show_level_values(level) == condition
                        ]
                    )
            else:
                view = qgrid.show_grid(self.spikes_df)

        elif name == "stimulus":
            view = qgrid.show_grid(self.stimulus_df)
        else:
            print("Unknow name for dataframe, showing default")
            view = qgrid.show_grid(self.spikes_df)
        self.df_view = view
        return view

    def use_view_as_filter(self, filter_name=False):
        if not filter_name:
            filter_name = "Filter"
            try:
                self.spikes_df.drop(columns="Filter")
            except:
                pass

        sub_df = self.df_view.get_changed_df()
        sub_df[filter_name] = "Yes"
        self.drop_and_fill(sub_df, index_for_drop=sub_df.index)
        self.spikes_df["Filter"] = self.spikes_df["Filter"].fillna("No")
        self.spikes_df = self.spikes_df.astype({"Filter": str})

    def delete_trigger_from_stimulus(self, stimulus, nr_trigger):
        print("Warning, this will remove " + str(nr_trigger) + " from all recordings")
        spikes_df, stimulus_df = self.get_stimulus_subset(name=stimulus)

        extract_trigger = stimulus_df["Trigger_Fr_relative"].to_numpy()
        for array, idx in zip(extract_trigger, range(np.shape(extract_trigger)[0])):
            extract_trigger[idx] = array[:-nr_trigger]
        stimulus_df["Trigger_Fr_relative"] = extract_trigger

        extract_trigger = stimulus_df["Trigger_int"].to_numpy()
        for array, idx in zip(extract_trigger, range(np.shape(extract_trigger)[0])):
            extract_trigger[idx] = array[:-nr_trigger]

        stimulus_df["Trigger_int"] = extract_trigger

        index_for_drop = stimulus_df.index

        print(index_for_drop)

        print(stimulus_df)

        self.stimulus_df.drop(index_for_drop, inplace=True)
        self.stimulus_df = pd.concat([self.stimulus_df, stimulus_df])
        self.stimulus_df = self.stimulus_df.sort_index(level="Recording")

    def add_cluster_ids(self, recording, cluster_ids):
        if "Cluster ID" in self.spikes_df:
            self.spikes_df.loc[
                self.spikes_df.index.get_level_values("Recording") == recording,
                "Cluster ID",
            ] = cluster_ids
        else:
            self.spikes_df.loc[:, "Cluster ID"] = [0] * len(self.spikes_df)
            self.spikes_df.loc[
                self.spikes_df.index.get_level_values("Recording") == recording,
                "Cluster ID",
            ] = cluster_ids

    def load_cluster_ids(self, recording_name=None):
        if not recording_name:
            for recording in range(self.nr_recordings):
                print(recording)
                path = self.path[recording] + "\\cluster_ids.mat"
                with h5py.File(path, "r") as hf:
                    data = hf["clusters_2nd"][:]

                nr_stimuli = len(
                    self.stimulus_df.loc[
                        self.stimulus_df.index.get_level_values("Recording")
                        == self.recording_name[recording]
                    ]
                )
                clusters = data[0]
                clusters = np.repeat(clusters, nr_stimuli).astype(int)
                self.add_cluster_ids(self.recording_name[recording], clusters)

    def load_kernels(self, recording_name=None, cluster_only=True):
        if not recording_name:
            for recording in range(self.nr_recordings):

                temp_df = self.spikes_df.loc[
                    self.spikes_df.index.get_level_values("Recording")
                    == self.recording_name[recording]
                ]

                temp_df = temp_df.loc[
                    temp_df.index.get_level_values("Stimulus ID") == 0
                ]
                temp_df.reset_index(inplace=True)
                temp_df["Stimulus ID"].replace(0, value=3, inplace=True)
                temp_df["Stimulus name"].replace(
                    "FFF", value="CNoise_FFF", inplace=True
                )
                temp_df = temp_df.drop("Spikes", 1)
                temp_df = temp_df.drop("Idx", 1)
                temp_df = temp_df.drop("Nr of Spikes", 1)
                temp_df = temp_df.drop("Area", 1)
                temp_df

                cells = temp_df.index.get_level_values(0).to_numpy()
                all_kernels = np.empty(np.shape(cells)[0], dtype=object)
                for cell, idx in zip(cells, range(len(temp_df))):
                    # print(temp_df.loc[cell]["Cluster ID"])
                    if temp_df.loc[cell]["Cluster ID"].any():
                        continue
                    else:
                        try:

                            path = (
                                self.path[recording]
                                + "\\Kernel\\Kernel_Cell_"
                                + str(cell)
                                + ".mat"
                            )
                            with h5py.File(path, "r") as hf:
                                all_kernels[idx] = hf["Kernels"][:]
                        except:
                            traceback.print_exc()
                            all_kernels[idx] = 0

        return all_kernels

    def convolve_gauss_all_clusters(
        self,
        by_clusters=True,
        by_stimulus=False,
        clusnr=None,
        stimname=None,
        binsize=0.05,
    ):
        if by_stimulus:
            by_clusters = False

        if by_clusters:
            if not clusnr:
                for cluster in self.spikes_df["Cluster ID"].unique()[1:]:
                    temp_df = parallel_gauss_conv(
                        self.stimulus_df, self.get_cluster_subset(cluster)
                    )
                    self.drop_and_fill(temp_df, column="Cluster ID", condition=cluster)
            else:
                temp_df = parallel_gauss_conv(
                    self.stimulus_df, self.get_cluster_subset(clusnr)
                )
                self.drop_and_fill(temp_df, column="Cluster ID", condition=clusnr)
        else:
            if not stimname:
                for stimulus in self.spikes_df["Stimulus name"].unique():
                    temp_df = parallel_gauss_conv(
                        self.stimulus_df,
                        self.get_stimulus_subset(name=stimulus),
                        binsize,
                    )
                    self.drop_and_fill(
                        temp_df, column="Stimulus name", condition=stimulus
                    )
            else:
                temp_df = parallel_gauss_conv(
                    self.stimulus_df, self.get_stimulus_subset(name=stimname)
                )
                self.drop_and_fill(temp_df, column="Stimulus name", condition=stimname)

    def psth_all_cells(
        self,
        by_clusters=True,
        by_stimulus=False,
        clusnr=None,
        stimname=None,
        binsize=0.05,
        filter_only=False,
    ):
        if by_stimulus:
            by_clusters = False

        if by_clusters:
            if not clusnr:
                for cluster in self.spikes_df["Cluster ID"].unique()[1:]:
                    temp_df = parallel_psth(
                        self.stimulus_df, self.get_cluster_subset(cluster), binsize
                    )
                    self.drop_and_fill(temp_df, column="Cluster ID", condition=cluster)
            else:
                temp_df = parallel_psth(
                    self.stimulus_df, self.get_cluster_subset(clusnr), binsize
                )
                self.drop_and_fill(temp_df, column="Cluster ID", condition=clusnr)
        else:
            if not stimname:
                for stimulus in self.spikes_df["Stimulus name"].unique():
                    temp_df = parallel_psth(
                        self.stimulus_df,
                        self.get_stimulus_subset(name=stimulus),
                        binsize,
                    )
                    self.drop_and_fill(
                        temp_df, column="Stimulus name", condition=stimulus
                    )
            else:
                spikes_df, stimulus_df = self.get_stimulus_subset(name=stimname)
                temp_df = parallel_psth(stimulus_df, spikes_df, binsize)
                if filter_only:
                    try:
                        temp_df.loc[temp_df["Filter"] == "Yes"]
                    except KeyError:
                        print("Not filter found, select first.")

                self.drop_and_fill(temp_df, column="Stimulus name", condition=stimname)

    def isi_all_cells(
        self, by_clusters=True, by_stimulus=False, clusnr=None, stimname=None
    ):
        if by_stimulus:
            by_clusters = False

        if by_clusters:
            if not clusnr:
                for cluster in self.spikes_df["Cluster ID"].unique()[1:]:
                    temp_df = spikes_isi_all(
                        self.stimulus_df, self.get_cluster_subset(cluster)
                    )
                    self.drop_and_fill(temp_df, column="Cluster ID", condition=cluster)
            else:
                temp_df = spikes_isi_all(
                    self.stimulus_df, self.get_cluster_subset(clusnr)
                )
                self.drop_and_fill(temp_df, column="Cluster ID", condition=clusnr)
        else:
            if not stimname:
                for stimulus in self.spikes_df["Stimulus name"].unique():
                    temp_df = spikes_isi_all(
                        self.stimulus_df, self.get_stimulus_subset(name=stimulus)
                    )
                    self.drop_and_fill(
                        temp_df, column="Stimulus name", condition=stimulus
                    )
            else:
                temp_df = spikes_isi_all(
                    self.stimulus_df, self.get_stimulus_subset(name=stimname)
                )
                self.drop_and_fill(temp_df, column="Stimulus name", condition=stimname)

    def sync_all_cells(
        self, by_clusters=True, by_stimulus=False, clusnr=None, stimname=None
    ):
        if by_stimulus:
            by_clusters = False

        if by_clusters:
            if not clusnr:
                for cluster in self.spikes_df["Cluster ID"].unique()[1:]:
                    temp_df = spikes_sync_all(
                        self.stimulus_df, self.get_cluster_subset(cluster)
                    )
                    self.drop_and_fill(temp_df, column="Cluster ID", condition=cluster)
            else:
                temp_df = spikes_sync_all(
                    self.stimulus_df, self.get_cluster_subset(clusnr)
                )
                self.drop_and_fill(temp_df, column="Cluster ID", condition=clusnr)
        else:
            if not stimname:
                for stimulus in self.spikes_df["Stimulus name"].unique():
                    spikes_df, stimulus_df = self.get_stimulus_subset(name=stimulus)
                    temp_df = spikes_sync_all(stimulus_df, spikes_df)
                    self.drop_and_fill(
                        temp_df, column="Stimulus name", condition=stimulus
                    )
            else:
                spikes_df, stimulus_df = self.get_stimulus_subset(name=stimname)
                temp_df = spikes_sync_all(stimulus_df, spikes_df)
                self.drop_and_fill(temp_df, column="Stimulus name", condition=stimname)

        # for cluster in self.spikes_df["Cluster ID"].unique()[1:]:
        #     temp_df = spikes_sync_all(self.stimulus_df, self.get_cluster_subset(cluster))
        #     self.drop_and_fill("Cluster ID", cluster, temp_df)
        #     print(cluster)

        # self.spikes_stimulus = spikes_conv_gauss_all(self.spikes_stimulus, trigger_complete,
        #                                       int(self.stimulus_info["Stimulus_repeat_logic"]),
        #                                       self.sampling_freq, resolution, penalty, sigma,
        #                                       plot_gaussian)

    def get_single_recording_spikes(self, recording=None, recording_idx=None):
        if recording is not None:
            return self.spikes_df.loc[
                self.spikes_df.index.get_level_values("Recording") == recording
            ]
        elif recording_idx is not None:
            return self.spikes_df.loc[
                self.spikes_df.index.get_level_values("Recording")
                == self.recording_name[recording_idx]
            ]

    def get_single_recording_stimulus(self, recording=None, recording_idx=None):
        if recording is not None:
            return self.stimulus_df.loc[
                self.stimulus_df.index.get_level_values("Recording") == recording
            ]
        elif recording_idx is not None:
            return self.stimulus_df.loc[
                self.stimulus_df.index.get_level_values("Recording")
                == self.recording_name[recording_idx]
            ]

    def get_cluster_subset(self, cluster_nr=1, return_logical=False):
        if return_logical:
            spikes_df = self.spikes_df["Cluster ID"] == cluster_nr
            return
        else:
            spikes_df = self.spikes_df.loc[
                self.spikes_df["Cluster ID"] == cluster_nr
            ].copy()
            return spikes_df

    def get_stimulus_subset(self, stimulus=0, name=None, return_logical=False):
        if name:
            if return_logical:
                spikes_df = self.spikes_df["Stimulus name"] == name
                stimulus_df = self.stimulus_df["Stimulus_name"] == name

            else:
                spikes_df = self.spikes_df.loc[
                    self.spikes_df["Stimulus name"] == name
                ].copy()

                stimulus_df = self.stimulus_df.loc[
                    self.stimulus_df["Stimulus_name"] == name
                ].copy()
        else:
            if return_logical:
                spikes_df = self.spikes_df.index.get_level_values(1) == stimulus
                stimulus_df = self.stimulus_df.index.get_level_values(0) == stimulus
            else:
                spikes_df = self.spikes_df.loc[
                    self.spikes_df.index.get_level_values(1) == stimulus
                ].copy()

                stimulus_df = self.stimulus_df.loc[
                    self.stimulus_df["Stimulus_name"] == stimulus
                ].copy()

        return spikes_df, stimulus_df

    def drop(self, column, condition):
        self.spikes_df.drop(self.spikes_df[self.spikes_df[column] == condition].index)

    def drop_and_fill(self, add_df, column=None, condition=None, index_for_drop=None):
        if index_for_drop is None:
            try:
                index_for_drop = self.spikes_df[
                    self.spikes_df[column] == condition
                ].index
            except KeyError:
                index_for_drop = self.spikes_df[
                    self.spikes_df.index.get_level_values(column) == condition
                ].index

        self.spikes_df.drop(index_for_drop, inplace=True)
        self.spikes_df = pd.concat([self.spikes_df, add_df])
        self.spikes_df = self.spikes_df.sort_index(level="Recording")

    def get_sorted_cluster_idx(self):
        return self.spikes_df.value_counts(subset="Cluster ID")[1:].index.to_numpy()

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as f:
            return pickle.load(f)

    ## Plot stuff

    def plot_cluster_overview(self, plots_per_figure=10, folder=None):
        nr_clusters = len(self.get_sorted_cluster_idx())
        nr_figures = math.ceil(nr_clusters / plots_per_figure)
        clusters_split = np.array_split(self.get_sorted_cluster_idx(), nr_figures)

        figure_store = np.empty(nr_figures, dtype=object)

        stimulus_order = ["FFF", "Chirp"]
        for figure in range(nr_figures):

            rows_per, cols_per, overview_fig = plot_overview.initialise_subplots(
                len(clusters_split[figure]), self.nr_stimuli
            )

            for stimulus in range(self.nr_stimuli):

                for cluster, idx in zip(
                    clusters_split[figure], range(len(clusters_split[figure]))
                ):

                    df_temp = self.spikes_df.loc[
                        self.get_stimulus_subset(
                            name=stimulus_order[stimulus], return_logical=True
                        )
                        & self.get_cluster_subset(cluster, return_logical=True)
                    ]
                    # print(df_temp)
                    # histogram
                    (
                        heatmap_trace,
                        heatmap_raster,
                        heatmap_trace_std_u,
                        heatmap_trace_std_l,
                    ) = self.heatmap(df_temp)

                    overview_fig.add_trace(
                        heatmap_trace_std_l,
                        row=idx * rows_per + 2,
                        col=stimulus * cols_per + 1,
                    )

                    overview_fig.add_trace(
                        heatmap_trace_std_u,
                        row=idx * rows_per + 2,
                        col=stimulus * cols_per + 1,
                    )
                    overview_fig.add_trace(
                        heatmap_trace,
                        row=idx * rows_per + 2,
                        col=stimulus * cols_per + 1,
                    )

                    overview_fig.add_trace(
                        heatmap_raster,
                        row=idx * rows_per + 3,
                        col=stimulus * cols_per + 1,
                    )

                    tvals = (
                        self.get_cluster_subset(cluster)
                        .index.get_level_values("Recording")
                        .value_counts()
                        .to_numpy()
                        / self.nr_stimuli
                    )

                    # overview_fig.add_hline(y=np.min(tvals), row=idx*rows_per+3,
                    #      col=stimulus*cols_per+1, )

                    if stimulus == 1 & idx == 1:
                        time, y = chirp()
                        overview_fig.add_trace(
                            go.Scattergl(x=time, y=y, line=dict(color="#000000")),
                            row=1,
                            col=2,
                        )

            for cluster, idx in zip(
                clusters_split[figure], range(len(clusters_split[figure]))
            ):
                temp_df = self.spikes_df.loc[
                    self.spikes_df.index.get_level_values("Stimulus ID") == 0
                ]
                temp_df = temp_df.loc[temp_df["Cluster ID"] == cluster]

                kernel_traces = self.plot_kernels(temp_df)

                nr_traces = np.shape(kernel_traces)[1] - 1
                # print(nr_traces)
                for i in range(4):

                    for cell in range(1, nr_traces + 1):
                        try:
                            overview_fig.add_trace(
                                kernel_traces[i, cell], row=idx * rows_per + 2, col=3
                            )
                            cell = cell + 1
                            overview_fig.add_trace(
                                kernel_traces[i, 0], row=idx * rows_per + 3, col=3
                            )
                        except:
                            cell = cell + 1
                            overview_fig.add_trace(
                                kernel_traces[i, 0], row=idx * rows_per + 3, col=3
                            )

            colours = spike_plotly.Colour_template()
            FFF_6_colours = colours.plot_colour_dataframe.loc["FFF_6"]["Colours"]
            # FFF_6_LEDs = colours.plot_colour_dataframe.loc["FFF_6"]["Description"]
            for c in range(12):
                overview_fig.add_vrect(
                    x0=0 + 2 * c,
                    x1=2 + 2 * c,
                    row="all",
                    col=1,
                    fillcolor=FFF_6_colours[c],
                    opacity=0.08,
                    line_width=0,
                )

            overview_fig.update_layout(
                {"plot_bgcolor": "rgba(0, 0, 0, 0)"},
                autosize=False,
                width=1200,
                height=1920,
                showlegend=False,
            )

            overview_fig.update_xaxes(showticklabels=False)
            overview_fig.update_yaxes(showticklabels=False)

            figure_store[figure] = overview_fig

        if folder:
            for figure, idx in zip(figure_store, range(np.shape(figure_store)[0])):
                figure.write_image(folder + "\\Figure_" + str(figure))

        return figure_store

    def plot_cluster_overview_2nd(self, plots_per_figure=10, folder=None):
        nr_clusters = len(self.get_sorted_cluster_idx())
        nr_figures = math.ceil(nr_clusters / plots_per_figure)
        clusters_split = np.array_split(self.get_sorted_cluster_idx(), nr_figures)

        figure_store = np.empty(nr_figures, dtype=object)

        stimulus_order = ["FFF", "Silent Substitution", "Contrast Steps", "Chirp"]
        for figure in range(nr_figures):

            rows_per, cols_per, overview_fig = plot_overview.initialise_subplots(
                len(clusters_split[figure]), self.nr_stimuli
            )

            for stimulus in range(self.nr_stimuli):

                for cluster, idx in zip(
                    clusters_split[figure], range(len(clusters_split[figure]))
                ):

                    df_temp = self.spikes_df.loc[
                        self.get_stimulus_subset(
                            name=stimulus_order[stimulus], return_logical=True
                        )
                        & self.get_cluster_subset(cluster, return_logical=True)
                    ]
                    # print(df_temp)
                    # histogram
                    (
                        heatmap_trace,
                        heatmap_raster,
                        heatmap_trace_std_u,
                        heatmap_trace_std_l,
                    ) = self.heatmap(df_temp)

                    overview_fig.add_trace(
                        heatmap_trace_std_l,
                        row=idx * rows_per + 2,
                        col=stimulus * cols_per + 1,
                    )

                    overview_fig.add_trace(
                        heatmap_trace_std_u,
                        row=idx * rows_per + 2,
                        col=stimulus * cols_per + 1,
                    )
                    overview_fig.add_trace(
                        heatmap_trace,
                        row=idx * rows_per + 2,
                        col=stimulus * cols_per + 1,
                    )

                    overview_fig.add_trace(
                        heatmap_raster,
                        row=idx * rows_per + 3,
                        col=stimulus * cols_per + 1,
                    )

                    tvals = (
                        self.get_cluster_subset(cluster)
                        .index.get_level_values("Recording")
                        .value_counts()
                        .to_numpy()
                        / self.nr_stimuli
                    )

                    # overview_fig.add_hline(y=np.min(tvals), row=idx*rows_per+3,
                    #      col=stimulus*cols_per+1, )

                    if stimulus == 3 & idx == 1:
                        time, y = chirp()
                        overview_fig.add_trace(
                            go.Scattergl(x=time, y=y, line=dict(color="#000000")),
                            row=1,
                            col=4,
                        )

            # for cluster, idx in zip(clusters_split[figure],
            #                             range(len(clusters_split[figure]))):
            #    temp_df = self.spikes_df.loc[
            #        self.spikes_df.index.get_level_values("Stimulus ID") == 0]
            #    temp_df = temp_df.loc[temp_df["Cluster ID"] == cluster]

            #    kernel_traces = self.plot_kernels(temp_df)

            #    nr_traces = np.shape(kernel_traces)[1]-1
            #    #print(nr_traces)
            #    for i in range(4):

            #          for cell in range(1, nr_traces+1):
            #              try:
            #                  overview_fig.add_trace(
            #                      kernel_traces[i, cell], row=idx*rows_per+2,
            #                                     col=3)
            #                  cell = cell+1
            #                  overview_fig.add_trace(
            #                      kernel_traces[i, 0], row=idx*rows_per+3,
            #                                     col=3)
            #              except:
            #                  cell = cell+1
            #                  overview_fig.add_trace(
            #                      kernel_traces[i, 0], row=idx*rows_per+3,
            #                                     col=3)

            colours = spike_plotly.Colour_template()
            FFF_6_colours = colours.plot_colour_dataframe.loc["FFF_6"]["Colours"]
            # FFF_6_LEDs = colours.plot_colour_dataframe.loc["FFF_6"]["Description"]
            for c in range(12):
                overview_fig.add_vrect(
                    x0=0 + 2 * c,
                    x1=2 + 2 * c,
                    row="all",
                    col=1,
                    fillcolor=FFF_6_colours[c],
                    opacity=0.08,
                    line_width=0,
                )

            # Silent Substitution
            Silent_sub_colours = colours.plot_colour_dataframe.loc[
                "Silent_Substitution"
            ]["Colours"]

            for c in range(18):
                overview_fig.add_vrect(
                    x0=0 + 2 * c,
                    x1=2 + 2 * c,
                    row="all",
                    col=2,
                    fillcolor=Silent_sub_colours[c],
                    opacity=0.08,
                    line_width=0,
                )

            Contrast_step_colours = colours.plot_colour_dataframe.loc["Contrast_Step"][
                "Colours"
            ]

            for c in range(20):
                overview_fig.add_vrect(
                    x0=0 + 2 * c,
                    x1=2 + 2 * c,
                    row="all",
                    col=3,
                    fillcolor=Contrast_step_colours[c],
                    opacity=0.08,
                    line_width=0,
                )

            overview_fig.update_layout(
                {"plot_bgcolor": "rgba(0, 0, 0, 0)"},
                autosize=False,
                width=1200,
                height=1920,
                showlegend=False,
            )

            overview_fig.update_xaxes(showticklabels=False)
            overview_fig.update_yaxes(showticklabels=False)

            figure_store[figure] = overview_fig

        if folder:
            for figure, idx in zip(figure_store, range(np.shape(figure_store)[0])):
                figure.write_image(folder + "\\Figure_" + str(figure))

        return figure_store

    def heatmap(spikes_df):

        histogram_column = spikes_df.loc[:, "PSTH"]
        histograms = histogram_column.values
        histogram_arr = np.zeros((len(spikes_df), np.shape(histograms[0])[0]))
        bins_column = spikes_df.loc[:, "PSTH_x"]
        bins = bins_column.values
        bins = bins[0]
        nr_cells = np.shape(histograms)[0]
        cell_indices = np.linspace(0, nr_cells - 1, nr_cells)
        for cell in range(nr_cells):

            histogram_arr[cell, :] = histograms[cell] / np.max(histograms[cell])
        histogram_mean = np.nanmean(histogram_arr, axis=0)
        histogram_std = np.nanstd(histogram_arr, axis=0)
        std_upper = np.add(histogram_mean, histogram_std)
        std_lower = np.subtract(histogram_mean, histogram_std)

        heatmap_trace = go.Scattergl(
            x=bins,
            y=histogram_mean,
            mode="lines",
            name="Average PSTH",
            line=dict(color="#000000"),
        )

        heatmap_trace_std_u = go.Scattergl(
            name="Upper Bound",
            x=bins,
            y=std_upper,
            mode="lines",
            marker=dict(color="#444"),
            line=dict(width=0),
            fill="tonexty",
            showlegend=False,
        )

        heatmap_trace_std_l = go.Scattergl(
            name="lower Bound",
            x=bins,
            y=std_lower,
            marker=dict(color="#444"),
            line=dict(width=0),
            mode="lines",
            fillcolor="rgba(68, 68, 68, 0.6)",
            fill="tonexty",
            showlegend=False,
        )

        heatmap_raster = go.Heatmap(
            x=bins,
            y=cell_indices,
            z=histogram_arr,
            colorscale=[
                [0, "rgb(250, 250, 250)"],
                [0.2, "rgb(200, 200, 200)"],
                [0.4, "rgb(150, 150, 150)"],
                [0.6, "rgb(100, 100, 100)"],
                [0.8, "rgb(50, 50, 50)"],
                [1.0, "rgb(0, 0, 0)"],
            ],
            showscale=False,
        )

        return heatmap_trace, heatmap_raster, heatmap_trace_std_u, heatmap_trace_std_l

    def gauss_mean_traces(self, spikes_df):
        gauss_column = spikes_df.loc[:, "Gauss_average"]
        gauss_averages = gauss_column.values
        gauss_arr = np.zeros((len(spikes_df), np.shape(gauss_averages[0])[0]))
        bins_column = spikes_df.loc[:, "Gauss_x"]
        bins = bins_column.values
        bins = bins[0]
        nr_cells = np.shape(gauss_averages)[0]
        for cell in range(nr_cells):
            # print(cell)
            gauss_arr[cell, :] = gauss_averages[cell] / np.max(gauss_averages[cell])
        gauss_mean = np.nanmean(gauss_arr, axis=0)
        gauss_std = np.nanstd(gauss_arr, axis=0)
        std_upper = np.add(gauss_mean, gauss_std)
        std_lower = np.subtract(gauss_mean, gauss_std)

        gauss_trace = go.Scattergl(
            x=bins,
            y=gauss_mean,
            mode="lines",
            name="Average PSTH",
            line=dict(color="#000000"),
        )

        gauss_trace_std_u = go.Scattergl(
            name="Upper Bound",
            x=bins,
            y=std_upper,
            mode="lines",
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False,
        )

        gauss_trace_std_l = go.Scattergl(
            name="lower Bound",
            x=bins,
            y=std_lower,
            marker=dict(color="#444"),
            line=dict(width=0),
            mode="lines",
            fillcolor="rgba(68, 68, 68, 0.6)",
            fill="tonexty",
            showlegend=False,
        )

        return gauss_trace, gauss_trace_std_u, gauss_trace_std_l

    def plot_kernels(self, spikes_df):
        cell_idx = spikes_df.index.get_level_values(0).to_numpy()

        recording = spikes_df.index.get_level_values(7).to_numpy()
        nr_cells = np.shape(cell_idx)[0]

        kernels_red = np.empty((nr_cells, 500), dtype=int)
        kernels_green = np.empty((nr_cells, 500), dtype=int)
        kernels_blue = np.empty((nr_cells, 500), dtype=int)
        kernels_vs = np.empty((nr_cells, 500), dtype=int)

        for cell, idx in zip(cell_idx, range(nr_cells)):

            try:
                path = (
                    "D:\\"
                    + str(recording[idx])
                    + "\\Kernel\\Kernel_Cell_"
                    + str(cell)
                    + ".mat"
                )
                with h5py.File(path, "r") as hf:
                    data = hf["Kernels"][:]
                kernels_red[idx, :] = data[0]
                kernels_green[idx, :] = data[1]
                kernels_blue[idx, :] = data[2]
                kernels_vs[idx, :] = data[3]

            except:
                traceback.print_exc()
                kernels_red[idx, :] = 0
                kernels_green[idx, :] = 0
                kernels_blue[idx, :] = 0
                kernels_vs[idx, :] = 0

        # red_kernels = np.divide(red_kernels, np.max(red_kernels))
        # green_kernels = np.divide(green_kernels, np.max(green_kernels))
        # blue_kernels = np.divide(blue_kernels, np.max(blue_kernels))
        # vs_kernels = np.divide(vs_kernels, np.max(vs_kernels))
        mean_kernel = np.empty((4, np.shape(kernels_vs)[1]), dtype=int)
        mean_kernel[0, :] = np.mean(kernels_red, 0)
        mean_kernel[1, :] = np.mean(kernels_green, 0)
        mean_kernel[2, :] = np.mean(kernels_blue, 0)
        mean_kernel[3, :] = np.mean(kernels_vs, 0)

        # mean_kernel = normalize(mean_kernel, axis=0)

        mean_kernel = np.divide(mean_kernel, np.mean(mean_kernel))

        # for cell in range(nr_cells):

        #     io.savemat("D:\\Kernels_out\\Kernel_Cell_"+
        #         str(spikes_df.iloc[0]["Cluster ID"])+
        #         str(spikes_df.iloc[0]["Cluster ID"])+
        #         str(cell)+".mat",
        #         {"Kernels": np.transpose(mean_kernel)})

        color_temp = ["#7c86fe", "#7cfcfe", "#8afe7c", "#fe7c7c"]

        traces = np.empty((4, 11), dtype=object)
        for i in range(4):

            traces[i, 0] = go.Scatter(
                x=np.linspace(0.001, 0.5, 400),
                y=mean_kernel[i, :400],
                line=dict(color=color_temp[i], width=2.5),
            )

        for n, idx in zip(
            [randint(0, nr_cells - 1) for p in range(0, 10)], range(1, 11)
        ):

            kernel = np.empty((4, 500), dtype=int)
            kernel[0, :] = kernels_red[n, :]
            kernel[1, :] = kernels_green[n, :]
            kernel[2, :] = kernels_blue[n, :]
            kernel[3, :] = kernels_vs[n, :]

            # kernel = normalize(kernel, axis=0)
            kernel = np.divide(kernel, np.mean(kernel))

            traces[0, idx] = go.Scattergl(
                x=np.linspace(0.001, 0.5, 400),
                y=kernel[0, :],
                line=dict(color=color_temp[0], width=0.8),
            )
            traces[1, idx] = go.Scattergl(
                x=np.linspace(0.001, 0.5, 400),
                y=kernel[1, :],
                line=dict(color=color_temp[1], width=0.8),
            )
            traces[2, idx] = go.Scattergl(
                x=np.linspace(0.001, 0.5, 400),
                y=kernel[2, :],
                line=dict(color=color_temp[2], width=0.8),
            )
            traces[3, idx] = go.Scattergl(
                x=np.linspace(0.001, 0.5, 400),
                y=kernel[3, :],
                line=dict(color=color_temp[3], width=0.8),
            )

        return traces

    # histogram_column = spikes_df.loc[:, 'PSTH']
    #     histograms = histogram_column.values
    #     histogram_arr = np.zeros((len(spikes_df), np.shape(histograms[0])[0]))
    #     bins_column = spikes_df.loc[:, 'PSTH_x']
    #     bins = bins_column.values
    #     bins = bins[0]
    #     nr_cells = np.shape(histograms)[0]
    #     cell_indices = np.linspace(0,nr_cells-1,nr_cells)
    #     for cell in range(nr_cells):

    #         histogram_arr[cell, :] = histograms[cell]/np.max(histograms[cell])
    #     histogram_mean = np.nanmean(histogram_arr, axis=0)
    #     histogram_std = np.nanstd(histogram_arr, axis=0)
    #     std_upper = np.add(histogram_mean, histogram_std)
    #     std_lower = np.subtract(histogram_mean, histogram_std)

    #     heatmap_trace = go.Scattergl(
    #                         x=bins, y=histogram_mean,
    #                         mode='lines', name="Average PSTH",
    #                         line=dict(color="#000000"))

    # return histogram_fig.show()


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
    rows = len(spike_df)

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

    for row in rows:
        try:

            spikes, spiketrains = sp.get_spikes_whole_stimulus_new(
                spike_df, trigger_complete, repeat_logic, sampling_freq, row=row
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
            print(str(row) + " Failed")
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


def spikes_conv_gauss_all_new(
    stimulus_df,
    spikes_df,
    sampling_freq=17852.767845719834,
    resolution=0.001,
    penalty=1,
    sigma=10,
    plot_gaussian=False,
    show_error=True,
    saveall=False,
):

    stim_idx = spikes_df.index.get_level_values(1).to_numpy()

    trigger_complete = stimulus_df["Trigger_Fr_relative"].to_numpy()
    repeat_logic = stimulus_df["Stimulus_repeat_logic"].to_numpy()

    # Define gaussian window:
    window_size = sigma * 3
    gauss_window = signal.windows.gaussian(window_size, std=sigma)
    if plot_gaussian:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(gauss_window)

    results = np.empty((len(spikes_df), 5), object)

    for row, pos, idx in zip(spikes_df.itertuples(), stim_idx, range(len(spikes_df))):
        try:

            spikes, spiketrains = sp.get_spikes_whole_stimulus_new(
                row.Spikes.compressed(),
                trigger_complete[pos],
                int(repeat_logic[pos]),
                sampling_freq,
            )

            # Define bining parameters
            nr_bins = math.ceil(spiketrains[0].t_end / resolution) * 2 - 2
            # nr_bins_simple = math.ceil(spiketrains[0].t_end/resolution-1)

            # Prepare arrays for loop over repeats
            repeats = len(spiketrains)
            gauss_bins = np.zeros(
                (nr_bins, repeats), dtype=np.uint8
            )  # This stores the inital bined spikes
            gauss_x = np.zeros(
                (nr_bins), dtype=np.half
            )  # This stores the x values for all repeats
            gauss_filter = np.zeros(
                (nr_bins, repeats), dtype=np.half
            )  # Stores the convolved traces for all repeats

            # Loop
            for repeat in range(repeats):
                print(repeat)
                # Calculate histogram
                psth = spk.psth([spiketrains[repeat]], resolution)
                x, y = psth.get_plottable_data()
                gauss_bins[:, repeat] = y
                # gauss_bins[:, repeat] = psth
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

            results[idx, 0] = gauss_bins
            results[idx, 1] = gauss_x
            results[idx, 2] = gauss_filter
            results[idx, 3] = gauss_average
            results[idx, 4] = np.max(gauss_average)

        # Add error handling
        except:
            if show_error == True:
                print(str(row) + " Failed")
            traceback.print_exc()
            results[idx, 0] = 0
            results[idx, 1] = 0
            results[idx, 2] = 0
            results[idx, 3] = 0
            results[idx, 4] = 0

    spikes_df["Gauss_bins"] = results[:, 0]
    spikes_df["Gauss_x"] = results[:, 1]
    spikes_df["Gauss_filter"] = results[:, 2]
    spikes_df["Gauss_average"] = results[:, 3]
    spikes_df["Gauss_max"] = results[:, 4]
    return spikes_df


def parallel_gauss_conv(stimulus, spikes_df):
    df_split = np.array_split(spikes_df, mp.cpu_count())
    pool = mp.Pool(mp.cpu_count())

    par_func = partial(spikes_conv_gauss_all_new, stimulus)
    result_df = pd.concat(pool.map(par_func, df_split))
    pool.close()
    pool.join()
    return result_df


def parallel_psth(stimulus, spikes_df, binsize):
    df_split = np.array_split(spikes_df, mp.cpu_count())
    pool = mp.Pool(mp.cpu_count())
    try:
        par_func = partial(spikes_psth_all, stimulus)
        result_df = pd.concat(pool.map(par_func, df_split))
    except:
        traceback.print_exc()
    pool.close()
    pool.join()
    return result_df


def spikes_psth_all(
    stimulus_df, spikes_df, sampling_freq=17852.767845719834, bin_size=0.05
):
    results = np.empty((len(spikes_df), 4), object)

    # Looping over pandas dataframe and the row indices
    for row, idx in zip(spikes_df.itertuples(), range(len(spikes_df))):
        # Get the index information for stimulus index and recording name

        stimulus_index = row[0][1]
        recording_name = row[0][2]

        # Extract the right row from the stimulus dataframe
        trigger_complete = stimulus_df.loc[stimulus_index, recording_name][
            "Trigger_Fr_relative"
        ]
        repeat_logic = stimulus_df.loc[stimulus_index, recording_name][
            "Stimulus_repeat_logic"
        ]
        if idx == 0 or idx == 1000:
            print(trigger_complete)

        # Need a "try" to account for cells with no spikes
        try:

            spikes, spiketrains = sp.get_spikes_whole_stimulus_new(
                row.Spikes.compressed(),
                trigger_complete,
                int(repeat_logic),
                sampling_freq,
            )

            # Define bining parameters
            nr_bins = math.ceil(spiketrains[0].t_end / bin_size) + 1

            # nr_bins_simple = math.ceil(spiketrains[0].t_end/resolution-1)

            # Prepare arrays for loop over repeats

            psth_bins = np.zeros(
                (nr_bins), dtype=np.uint8
            )  # This stores the inital bined spikes
            psth_x = np.zeros(
                (nr_bins + 1), dtype=np.half
            )  # This stores the x values for all repeats

            # Calculate histogram
            x, y = psth(spiketrains, bin_size)

            psth_bins[:] = y
            # gauss_bins[:, repeat] = psth
            # Store x values

            psth_x[:] = x

            results[idx, 0] = psth_x
            results[idx, 1] = psth_bins
            results[idx, 2] = np.mean(y)
            results[idx, 3] = np.max(y)

        except:

            traceback.print_exc()

            results[:, 0] = 0
            results[:, 1] = 0
            results[:, 2] = 0
            results[:, 3] = 0

    spikes_df["PSTH_x"] = results[:, 0]
    spikes_df["PSTH"] = results[:, 1]
    spikes_df["PSTH_mean"] = results[:, 2].astype(float)
    spikes_df["PSTH_max"] = results[:, 3].astype(float)
    return spikes_df


def psth(spike_trains, bin_size):
    """ Computes the peri-stimulus time histogram of a set of
    :class:`.SpikeTrain`. The PSTH is simply the histogram of merged spike
    events. The :code:`bin_size` defines the width of the histogram bins.

    :param spike_trains: list of :class:`.SpikeTrain`
    :param bin_size: width of the histogram bins.
    :return: The PSTH as a :class:`.PieceWiseConstFunc`
    """

    bin_count = (
        int(math.ceil((spike_trains[0].t_end - spike_trains[0].t_start) / bin_size)) + 2
    )
    bin_edges = np.arange(0, bin_count * bin_size, bin_size)
    # bins = np.linspace(spike_trains[0].t_start, spike_trains[0].t_end,
    #                   bin_count+1)

    # N = len(spike_trains)
    combined_spike_train = spike_trains[0].spikes
    for i in range(1, len(spike_trains)):
        combined_spike_train = np.append(combined_spike_train, spike_trains[i].spikes)

    vals, edges = np.histogram(
        combined_spike_train,
        bins=bin_edges,
        range=(spike_trains[0].t_start, spike_trains[0].t_end),
        density=False,
    )

    bin_size = edges[1] - edges[0]
    return edges, vals  # /(N*bin_size))


def spikes_isi_all(stimulus_df, spikes_df, sampling_freq=17852.767845719834):

    stim_idx = spikes_df.index.get_level_values(1).to_numpy()

    trigger_complete = stimulus_df["Trigger_Fr_relative"].to_numpy()
    repeat_logic = stimulus_df["Stimulus_repeat_logic"].to_numpy()

    results = np.empty((len(spikes_df), 3), object)

    for row, pos, idx in zip(spikes_df.itertuples(), stim_idx, range(len(spikes_df))):
        try:
            # print(idx)
            spikes, spiketrains = sp.get_spikes_whole_stimulus_new(
                row.Spikes.compressed(),
                trigger_complete[pos],
                int(repeat_logic[pos]),
                sampling_freq,
            )

            isi = spk.isi_profile(spiketrains)

            x, y = isi.get_plottable_data()

            results[idx, 0] = x
            results[idx, 1] = y
            results[idx, 2] = np.mean(y)

        except:
            traceback.print_exc()
            results[:, 0] = 0
            results[:, 1] = 0
            results[:, 3] = 0

    spikes_df["ISI_x"] = results[:, 0]
    spikes_df["ISI"] = results[:, 1]
    spikes_df["ISI_mean"] = results[:, 2].astype(float)
    return spikes_df


def spikes_sync_all(stimulus_df, spikes_df, sampling_freq=17852.767845719834):
    # Create empty result array for the output of the loop
    results = np.empty((len(spikes_df), 7), object)

    # Looping over pandas dataframe and the row indices
    for row, idx in zip(spikes_df.itertuples(), range(len(spikes_df))):
        # Get the index information for stimulus index and recording name

        stimulus_index = row[0][1]
        recording_name = row[0][2]

        # Extract the right row from the stimulus dataframe
        trigger_complete = stimulus_df.loc[stimulus_index, recording_name][
            "Trigger_Fr_relative"
        ]
        repeat_logic = stimulus_df.loc[stimulus_index, recording_name][
            "Stimulus_repeat_logic"
        ]

        # Need a "try" to account for cells with no spikes
        try:

            spikes, spiketrains = sp.get_spikes_whole_stimulus_new(
                row.Spikes.compressed(),
                trigger_complete,
                int(repeat_logic),
                sampling_freq,
            )

            sync = spk.spike_sync_profile(spiketrains)

            x, y = sync.get_plottable_data()

            results[idx, 0] = x
            results[idx, 1] = y
            results[idx, 2] = np.mean(y)
            results[idx, 3] = np.max(y)
            results[idx, 4] = np.min(y)
            results[idx, 5] = spk.spike_sync(spiketrains)
            results[idx, 6] = np.std(y)

        except:
            traceback.print_exc()
            results[:, 0] = 0
            results[:, 1] = 0
            results[:, 2] = 0
            results[:, 3] = 0
            results[:, 4] = 0
            results[:, 5] = 0
            results[:, 6] = 0

    spikes_df["SYNC_x"] = results[:, 0]
    spikes_df["SYNC"] = results[:, 1]
    spikes_df["SYNC_mean"] = results[:, 2].astype(float)
    spikes_df["SYNC_max"] = results[:, 3].astype(float)
    spikes_df["SYNC_min"] = results[:, 4].astype(float)
    spikes_df["SYNC_complete"] = results[:, 5]
    spikes_df["SYNC_std"] = results[:, 6].astype(float)
    return spikes_df


def chirp():
    import numpy as np
    import math
    from scipy.signal import chirp

    # saving_path = 'C:\\Users\\euler\\Desktop\\'
    # os.chdir(saving_path)
    p = {
        "nTrials": 3,
        "chirpDur_s": 30.0,
        "Contrast_Dur_s": 60.0,  # Rising chirp phase
        "chirpMinFreq_Hz": 1,
        "chirpMaxFreq_Hz": 30.0,  # Peak frequency of chirp (Hz)
        "ContrastFreq_Hz": 0.2,  # Freqency at which contrast
        # is modulated
        "tSteadyOFF_s": 3.0,  # Light OFF at beginning ...
        "tSteadyOFF2_s": 2.0,  # ... and end of stimulus
        "tSteadyON_s": 3.0,  # Light 100% ON before and
        # after chirp
        "tSteadyMID_s": 2.0,  # Light at 50% for steps
        "IHalf": 127,
        "IFull": 254,
        "dxStim_um": 1000,  # Stimulus size
        "StimType": 2,  # 1 = Box, 2 = Circle/
        "durFr_s": 1 / 60.0,  # Frame duration
        "nFrPerMarker": 3,
    }
    T_2s = 2
    T_3s = 3
    durFr_s = p["durFr_s"]  # Frame duration
    nPntChirp = int(p["chirpDur_s"] / p["durFr_s"])
    cPntChirp = int(p["Contrast_Dur_s"] / p["durFr_s"])
    Intensity = []
    t = np.linspace(0, int(p["chirpDur_s"]), nPntChirp)
    chirp_out = (
        chirp(
            t,
            p["chirpMinFreq_Hz"],
            p["chirpDur_s"],
            p["chirpMaxFreq_Hz"],
            method="logarithmic",
        )
        * 0.5
        + 0.5
    )
    a_div = math.pow(cPntChirp, 2) / 0.5
    for iT1 in range(int(T_3s / durFr_s)):
        Intensity.append(0.0)
    for iT2 in range(int(T_2s / durFr_s)):
        Intensity.append(0.5)
    for iPnt in range(nPntChirp):
        Intensity.append(chirp_out[iPnt])
    for iT1 in range(int(T_3s / durFr_s)):
        Intensity.append(0.0)
    for iT2 in range(int(T_2s / durFr_s)):
        Intensity.append(0.5)
    for iPnt in range(cPntChirp):
        t_s = iPnt * p["durFr_s"]
        IRamp = math.pow(iPnt, 2) / a_div
        Intensity.append(
            math.sin(2 * math.pi * p["ContrastFreq_Hz"] * t_s) * IRamp + 0.5
        )
    for iT2 in range(int(T_2s / durFr_s)):
        Intensity.append(0.5)
    for iT1 in range(int(T_3s / durFr_s)):
        Intensity.append(0.0)
    y = np.asarray(Intensity)
    time_pnts = len(y)
    time = np.linspace(0, (time_pnts * durFr_s), time_pnts)

    return time, y
