# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 15:26:53 2021
This file stores all functions which are used to synchronize the spikes with the
stimulus (trigger signals)
@author: Marvin
"""
import numpy as np
from MEA_analysis.backbone import bisection
from MEA_analysis import spike_plotly
import math
import pyspike as spk
import plotly
import plotly.graph_objects as go


def spikes_and_stimulus(spikes, stimuli):
    """
    This function looks for which spikes fall into the time at which a given stimulus
    was played and returns the indices of the spiketrains for each cell, for the first
    and last spike within the time of the stimulus
    "spike>first_trigger, spike<last_trigger"
    """
    begin_idx = np.zeros((len(stimuli), len(spikes)), dtype=int)
    end_idx = np.zeros((len(stimuli), len(spikes)), dtype=int)
    for stimulus in range(len(stimuli)):
        for cell in range(len(spikes)):
            begin_idx[stimulus, cell] = bisection(
                spikes["Spiketimestamps"].loc[cell][: spikes["Nr of spikes"].loc[cell]],
                stimuli["Begin_Fr"][stimulus],
            )
            end_idx[stimulus, cell] = bisection(
                spikes["Spiketimestamps"].loc[cell][: spikes["Nr of spikes"].loc[cell]],
                stimuli["End_Fr"][stimulus],
            )

            # validate the begin and end times. in case search for begin returns -1

            if begin_idx[stimulus, cell] == -1 and end_idx[stimulus, cell] != -1:
                begin_idx[stimulus, cell] = 0

    return begin_idx, end_idx


def extract_stimulus_spikes(spikes, stimuli, begin_idx, end_idx, begin_frs):
    """
    This function collects all the spikes that fall within a stimulus and sorts
    them into a masked numpy array. The size of the array is equal to the number
    of spikes for the cell which spikes most in the given stimulus and the number
    of cells in the recording
    """
    nr_cells = len(spikes)
    nr_spikes_stimulus = end_idx - begin_idx
    max_spikes_stimulus = np.max(nr_spikes_stimulus, 1)
    spike_df = stimuli.to_frame().copy()

    spikes_for_df = []
    for stimulus in range(len(stimuli)):
        stimulus_spikes = np.zeros((max_spikes_stimulus[stimulus], nr_cells), dtype=int)
        masked_spikes = np.zeros((max_spikes_stimulus[stimulus], nr_cells), dtype=int)
        for cell in range(nr_cells):

            if end_idx[stimulus, cell] == -1:
                masked_spikes[nr_spikes_stimulus[stimulus, cell] :, cell] = 1
                continue

            if end_idx[stimulus, cell] == begin_idx[stimulus, cell]:
                masked_spikes[nr_spikes_stimulus[stimulus, cell] :, cell] = 1
                continue

            else:
                # stimulus_spikes[:nr_spikes_stimulus
                #                [stimulus, cell], cell] = spiketimestamps[begin_idx[stimulus, cell]:
                #                                                          end_idx[stimulus, cell], cell]

                stimulus_spikes[: nr_spikes_stimulus[stimulus, cell], cell] = spikes[
                    "Spiketimestamps"
                ].loc[cell][begin_idx[stimulus, cell] : end_idx[stimulus, cell]]
                # Substract stimulus begin Fr so spikes are relative to stimulus frame
                stimulus_spikes[: nr_spikes_stimulus[stimulus, cell], cell] = (
                    stimulus_spikes[: nr_spikes_stimulus[stimulus, cell], cell]
                    - begin_frs[stimulus]
                )

                masked_spikes[nr_spikes_stimulus[stimulus, cell] :, cell] = np.ones(
                    (max_spikes_stimulus[stimulus] - nr_spikes_stimulus[stimulus, cell])
                )

        stimulus_spikes_masked = np.ma.masked_array(stimulus_spikes, mask=masked_spikes)
        spikes_for_df.append(stimulus_spikes_masked)

    spike_df.insert(1, "Spikes", spikes_for_df)
    return spike_df


def find_trigger_spikes(triggers, spikes):
    """
    This function is similar to spikes_and_stimulus, but finds the spikes that
    fall between two trigger signals within a given stimulus
    """
    begin_idx = np.zeros((len(triggers) - 1), dtype=int)
    end_idx = np.zeros((len(triggers) - 1), dtype=int)
    repeats = len(triggers) - 1
    for repeat in range(repeats):
        # print(repeat)
        try:
            begin_idx[repeat] = bisection(spikes, triggers[repeat])
            end_idx[repeat] = bisection(spikes, triggers[repeat + 1])

        except IndexError:
            begin_idx[repeat] = -1
            end_idx[repeat] = -1

        if begin_idx[repeat] == -1 and end_idx[repeat] != -1:
            begin_idx[repeat] = 0

    return begin_idx, end_idx


def index_repeats(to_index, repeat_logic, start):
    """
    This function checks how many repeats were played for each unique sequence
    of the stimulus
    """
    if start > repeat_logic - 1:
        print("Start index larger than repeat logic, return empty list")
        return []
    else:
        idx = start
        out = []
        idx_out = np.array([], dtype=int)
        while True:
            try:
                out.append(to_index[idx])
                idx_out = np.append(idx_out, idx)
                idx = idx + int(repeat_logic)

            except Exception as e:
                # print(e)
                break
        return out, idx_out


def real_trigger_windows(trigger, trigger_logic):
    """
    This function alignes the between trigger windows, so that there are the same
    for all trigger repeats. The time between two trigger signals may vary by a
    small amount between different repeats of the stimulus. This function looks
    for the smallest interval and cuts time of from all other intervals, so that
    their time is equal to the smalles interval in the stimulus
    """
    intervals = np.diff(trigger)
    trigger_begin = np.zeros(np.shape(trigger)[0] - 1)
    trigger_end = np.zeros(np.shape(trigger)[0] - 1)
    min_interval = np.array([])

    for repeat in range(trigger_logic):

        group_intervals, idx = index_repeats(intervals, trigger_logic, repeat)

        min_interval = np.append(min_interval, np.min(group_intervals))

        trigger_begin[idx] = trigger[idx]
        trigger_end[idx] = trigger[idx] + min_interval[repeat]

    return trigger_begin, trigger_end, min_interval


def get_spikes_per_trigger_type(
    spikes_df, trigger_complete, cell, trigger_index, repeat_log
):
    """
    This function gets all spikes for a specific trigger type within a stimulus.
    If a stimulus contains different sequences (Red, Green Blue flashes). This
    function will sort the spikes so that all spikes are aligned accordingly.
    """
    if trigger_index > repeat_log:
        raise ("Index for trigger type smaller than repeat logic")
    try:
        spikes = np.array(spikes_df["Spikes"].xs(cell, level="Cell index"))[
            0
        ].compressed()
    except KeyError:
        print("Cell index not found, return empty")
        return

    # First identify the right trigger sequence
    nr_trigger = np.shape(trigger_complete)[0] - 1
    nr_repeats = int(math.ceil((nr_trigger) / repeat_log))
    trigger_per_repeat = int(math.ceil(nr_trigger / nr_repeats))

    trigger_increase = (
        np.linspace(0, nr_trigger - trigger_per_repeat, nr_repeats, dtype=int)
        + trigger_index
    )

    # Calculate real trigger windows
    trigger_begin, trigger_end, interval = real_trigger_windows(
        trigger_complete, repeat_log
    )
    interval = interval[trigger_index]
    trigger_begin = trigger_begin[trigger_increase]
    trigger_end = trigger_end[trigger_increase]

    # Test the spikes
    spikes_repeats = []

    for repeat in range(nr_repeats):
        begin_idx, end_idx = find_trigger_spikes(
            np.array([trigger_begin[repeat], trigger_end[repeat]]), spikes
        )
        spikes_temp = spikes[begin_idx[0] : end_idx[0]]
        spikes_relative = spikes_temp - trigger_begin[repeat]
        spikes_repeats.append(spikes_relative)

        if repeat == 0:
            spiketrains_repeat = [
                spk.SpikeTrain(spikes_relative, edges=(0.0, interval))
            ]
        else:
            spiketrains_repeat.append(
                spk.SpikeTrain(spikes_relative, edges=(0.0, interval))
            )

    return spikes_repeats, spiketrains_repeat


def get_spikes_per_trigger_type_new(
    spikes, trigger_complete, trigger_index, repeat_log
):
    """
    This function gets all spikes for a specific trigger type within a stimulus.
    If a stimulus contains different sequences (Red, Green Blue flashes). This
    function will sort the spikes so that all spikes are aligned accordingly.
    """
    if trigger_index > repeat_log:
        raise ("Index for trigger type smaller than repeat logic")
    # try:
    #     if cell:
    #         spikes = np.array(spikes_df['Spikes'].xs(cell, level="Cell index")
    #                           )[0].compressed()
    #     elif row:
    #         spikes = np.array(spikes_df.iloc[row]["Spikes"].compressed())
    # except KeyError:
    #     print("Cell index not found, return empty")
    #     return

    # First identify the right trigger sequence
    nr_trigger = np.shape(trigger_complete)[0] - 1
    nr_repeats = int(math.ceil((nr_trigger) / repeat_log))
    trigger_per_repeat = int(math.ceil(nr_trigger / nr_repeats))
    # print(nr_trigger, nr_repeats, trigger_per_repeat)
    # BUG potentially not working correctly in cases with less trigger than should

    trigger_increase = (
        np.linspace(0, nr_trigger - trigger_per_repeat, nr_repeats, dtype=int)
        + trigger_index
    )

    # Calculate real trigger windows
    trigger_begin, trigger_end, interval = real_trigger_windows(
        trigger_complete, repeat_log
    )
    interval = interval[trigger_index]
    trigger_begin = trigger_begin[trigger_increase]
    trigger_end = trigger_end[trigger_increase]

    # Test the spikes
    spikes_repeats = []

    for repeat in range(nr_repeats):
        begin_idx, end_idx = find_trigger_spikes(
            np.array([trigger_begin[repeat], trigger_end[repeat]]), spikes
        )
        spikes_temp = spikes[begin_idx[0] : end_idx[0]]
        spikes_relative = spikes_temp - trigger_begin[repeat]
        spikes_repeats.append(spikes_relative)

        if repeat == 0:
            spiketrains_repeat = [
                spk.SpikeTrain(spikes_relative, edges=(0.0, interval))
            ]
        else:
            spiketrains_repeat.append(
                spk.SpikeTrain(spikes_relative, edges=(0.0, interval))
            )

    return spikes_repeats, spiketrains_repeat


def get_spikes_whole_stimulus(spike_df, trigger_complete, cell, repeat_logic, freq=1):
    """
    This function returns all spikes for the stimulus in a ordered fashion which means, repeats
    are aligned, same trigger types are aligned, relative times are corrected.
    """
    trigger_begin, trigger_end, interval = real_trigger_windows(
        trigger_complete, repeat_logic
    )
    relative_begin = trigger_begin[:repeat_logic]
    relative_end = np.sum(interval)
    # print(relative_begin, relative_end)
    all_spikes = []
    all_spiketrains = []
    for trigger in range(repeat_logic):
        spikes_repeat, spiketrains_repeat = get_spikes_per_trigger_type(
            spike_df, trigger_complete, cell, trigger, repeat_logic
        )
        all_spikes.append(spikes_repeat)
        all_spiketrains.append(spiketrains_repeat)

    nr_repeats = len(all_spikes[0])
    spikes_combined = []
    spiketrain_combined = []
    for repeat in range(nr_repeats):
        for trigger in range(len(all_spikes)):
            if trigger == 0:
                temp_spikes = all_spikes[trigger][repeat] / freq
                all_spiketrains[trigger][repeat].spikes = (
                    all_spiketrains[trigger][repeat].spikes / freq
                )
                spiketrain_temp = all_spiketrains[trigger][repeat]
            else:
                temp_spikes = np.append(
                    temp_spikes, all_spikes[trigger][repeat] + relative_begin[trigger]
                )
                spiketrain_temp.spikes = np.append(
                    spiketrain_temp.spikes,
                    (all_spiketrains[trigger][repeat].spikes + relative_begin[trigger])
                    / freq,
                )
        spikes_combined.append(temp_spikes)
        spiketrain_combined.append(spiketrain_temp)
        spiketrain_combined[repeat].t_end = relative_end / freq
    return spikes_combined, spiketrain_combined


def get_spikes_whole_stimulus_new(spikes, trigger_complete, repeat_logic, freq=1):
    """
    This function returns all spikes for the stimulus in a ordered fashion which means, repeats
    are aligned, same trigger types are aligned, relative times are corrected.
    """
    trigger_begin, trigger_end, interval = real_trigger_windows(
        trigger_complete, repeat_logic
    )
    relative_begin = trigger_begin[:repeat_logic]
    relative_end = np.sum(interval)
    # print(relative_begin, relative_end)
    all_spikes = []
    all_spiketrains = []
    for trigger in range(repeat_logic):
        spikes_repeat, spiketrains_repeat = get_spikes_per_trigger_type_new(
            spikes, trigger_complete, trigger, repeat_logic
        )

        all_spikes.append(spikes_repeat)
        all_spiketrains.append(spiketrains_repeat)

    # print(all_spikes)
    # For debugging
    if not all_spikes:
        print(repeat_logic, trigger_complete)
    nr_repeats = len(all_spikes[0])
    spikes_combined = []
    spiketrain_combined = []
    for repeat in range(nr_repeats):
        for trigger in range(len(all_spikes)):
            if trigger == 0:
                temp_spikes = all_spikes[trigger][repeat] / freq
                all_spiketrains[trigger][repeat].spikes = (
                    all_spiketrains[trigger][repeat].spikes / freq
                )
                spiketrain_temp = all_spiketrains[trigger][repeat]
            else:
                temp_spikes = np.append(
                    temp_spikes, all_spikes[trigger][repeat] + relative_begin[trigger]
                )
                spiketrain_temp.spikes = np.append(
                    spiketrain_temp.spikes,
                    (all_spiketrains[trigger][repeat].spikes + relative_begin[trigger])
                    / freq,
                )
        spikes_combined.append(temp_spikes)
        spiketrain_combined.append(spiketrain_temp)
        spiketrain_combined[repeat].t_end = relative_end / freq
    return spikes_combined, spiketrain_combined


def plot_raster(spiketrain):
    "Plot raster for a signle cells, OLD VERSION - UNUSED"
    try:
        plt.close(test_raster)
    except:
        pass

    avrg_isi_profile = spk.isi_profile(spiketrain)

    test_raster = plotly.subplots.make_subplots(
        rows=2, cols=1, subplot_titles=("Spiketrains"), shared_xaxes=True
    )

    # First, plot spiketrain
    for repeat in range(len(spiketrain)):

        spikes_temp = spiketrain[repeat].spikes
        # spikes_temp = spikes['Spiketimestamps'].loc[cell][:spikes['Nr of Spikes'].loc[cell]]

        nr_spikes = np.shape(spikes_temp)[0]
        yvalue = np.ones(nr_spikes) * repeat
        test_raster.add_trace(
            go.Scattergl(
                mode="markers",
                x=spikes_temp,
                y=yvalue,
                name="Repeat " + str(repeat),
                marker=dict(color="Black", size=2),
            )
        )
        # raster_axis.set_ylabel('Cell ID')
        # raster_axis.spines['top'].set_visible(False)
        # raster_axis.spines['right'].set_visible(False)

        # PLot ISI distance average
    x, y = avrg_isi_profile.get_plottable_data()
    test_raster.add_trace(
        go.Scatter(
            x=x, y=y, name="Average Isi distance", line=dict(color="Black", dash="dot")
        ),
        row=2,
        col=1,
    )
    # Set title labels
    test_raster.update_xaxes(title_text="Time in Seconds", row=2, col=1)
    test_raster.update_yaxes(title_text="Cell Index", row=1, col=1)
    test_raster.update_yaxes(title_text="ISI", row=2, col=1)
    # raster_axis[0].set_title('Spiketrains in Recording')
    test_raster.show()

    return test_raster
