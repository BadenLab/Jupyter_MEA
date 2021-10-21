import h5py
import plotly
import numpy as np
import pyspike as spk
from matplotlib import pyplot as plt
from plotly import graph_objs as go

plt.rcParams["figure.figsize"] = (30, 3)


def plot_quick_recording_overview(recording_file):

    spikes = {}
    with h5py.File(recording_file, "r") as f:
        spikes["centres"] = np.array(f["/centres"], dtype=float)
        spikes["cluster_id"] = np.array(f["/cluster_id"], dtype=int)
        spikes["times"] = np.array(f["/times"], dtype=int)
        spikes["sampling"] = np.array(f["/Sampling"], dtype=float)
        spikes["channels"] = np.array(f["/ch"], dtype=float)
        spikes["spike_freq"] = np.array(
            np.unique(spikes["cluster_id"], return_counts=True)
        )
        spikes["nr_cells"] = np.max(spikes["spike_freq"][0, :]) + 1
        spikes["cell_indices"] = np.linspace(
            1, spikes["nr_cells"], spikes["nr_cells"], dtype=int
        )
        spikes["max_spikes"] = np.max(spikes["spike_freq"][1, :])
    cells_to_load = np.unique(np.random.randint(0, spikes["nr_cells"], 1000))

    spikes["spiketimestamps"] = np.zeros(
        (spikes["max_spikes"], np.shape(cells_to_load)[0])
    )
    for cell, count in zip(cells_to_load, range(np.shape(cells_to_load)[0])):
        spikes["spiketimestamps"][0 : spikes["spike_freq"][1, cell], count] = spikes[
            "times"
        ][spikes["cluster_id"] == -1 + spikes["cell_indices"][cell]]
    spikes["spiketimestamps"] = spikes["spiketimestamps"] / spikes["sampling"]

    for cell in range(np.shape(spikes["spiketimestamps"])[1]):
        try:
            max_index = int(
                np.where(
                    spikes["spiketimestamps"][:, cell]
                    == np.max(spikes["spiketimestamps"][:, cell])
                )[0]
            )

            if cell == 0:
                spiketrains = [
                    spk.SpikeTrain(
                        spikes["spiketimestamps"][:max_index, cell],
                        edges=(0.0, np.max(spikes["spiketimestamps"][:, cell])),
                    )
                ]
            else:
                spiketrains.append(
                    spk.SpikeTrain(
                        spikes["spiketimestamps"][:max_index, cell],
                        edges=(0.0, np.max(spikes["spiketimestamps"][:, cell])),
                    )
                )
        except:
            pass

    average_spiketrain = spk.psth(spiketrains, bin_size=0.5)

    x, y = average_spiketrain.get_plottable_data()
    window = 50
    mean_y = np.convolve(y, np.ones(window), "valid") / window

    fig = go.Figure(data=go.Scatter(x=x / 60, y=y, line=dict(color="black")))
    fig.add_trace(
        go.Scatter(
            x=x / 60,
            y=mean_y,
            mode="lines",
            line=go.scatter.Line(color="red"),
            name="moving average",
        )
    )
    fig.update_layout(
        width=2000,
        height=500,
        title="Spikes per second over whole recording",
        xaxis_title="Time in Minutes",
        yaxis_title="Nr of spikes per second",
    )

    return fig


file = "D:\\Chicken_19_08_21\\Phase_01\\"
fig = plot_quick_recording_overview(file + "HS2_sorted.hdf5")
fig.write_image(file + "Recording_overview.png")
