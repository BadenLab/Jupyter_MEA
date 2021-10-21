import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import plotly
from plotly import graph_objs as go
from plotly import subplots

dir(plotly)
#
recordings = []

recordings.append("D:\\Chicken_03_08_21\\Phase_01\\Stimulus_13\\Kernel\\")
recordings.append("D:\\Chicken_04_08_21\\Phase_01\\Stimulus_2\\Kernel\\")
recordings.append("D:\\Chicken_05_08_21\\Phase_01\\Stimulus_2\\Kernel\\")
recordings.append("D:\\Chicken_06_08_21\\Stimulus_2\\Kernel\\")
recordings.append("D:\\Chicken_11_08_21\\Phase_00\\Stimulus_2\\Kernel\\")
recordings.append("D:\\Chicken_12_08_21\\Phase_00\\Stimulus_2\\Kernel\\")
recordings.append(
    "D:\\Chicken_13_08_21\\Phase_00\\bandwidth_08_thr_5\\Stimulus_2\\Kernel\\"
)
recordings.append("D:\\Chicken_14_08_21\\Phase_00\\Stimulus_2\\Kernel\\")
recordings.append("D:\\Chicken_18_08_21\\Phase_00\\Stimulus_3\\Kernel\\")

nr_files = 3000
nr_recordings = len(recordings)

Kernel_data = np.empty((nr_files * nr_recordings, 4800))
Kernel_name = np.empty(nr_files * nr_recordings, dtype=str)
Kernel_false = np.ones(nr_files * nr_recordings, dtype=bool)

count = 0

for recording in recordings:
    name = recording[3:19]
    print(name)
    for filenr in range(nr_files):
        filename = "Kernel_Cell_" + str(filenr) + ".mat"

        try:
            with h5py.File(recording + filename, "r") as hf:
                data = hf["Kernels"][:]

            data_f = data.ravel()
            std = np.std(data_f)
            if std < 10:
                Kernel_false[count] = False

            data_f = np.divide(data_f, np.max(data_f))
            data_f = np.nan_to_num(data_f, 0)
            Kernel_data[count, :] = data_f
            Kernel_name[count] = recording + filename

        except:
            Kernel_data[count, :] = 0
            Kernel_false[count] = False
            Kernel_name[count] = recording + filename

            continue

        count = count + 1


np.shape(Kernel_false)
np.shape(Kernel_data)

plt.plot(Kernel_false)
plt.show()


Kernel_data = Kernel_data[Kernel_false, :]
Kernel_name = Kernel_name[Kernel_false]


from sklearn.cluster import MeanShift
from sklearn.decomposition import PCA
import sklearn


pca = PCA(n_components=5, whiten=True)
pca_result = pca.fit_transform(Kernel_data)
pca_figure = plotly.subplots.make_subplots(rows=1, cols=1)

pca_figure.add_trace(
    go.Scatter(
        x=pca_result[:, 0],
        y=pca_result[:, 1],
        name="PCA",
        mode="markers",
        marker=dict(color=pca_result[:, 2]),
    )
)
pca_figure.update_layout(title="PCA of traces", xaxis_title="PCA1", yaxis_title="PCA2")
pca_figure.show()


shift_clusters = sklearn.cluster.MeanShift(bandwidth=2.5, n_jobs=-1).fit(pca_result)
nr_shift_clusters = int(np.max(np.unique(shift_clusters.labels_)))
print(nr_shift_clusters)
figure_shift_clustered = plotly.subplots.make_subplots(rows=1, cols=1)

figure_shift_clustered.add_trace(
    go.Scatter(
        x=pca_result[:, 0],
        y=pca_result[:, 1],
        name="PCA clustered",
        mode="markers",
        marker=dict(color=shift_clusters.labels_),
    )
)
figure_shift_clustered.update_layout(
    title="PCA of clustered responses", xaxis_title="PCA1", yaxis_title="PCA2"
)
figure_shift_clustered.show()


# %%
kernel_colour = []
kernel_colour.append("#f300ff")
kernel_colour.append("#0049ff")
kernel_colour.append("#5dff00")
kernel_colour.append("#ff0000")

nr_cells_shift = np.zeros(nr_shift_clusters)

figure_shift_c_traces = plotly.subplots.make_subplots(
    rows=nr_shift_clusters + 1, cols=1, shared_xaxes=True
)

nr_cells_shift = np.zeros(nr_shift_clusters)
mean_kernel = np.zeros((nr_shift_clusters, 1200, 4))

for i in range(nr_shift_clusters):  # nr_shift_clusters
    nr_cells_shift[i] = np.count_nonzero(shift_clusters.labels_ == i)
    kernel_end = 1200
    kernel = np.zeros((np.shape(Kernel_data[shift_clusters.labels_ == i])[0], kernel_end))


    for colour in range(4):
        kernel = Kernel_data[shift_clusters.labels_ == i,
        0 + kernel_end * colour : kernel_end + kernel_end * colour]

        mean_kernel[i, :, colour] = np.mean(kernel, 0)




figure_kernel_traces = plotly.subplots.make_subplots(
    rows=nr_shift_clusters, cols=1, shared_xaxes=True
)
for cluster in range(nr_shift_clusters):
    for colour in range(4):
        figure_kernel_traces.add_trace(
            go.Scatter(
                x=np.arange(1200),
                y=mean_kernel[cluster, :, colour],
                name="Cluster " + str(cluster + 1) + "n =" + str(nr_cells_shift[cluster]),
                line=dict(color=kernel_colour[colour]),
            ),
            row=cluster+1,
            col=1,
        )
figure_kernel_traces.update_layout(height=2400)
figure_kernel_traces.show()






plt.plot(mean_kernel[11, :, 2])
plt.show()

    cluster_mean = histogram_arr[shift_clusters.labels_ == i, :].mean(axis=0)
    cluster_std = histogram_arr[shift_clusters.labels_ == i, :].std(axis=0)
    c = c + 1
    if c > 7:
        c = 0
    figure_shift_c_traces.add_trace(
        go.Scatter(
            x=bins,
            y=cluster_mean,
            name="Cluster " + str(i + 1) + "n =" + str(nr_cells_shift[i]),
            line=dict(color=plotly.colors.qualitative.Dark2[c]),
        ),
        row=i + 1,
        col=1,
    )
    # pca_figure_clustered.add_trace(plotly.graph_objs.Scatter(x=x_values, y=
    #                                cluster_mean+cluster_std, name="Upper_std",
    #                                line=dict(color=plotly.colors.qualitative.Set2[c],
    #                                          width=0), mode='lines'),row=i+1, col=1)
    # pca_figure_clustered.add_trace(plotly.graph_objs.Scatter(x=x_values, y=
    #                                cluster_mean-cluster_std, name="lower_std",
    #                                line=dict(width=0), mode='lines', fillcolor=
    #                                plotly.colors.qualitative.Set2[c], fill=
    #                                'tonexty'),row=i+1, col=1)


figure_shift_c_traces.update_layout(
    height=1200, width=700, title_text="Responses per cluster"
)
nr_stim = 18
time_end = np.max(bins)
trigger_dur = time_end / nr_stim

for i in range(nr_stim):
    # print(i)
    figure_shift_c_traces.add_trace(
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
        row=nr_shift_clusters + 1,
        col=1,
    )


figure_shift_c_traces.show()
