# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 17:03:53 2021

@author: Marvin
"""

import plotly
from ipywidgets import widgets
import plotly.graph_objects as go


def initialise_subplots(clusters, stimuli):
    rows_per_stimulus = 4
    row_width = [0.02, 0.5, 0.5, 0.01]*clusters
    row_width[-1] = 0.1
    #row_width = [0.05, 0.8, 0.2]
    rows_complete = rows_per_stimulus*clusters
    
    columns_per_stimulus = 1
    columns_complete = columns_per_stimulus*stimuli
    col_width = [1, 1, 1, 1]
    extra_column = 0
    
    overview_fig = plotly.subplots.make_subplots(
        rows=rows_complete, cols=columns_complete+
        extra_column, row_width=row_width,
        column_width = col_width, vertical_spacing=0.001, shared_xaxes=True)
    
    
    overview_fig.update_layout(
    autosize=False,
    width=1200,
    height=1920)
    
    return  rows_per_stimulus, columns_per_stimulus, overview_fig



    
    
    
    
    
    
    
    
    # def plot_heatmap_new(QC_df, stimulus_info, Colours, stimulus_trace=False):
    
    
    # histogram_column = QC_df.loc[:, 'PSTH']
    # histograms = histogram_column.values
    # histogram_arr = np.zeros((len(QC_df), np.shape(histograms[0])[0]))
    # bins_column = QC_df.loc[:, 'PSTH_x']
    # bins = bins_column.values
    # bins = bins[0]
    # nr_cells = np.shape(histograms)[0]
    # cell_indices = np.linspace(0,nr_cells-1,nr_cells)
    # for cell in range(nr_cells):
    #     histogram_arr[cell, :] = histograms[cell]/np.max(histograms[cell])
    
    
    # histogram_fig = plotly.subplots.make_subplots(rows=3, cols=1,
    #                                               row_width=[0.05, 0.8, 0.2],
    #                                               vertical_spacing=0.01,
    #                                               ) 
    # histogram_fig.add_trace(go.Scatter(x=bins, y=np.mean(histogram_arr, axis=0),
    #                                    mode='lines', name="Average PSTH",
    #                                    line=dict(color="#000000"), fill='tozeroy'),
    #                         row=1, col=1)
    
    # histogram_fig.add_trace(go.Heatmap(x=bins, y=cell_indices, z=histogram_arr,
    #                                    colorscale= [
    #     [0, 'rgb(250, 250, 250)'],        #0
    #     [0.2, 'rgb(200, 200, 200)'], #10
    #     [0.4, 'rgb(150, 150, 150)'],  #100
    #     [0.6, 'rgb(100, 100, 100)'],   #1000
    #     [0.8, 'rgb(50, 50, 50)'],       #10000
    #     [1., 'rgb(0, 0, 0)'],             #100000
    #     ]), row=2, col=1)
    # histogram_fig.update_traces(showscale=False, selector=dict(type="heatmap"))
    # nr_stim = int(stimulus_info["Stimulus_repeat_logic"])*int(stimulus_info["Stimulus_repeat_sublogic"])
    # time_end = np.max(bins)
    # trigger_dur = time_end/nr_stim
    
    
    
    # if stimulus_trace == False:
    #     for i in range(nr_stim):
    #             #print(i)
    #             histogram_fig.add_trace(go.Scatter(x=[trigger_dur*i, trigger_dur*i,
    #                                         trigger_dur*(i+1), trigger_dur*(i+1), trigger_dur*i],
    #                                         y=[0, 1, 1, 0, 0], fill="toself",
    #                                         fillcolor=Colours.axcolours[i], 
    #                                         name=Colours.LED_names[i], line=dict(
    #                                         width=0), mode="lines"), row=3, col=1)
        
    # else:
    #     histogram_fig.add_trace(go.Scatter(x=stimulus_trace[:, 0], y=stimulus_trace[:, 1],
    #                                            mode="lines"), row=3, col=1)
        
    # histogram_fig.update_xaxes(showticklabels=False, row=1, col=1)
    # histogram_fig.update_yaxes(showticklabels=False, row=3, col=1)
    # histogram_fig.update_yaxes(showticklabels=False, row=1, col=1)
    # histogram_fig.update_xaxes(showticklabels=False, row=2, col=1)
    # histogram_fig.update_yaxes(showticklabels=False, row=2, col=1)
    # histogram_fig.update_xaxes(title_text='Time in Seconds', row=3, col=1)
    
    
    # histogram_fig.update_layout(
    #             {'plot_bgcolor': 'rgba(0, 0, 0, 0)'},
    #             showlegend=False)
    
    
    # return histogram_fig.show()