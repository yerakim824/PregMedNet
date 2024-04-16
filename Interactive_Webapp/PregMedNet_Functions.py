import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from sklearn.manifold import TSNE

import time
import math
from pathlib import Path

import bokeh
from bokeh.io import output_notebook, show, save,output_file
from bokeh.models import Range1d, Circle, ColumnDataSource, MultiLine,GraphRenderer,BoxSelectTool, Circle, EdgesAndLinkedNodes, HoverTool,MultiLine, NodesAndLinkedEdges, Plot, Range1d, TapTool, CustomJS, Column, TextInput
from bokeh.plotting import figure
from bokeh.palettes import Spectral4
from bokeh.plotting import from_networkx
from bokeh.palettes import Spectral4
from bokeh import events

from bokeh.io import show
from bokeh.models import ColumnDataSource, DataTable, DateFormatter, TableColumn, Row

final_color_dict = {'Disease': '#696969',
 'Anti-Infective Agents': '#e43972',
 'Immunosuppressants': '#a6bbff',
 'Hormones & Synthetic Subst': '#000080',
 'Cardiovascular Agents': '#9966cc',
 'Central Nervous System': '#ff0000',
 'Electrolytic, Caloric, Water': '#00cc99',
 'Eye, Ear, Nose Throat': '#cc99cc', 
 'Antihistamines & Comb.': '#00cc66',
 'Skin & Mucous Membrane': '#cc6699', 
 'Autonomic Drugs': '#FF5733','Gastrointestinal Drugs': '#009999',
 'Blood Form/Coagul Agents': '#003366',
 'Vitamins & Comb': '#00ff99',
 'Respiratory Tract Agents': '#0099ff',
 'Pharmaceutical Aids/Adjuvants': '#33cccc',
 'Antineoplastic Agents': '#0d53ad',
 'Serums, Toxoids, Vaccines': '#ff7ca6',
'Other Medications':'#138484'}
markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in final_color_dict.values()]


def RAW_ODDS_RATIOS():
    file_path_raw_or = Path(__file__).parents[0] / '2024_reference_tables/raw_edges.csv'
    raw_edge_df = pd.read_csv(file_path_raw_or).drop(columns=['Unnamed: 0'],axis=1)
    raw_edge_df_modi = raw_edge_df.copy()
    raw_edge_df_modi['color_clicked_modi'] = np.where(raw_edge_df_modi['color']=='#c9c9c9',raw_edge_df_modi['color_clicked'],'#0000ff')
    raw_edge_df_modi['color']='#c9c9c9'
    raw_edge_df_modi = raw_edge_df_modi.drop(columns=['color_clicked']).rename(columns={'color_clicked_modi':'color_clicked'})
    return raw_edge_df_modi

def ADJ_ODDS_RATIOS():
    file_path_adj_or = Path(__file__).parents[0] / '2024_reference_tables/adj_edges.csv'
    adj_edge_df = pd.read_csv(file_path_adj_or).drop(columns=['Unnamed: 0'],axis=1)
    adj_edge_df_modi = adj_edge_df.copy()
    adj_edge_df_modi['color_clicked_modi'] = np.where(adj_edge_df_modi['color']=='#c9c9c9',adj_edge_df_modi['color_clicked'],'#0000ff')
    adj_edge_df_modi['color']='#c9c9c9'
    adj_edge_df_modi = adj_edge_df_modi.drop(columns=['color_clicked']).rename(columns={'color_clicked_modi':'color_clicked'})
    return adj_edge_df_modi

def Interactive_Plot(data):
    file_path_nodes = Path(__file__).parents[0] / '2024_reference_tables/node_tsne.csv'
    node_df = pd.read_csv(file_path_nodes).set_index('node')
    edge_df = data
    
    ## Final Edge Info to edge_list ##
    edge_list = []
    edge_dict = edge_df.to_dict(orient='index')
    for key in edge_dict.keys():
        each_edge=(edge_dict[key]['Disease'],edge_dict[key]['Medication'],
                    {
                        # 'weight': edge_dict[key]['weight'],
                        'weight_modi':edge_dict[key]['weight_modi'],
                        'color': edge_dict[key]['color'],
                        'color_clicked': edge_dict[key]['color_clicked'],
                        'odds_ratio':edge_dict[key]['odds ratio'],
                        'p-val':edge_dict[key]['p-val']
                    }
                )
        edge_list.append(each_edge)
        
    ## Node Size based on Edges ##
    dz_edge_num = edge_df.groupby('Disease').count()[['odds ratio']]
    med_edge_num = edge_df.groupby('Medication').count()[['odds ratio']]
    node_size_df = pd.concat([dz_edge_num,med_edge_num])
    node_size_dict = node_size_df.to_dict(orient='index')
    node_size_final={}
    for key in node_size_dict.keys():
        if node_size_dict[key]['odds ratio']==1:
            node_size_dict[key]['odds ratio']=2
        node_size_final[key] = math.log2(node_size_dict[key]['odds ratio'])*7
        node_size_final[key] = node_size_dict[key]['odds ratio']

    ## Final Node Information to node_list##
    node_list = []
    node_dict = node_df.to_dict(orient='index')
    for key in node_dict.keys():
        try:
            size=node_size_final[key]
        except:
            size=1
        each_node = (key,{'pos':(node_dict[key]['tsne-origin-one-modi'],node_dict[key]['tsne-origin-two-modi']),
                            'class':node_dict[key]['New_Med_Group'], 'size':size, 'color':node_dict[key]['color']})
        node_list.append(each_node)
        
    ##### Bokeh Network Graph #####
    #Choose a title!
    output_notebook()
    #title = 'PregMedNet Network Graph'

    #Establish which categories will appear when hovering over each node
    HOVER_TOOLTIPS = [('Med/Disease', '$index'), ('start', '@start'), ('end', '@end')]
    node_hover_tool = [('Node','@index'),('Class','@class')]

    #Create a plot — set dimensions, toolbar, and title
    plot = figure(tooltips = node_hover_tool,tools="pan,wheel_zoom,box_zoom,reset", toolbar_location="above",plot_width=1350,plot_height=1200) #x_range=(-2000, 1700), y_range=(-2000, 4000),xwheel_pan, ywheel_pan,, title=title,
    plot.add_tools(HoverTool(tooltips=node_hover_tool), TapTool(), BoxSelectTool())

    G=nx.Graph()
    G.add_nodes_from(node_list)
    G.add_edges_from(edge_list)


    # edge_weight = [i['weight_modi'] for i in dict(G.edges).values()]
    # edge_color = [i['color'] for i in dict(G.edges).values()]

    fixed_nodes = node_dict.keys()
    fixed_positions = nx.get_node_attributes(G,'pos')
    node_sizes = nx.get_node_attributes(G,'size')
    node_colors = nx.get_node_attributes(G,'color')

    pos = nx.spring_layout(G,pos=fixed_positions)#,pos=fixed_positions, fixed = fixed_nodes
    edge_width = nx.get_edge_attributes(G,'weight_modi')
    edge_color = nx.get_edge_attributes(G,'color')
    edge_color_click=nx.get_edge_attributes(G,'color_clicked')


    nx.set_node_attributes(G, node_colors, 'node_color')
    nx.set_node_attributes(G, node_sizes, 'node_size')
    nx.set_edge_attributes(G, edge_color, "edge_color")
    nx.set_edge_attributes(G, edge_color_click, "edge_color_click")

    network_graph = from_networkx(G, nx.spring_layout,pos=fixed_positions, fixed = fixed_nodes,scale=10,center=(0,0))

    ######## Test #########
    network_graph.node_renderer.glyph = Circle(size='node_size', fill_color='node_color')
    network_graph.node_renderer.selection_glyph = Circle(size='node_size', fill_color='node_color')
    network_graph.node_renderer.hover_glyph = Circle(size='node_size', fill_color='node_color')

    network_graph.edge_renderer.data_source.data["line_color"] = [G.get_edge_data(a,b)['color'] for a, b in G.edges()]
    network_graph.edge_renderer.glyph = MultiLine(line_color="edge_color", line_alpha=0.2)
    network_graph.edge_renderer.selection_glyph = MultiLine(line_color= "edge_color_click",  line_alpha=1)
    network_graph.edge_renderer.hover_glyph = MultiLine(line_color="edge_color_click", line_alpha=1)
    network_graph.edge_renderer.data_source.data["line_width"] = [G.get_edge_data(a,b)['weight_modi'] for a, b in G.edges()]
    network_graph.edge_renderer.glyph.line_width = {'field': 'line_width'}
    network_graph.edge_renderer.selection_glyph.line_width = {'field': 'line_width'}
    network_graph.edge_renderer.hover_glyph.line_width = {'field': 'line_width'}

    network_graph.selection_policy = NodesAndLinkedEdges()

    #Add network graph to the plot
    plot.renderers.append(network_graph)


    plot.xgrid.visible = False
    plot.ygrid.visible = False
    plot.axis.visible = False
    plot.outline_line_color = None
    
    return plot





























