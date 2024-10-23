import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import networkx as nx

from sklearn.manifold import TSNE

import time
import math

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

from pathlib import Path
import os

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
    # file_path_adj_or = Path(__file__).parents[0] / '2024_reference_tables/adj_edges.csv' ## Previous Results
    file_path_adj_or = Path(__file__).parents[0] / '2024_reference_tables/adj_edges_modified.csv'
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
    dz_list = list(dz_edge_num.index)
    node_size_final={}
    for key in node_size_dict.keys():
        if node_size_dict[key]['odds ratio']<=4:
            node_size_dict[key]['odds ratio']=4
        if key in dz_list:
            node_size_final[key] = math.log2(node_size_dict[key]['odds ratio'])*5
        else:
            node_size_final[key] = math.log2(node_size_dict[key]['odds ratio'])*3

    ## Final Node Information to node_list##
    node_list = []
    node_dict = node_df.to_dict(orient='index')
    for key in node_dict.keys():
        try:
            size=node_size_final[key]
        except:
            size=4
        each_node = (key,{'pos':(node_dict[key]['tsne-origin-one-modi'],node_dict[key]['tsne-origin-two-modi']),
                            'class':node_dict[key]['New_Med_Group'], 'size':size, 'color':node_dict[key]['color']})
        node_list.append(each_node)
        
    ##### Bokeh Network Graph #####
    #Choose a title!
    # output_notebook()
    title = 'PregMedNet Network Graph'

    #Establish which categories will appear when hovering over each node
    HOVER_TOOLTIPS = [('Med/Disease', '$index'), ('start', '@start'), ('end', '@end')]
    node_hover_tool = [('Node','@index'),('Class','@class')]

    #Create a plot — set dimensions, toolbar, and title
    plot = figure(tooltips = node_hover_tool,tools="pan,wheel_zoom,box_zoom,reset", toolbar_location="above",width=1000,height=1000) #,plot_width=1350,plot_height=1200
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


def DDI_Plot(ddi_node,ddi_edge):
    ### define attributes of the edges ###
    final_edge_df = ddi_edge[['Med1','Med2','b3','pval(b3)']]
    final_edge_df['weight']=-np.log10(final_edge_df['pval(b3)'])/3
    max_weight = np.sort(final_edge_df['weight'].unique())[-2]
    final_edge_df['weight_modi']=np.where(final_edge_df['weight']==np.inf,max_weight,final_edge_df['weight'])
    final_edge_df['color']=np.where(final_edge_df['b3']<0,'#0000ff','#FF0000')
    final_edge_df['color_clicked']=np.where(final_edge_df['b3']<0,'#0000ff','#FF0000')
    edge_list = []
    edge_dict = final_edge_df.to_dict(orient='index')
    edge_dict
    for key in edge_dict.keys():
        each_edge=(edge_dict[key]['Med1'],edge_dict[key]['Med2'],
                    {
                        # 'weight': edge_dict[key]['weight'],
                        'weight_modi':edge_dict[key]['weight_modi'],
                        'color': edge_dict[key]['color'],
                        'color_clicked': edge_dict[key]['color_clicked'],
                        'b3':edge_dict[key]['b3'],
                        'pval(b3)':edge_dict[key]['pval(b3)']
                    }
                )
        edge_list.append(each_edge)
    
    ### define attributes of the nodes ###
    node_df = ddi_node.set_index('node')
    
    ## Add node size
    node_dict = node_df.to_dict(orient='index')
    med1_count = ddi_edge.groupby('Med1').count()[['Disease']].reset_index().rename(columns={'Med1':'Medication'})
    med2_count = ddi_edge.groupby('Med2').count()[['Disease']].reset_index().rename(columns={'Med2':'Medication'})
    node_size_df = pd.concat([med1_count,med2_count]).groupby('Medication').sum().rename(columns={'Disease':'Count'})
    node_size_dict = node_size_df.to_dict(orient='index')
    node_size_dict
    node_size_final={}
    for key in node_size_dict.keys():
        if node_size_dict[key]['Count']<1:
            node_size_dict[key]['Count']=4
        else:
            node_size_final[key] = 15 #node_size_dict[key]['Count']*2
            
    node_list = []
    for key in node_dict.keys():
        try:
            size=node_size_final[key]
        except:
            size=4
        each_node = (key,{'pos':(node_dict[key]['tsne-origin-one-modi'],node_dict[key]['tsne-origin-two-modi']),
                            'class':node_dict[key]['New_Med_Group'], 'size':size, 'color':node_dict[key]['color']})
        node_list.append(each_node)
        
    ### bokeh network graph ###
    output_notebook()
    title = 'Drug-Drug Interactions Network Graph'

    #Establish which categories will appear when hovering over each node
    HOVER_TOOLTIPS = [('Med/Disease', '$index'), ('start', '@start'), ('end', '@end')]
    node_hover_tool = [('Node','@index'),('Class','@class')]

    #Create a plot — set dimensions, toolbar, and title
    plot = figure(tooltips = node_hover_tool, tools="pan,wheel_zoom,box_zoom,reset", toolbar_location="above",width=1000,height=1000) #title=title, 
    
    plot.add_tools(HoverTool(tooltips=node_hover_tool), TapTool(), BoxSelectTool())

    G=nx.Graph()
    G.add_nodes_from(node_list)
    G.add_edges_from(edge_list)


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

    ddi_graph = from_networkx(G, nx.spring_layout,pos=fixed_positions, fixed = fixed_nodes,scale=10,center=(0,0))

    ######## Test #########
    ddi_graph.node_renderer.glyph = Circle(size='node_size', fill_color='node_color')
    ddi_graph.node_renderer.selection_glyph = Circle(size='node_size', fill_color='node_color')
    ddi_graph.node_renderer.hover_glyph = Circle(size='node_size', fill_color='node_color')

    ddi_graph.edge_renderer.data_source.data["line_color"] = [G.get_edge_data(a,b)['color'] for a, b in G.edges()]
    ddi_graph.edge_renderer.glyph = MultiLine(line_color="edge_color", line_alpha=0.2)
    ddi_graph.edge_renderer.selection_glyph = MultiLine(line_color= "edge_color_click",  line_alpha=1)
    ddi_graph.edge_renderer.hover_glyph = MultiLine(line_color="edge_color_click", line_alpha=1)
    ddi_graph.edge_renderer.data_source.data["line_width"] = [G.get_edge_data(a,b)['weight_modi'] for a, b in G.edges()]
    ddi_graph.edge_renderer.glyph.line_width = {'field': 'line_width'}
    ddi_graph.edge_renderer.selection_glyph.line_width = {'field': 'line_width'}
    ddi_graph.edge_renderer.hover_glyph.line_width = {'field': 'line_width'}

    ddi_graph.selection_policy = NodesAndLinkedEdges()

    #Add network graph to the plot
    plot.renderers.append(ddi_graph)


    plot.xgrid.visible = False
    plot.ygrid.visible = False
    plot.axis.visible = False
    plot.outline_line_color = None
    
    return plot


def make_node_list(sel_sel_kg):
    node_color = {
            'node_type': [
                'gene/protein', 
                'drug', 
                'biological_process', 
                'molecular_function', 
                'cellular_component', 
                'disease'
            ],
            'color': [
                '#16AEEF', 
                '#946BE1', 
                '#FF781E', 
                '#FF9F21', 
                '#F9CF57', 
                '#5DC264'
            ]
        }
    node_color = pd.DataFrame(node_color)
    kg_node = sel_sel_kg.drop(columns=['relation','display_relation'],axis=1)
    x_nodes = [i for i in kg_node.columns if i.__contains__('x_')]
    y_nodes = [i for i in kg_node.columns if i.__contains__('y_')]
    x_nodes_new = {}
    for i in x_nodes:
        new_name = 'node'+i[1:]
        x_nodes_new[i]=new_name
    y_nodes_new = {}
    for i in y_nodes:
        new_name = 'node'+i[1:]
        y_nodes_new[i]=new_name
    kg_node_x = kg_node[x_nodes].rename(columns=x_nodes_new)
    kg_node_y = kg_node[y_nodes].rename(columns=y_nodes_new)
    kg_node_merge = pd.concat([kg_node_x,kg_node_y]).drop_duplicates()
    kg_node_merge = pd.merge(kg_node_merge,node_color,on='node_type',how='left')
    kg_node_merge = kg_node_merge.reset_index().drop(columns=['index'],axis=1)
    merging_types = ['gene/protein',  'biological_process', 'molecular_function','cellular_component']
    kg_node_merge['node_type_merged']=np.where(kg_node_merge['node_type'].isin(merging_types),'Biology',kg_node_merge['node_type'])
    kg_node_merge
    node_dict = kg_node_merge.to_dict(orient='index')
    node_list = []
    for key in node_dict.keys():
        each_node=(node_dict[key]['node_id'],
        {'node_name':node_dict[key]['node_name'],'node_type':node_dict[key]['node_type'],
        'node_color':node_dict[key]['color'],
        'node_type_merged':node_dict[key]['node_type_merged']}) #'node_source':node_dict[key]['node_source'],
        node_list.append(each_node)
    return node_list

def make_edge_list(sel_sel_kg):
    edge_df = sel_sel_kg[['relation','display_relation','x_id','y_id']].drop_duplicates()
    edge_dict = edge_df.to_dict(orient='index')

    edge_list = []
    for key in edge_dict.keys():
        each_edge = (edge_dict[key]['x_id'],edge_dict[key]['y_id'], {'relation':edge_dict[key]['relation'],'display_relation':edge_dict[key]['display_relation'],'weight':0.3,'color':'#ADADAD'})
        edge_list.append(each_edge)
    return edge_list
    


def MoA_final_kg(dz_name, dz_id_list, med_id):
    kg = pd.read_csv(Path(__file__).parents[0] / '2024_reference_tables/kg.csv')
    
    if dz_name in ['BPD_OLD_Baby','Jaundice_Baby']:
        sel_relation = ['protein_protein', 'bioprocess_protein', 'molfunc_protein', 'cellcomp_protein', 
                        'bioprocess_bioprocess', 'molfunc_molfunc', 'cellcomp_cellcomp']
        disease_kg = kg[(kg['relation'] == 'disease_protein') & (kg['x_id'].isin(dz_id_list))]
        disease_kg_y = kg[(kg['relation'] == 'disease_protein') & (kg['y_id'].isin(dz_id_list))]
    
    elif dz_name in ['LGA_Baby', 'Neonatal_Death_Baby', 'Sepsis_Baby', 'SGA_Baby', 'IVH_Baby']:
        sel_relation = ['disease_protein', 'phenotype_phenotype', 'disease_phenotype_positive', 'disease_phenotype_negative',
                        'protein_protein', 'bioprocess_protein', 'molfunc_protein', 'cellcomp_protein', 
                        'bioprocess_bioprocess', 'molfunc_molfunc', 'cellcomp_cellcomp']
        disease_kg = kg[(kg['x_type'] == 'effect/phenotype') & (kg['x_id'].isin(dz_id_list))]
        disease_kg_y = kg[(kg['y_type'] == 'effect/phenotype') & (kg['y_id'].isin(dz_id_list))]
    
    elif dz_name == 'NAS_Baby':
        sel_relation = ['disease_disease', 'protein_protein', 'bioprocess_protein', 'molfunc_protein', 'cellcomp_protein', 
                            'bioprocess_bioprocess', 'molfunc_molfunc', 'cellcomp_cellcomp']
        disease_kg = kg[(kg['relation'] == 'disease_protein') & (kg['x_id'].isin(dz_id_list))]
        disease_kg_y = kg[(kg['relation'] == 'disease_protein') & (kg['y_id'].isin(dz_id_list))]
    
    elif dz_name == 'Hypoglycemia_Baby':
        sel_relation = ['disease_protein', 'phenotype_phenotype', 'disease_phenotype_positive', 'disease_phenotype_negative',
                        'protein_protein', 'bioprocess_protein', 'molfunc_protein', 'cellcomp_protein', 
                        'bioprocess_bioprocess', 'molfunc_molfunc', 'cellcomp_cellcomp']
        disease_kg = kg[(kg['relation'] == 'disease_protein') & (kg['x_id'].isin(dz_id_list))]
        disease_kg_y = kg[(kg['relation'] == 'disease_protein') & (kg['y_id'].isin(dz_id_list))]
    
    elif dz_name == 'RDS_Baby':
        sel_relation = ['protein_protein', 'bioprocess_protein', 'molfunc_protein', 'cellcomp_protein', 
                        'bioprocess_bioprocess', 'molfunc_molfunc', 'cellcomp_cellcomp']
        disease_kg = kg[(kg['x_type'] == 'disease') & (kg['x_id'].isin(dz_id_list))]
        disease_kg_y = kg[(kg['x_type'] == 'disease') & (kg['y_id'].isin(dz_id_list))]
    
    elif dz_name == 'ROP_Baby':
        sel_relation = ['disease_protein', 'phenotype_phenotype', 'disease_phenotype_positive', 'disease_phenotype_negative',
                        'protein_protein', 'bioprocess_protein', 'molfunc_protein', 'cellcomp_protein', 
                        'bioprocess_bioprocess', 'molfunc_molfunc', 'cellcomp_cellcomp']
        rop_effect_kg = kg[(kg['x_type'] == 'effect/phenotype') & (kg['x_id'].isin(dz_id_list))]
        rop_effect_kg_y = kg[(kg['y_type'] == 'effect/phenotype') & (kg['y_id'].isin(dz_id_list))]
        rop_disease_kg = kg[(kg['x_type'] == 'disease') & (kg['x_id'].isin(dz_id_list))]
        rop_disease_kg_y = kg[(kg['y_type'] == 'disease') & (kg['y_id'].isin(dz_id_list))]
        disease_kg = pd.concat([rop_effect_kg, rop_disease_kg])
        disease_kg_y = pd.concat([rop_effect_kg_y, rop_disease_kg_y])
    
    elif dz_name == 'UTI_Baby':
        sel_relation = ['indication', 'protein_protein', 'bioprocess_protein', 'molfunc_protein', 'cellcomp_protein', 
                        'bioprocess_bioprocess', 'molfunc_molfunc', 'cellcomp_cellcomp']
        disease_kg = kg[(kg['x_type'] == 'disease') & (kg['x_id'].isin(dz_id_list))]
        disease_kg_y = kg[(kg['x_type'] == 'disease') & (kg['y_id'].isin(dz_id_list))]
    
    elif dz_name in ['Anemia_All_Baby', 'Anemia_AOP_Baby']:
        sel_relation = ['disease_protein', 'phenotype_phenotype', 'disease_phenotype_positive', 'disease_phenotype_negative',
                        'protein_protein', 'bioprocess_protein', 'molfunc_protein', 'cellcomp_protein', 
                        'bioprocess_bioprocess', 'molfunc_molfunc', 'cellcomp_cellcomp']
        disease_kg = kg[(kg['x_type'] == 'disease') & (kg['x_id'].isin(dz_id_list))]
        disease_kg_y = kg[(kg['y_type'] == 'disease') & (kg['y_id'].isin(dz_id_list))]
    
    else:
        return pd.DataFrame()  # Return an empty DataFrame if no conditions match

    sel_kg = kg[kg['relation'].isin(sel_relation)]
    drug_kg = kg[(kg['relation'] == 'drug_protein') & (kg['x_id'] == med_id)]
    drug_kg_y = kg[(kg['relation'] == 'drug_protein') & (kg['y_id'] == med_id)]
    
    final_kg = pd.concat([sel_kg, disease_kg, disease_kg_y, drug_kg, drug_kg_y])

    drop_col_dict = {
        'NAS_Baby': ['PRKACA'],
        'Hypoglycemia_Baby': ['ELF2', 'Foster-Kennedy syndrome', 'angiotensin-mediated vasoconstriction involved in regulation of systemic arterial blood pressure', 'sternocostal joint'],
        'BPD_OLD_Baby': [],
        'Jaundice_Baby': ['SLC22A5', 'catecholamine metabolic process'],
        'LGA_Baby': ['CTSS', 'kleptomania', 'outer dense fiber'],
        'Neonatal_Death_Baby': ['KIR3DL1', 'ovarian seromucinous tumor'],
        'RDS_Baby': ['NR1H4', 'Polydactyly affecting the 4th finger'],
        'Sepsis_Baby': [],
        'SGA_Baby': ['voltage-gated sodium channel complex'],
        'ROP_Baby': ['defense response'],
        'UTI_Baby': ['DNTT'],
        'IVH_Baby': ["isoflavone 4'-O-methyltransferase activity"],
        'Seizures_Baby': [],
        'Anemia_All_Baby': ['CMKLR1'],
        'Anemia_AOP_Baby': ['Wrist flexion contracture']
    }
    
    drop_names = drop_col_dict.get(dz_name, [])
    final_kg = final_kg[~final_kg['x_name'].isin(drop_names)]
    final_kg = final_kg[~final_kg['y_name'].isin(drop_names)]
    
    return final_kg



def MoA_node_color_df():
    node_color_dict= {'node_type': {0: 'gene/protein',
                                  1: 'drug',
                                  2: 'effect/phenotype',
                                  3: 'disease',
                                  4: 'biological_process',
                                  5: 'molecular_function',
                                  6: 'cellular_component',
                                  7: 'exposure',
                                  8: 'pathway',
                                  9: 'anatomy'},
                                 'color': {0: '#16AEEF',
                                  1: '#946BE1',
                                  2: '#41afaa',
                                  3: '#5DC264',
                                  4: '#FF781E',
                                  5: '#FF9F21',
                                  6: '#F9CF57',
                                  7: '#466eb4',
                                  8: '#00a0e1',
                                  9: '#e6a532'}}

    node_color_df = pd.DataFrame(node_color_dict)
    return node_color_df
    

def MoA_legend_handles(node_color_df):
    legend_handles=[]
    for ix,row in node_color_df.iterrows():
        type_label = row['node_type']
        type_color = row['color']
        patch = mpatches.Patch(color=type_color, label=type_label)
        legend_handles.append(patch)
    return legend_handles


def MoA_make_node_list(final_kg_final,node_color_df):
    ## make `node_list` based on `final_kg`
    kg_node = final_kg_final.drop(columns=['relation','display_relation'],axis=1)

    x_nodes = [i for i in kg_node.columns if i.__contains__('x_')]
    y_nodes = [i for i in kg_node.columns if i.__contains__('y_')]

    x_nodes_new = {}
    for i in x_nodes:
        new_name = 'node'+i[1:]
        x_nodes_new[i]=new_name

    y_nodes_new = {}
    for i in y_nodes:
        new_name = 'node'+i[1:]
        y_nodes_new[i]=new_name

    kg_node_x = kg_node[x_nodes].rename(columns=x_nodes_new)
    kg_node_y = kg_node[y_nodes].rename(columns=y_nodes_new)
    kg_node_merge = pd.concat([kg_node_x,kg_node_y]).drop_duplicates()
    kg_node_merge = pd.merge(kg_node_merge,node_color_df,on='node_type',how='left')
    kg_node_merge = kg_node_merge.reset_index().drop(columns=['index'],axis=1)

    merging_types = ['gene/protein',  'biological_process', 'molecular_function','cellular_component']
    kg_node_merge['node_type_merged']=np.where(kg_node_merge['node_type'].isin(merging_types),'Biology',kg_node_merge['node_type'])
    kg_node_merge

    node_dict = kg_node_merge.to_dict(orient='index')

    node_list = []
    for key in node_dict.keys():
        each_node=(node_dict[key]['node_id'],
        {'node_name':node_dict[key]['node_name'],'node_type':node_dict[key]['node_type'],
        'node_color':node_dict[key]['color'],
        'node_type_merged':node_dict[key]['node_type_merged']}) #'node_source':node_dict[key]['node_source'],
        node_list.append(each_node)
    
    return node_list


def MoA_make_edge_list(final_kg_final):
    edge_df = final_kg_final[['relation','display_relation','x_id','y_id']].drop_duplicates()
    edge_dict = edge_df.to_dict(orient='index')
    edge_list = []
    for key in edge_dict.keys():
        each_edge = (edge_dict[key]['x_id'],edge_dict[key]['y_id'],
        {'relation':edge_dict[key]['relation'],'display_relation':edge_dict[key]['display_relation'],'weight':0.3,'color':'#ADADAD'})
        edge_list.append(each_edge)
    return edge_list


def MoA_construct_graph(G,med_id,dz_id_list,dz_name,med_name):
    if len(dz_id_list)==2:
        G = nx.contracted_nodes(G, dz_id_list[0], dz_id_list[1], self_loops=False, copy=True)
    elif len(dz_id_list)==3:
        G = nx.contracted_nodes(G, dz_id_list[0], dz_id_list[1], self_loops=False, copy=True)
        G = nx.contracted_nodes(G, dz_id_list[0], dz_id_list[2], self_loops=False, copy=True)
    else:
        pass
    
    num = 0
    subgraph_list = []
    shortest_paths_dict = {}
    for path in nx.all_shortest_paths(G, source=med_id, target=dz_id_list[0]):
        subgraph_list+=path
        shortest_paths_dict[num]=path
        num+=1

    T= G.subgraph(subgraph_list)
    return T, shortest_paths_dict, num

def MoA_plot_subgraph(T, node_color_df, dz_name, med_name, fig_size_tuple=(20, 14)):
    fig, ax = plt.subplots(figsize=fig_size_tuple)
    node_color = [i['node_color'] for i in dict(T.nodes).values()]
    labels = nx.get_node_attributes(T, 'node_name')
    
    #ax.set_title('{} and {}'.format(dz_name, med_name))
    pos = nx.spring_layout(T)  # You can choose the layout of your graph here
    nx.draw(T, pos, labels=labels, with_labels=True, font_size=20, edge_color='#DBDBDB', node_color=node_color, node_size=[T.degree(n)*100 for n in T.nodes()], ax=ax)
    
    handles = MoA_legend_handles(node_color_df)
    ax.legend(handles=handles,loc='upper left')
    plt.margins(x=0.3)

    return fig
    
def MoA_plot_shortest_paths(T, shortest_paths_dict, med_id, dz_id_list, fig_size_tuple=(10, 3)):
    figs = []
    for ix in shortest_paths_dict:
        imp_nodes = shortest_paths_dict[ix]
        S = T.subgraph(imp_nodes)
        node_colors = list(nx.get_node_attributes(S, 'node_color').values())
        labels = nx.get_node_attributes(S, 'node_name')

        fig, ax = plt.subplots(figsize=fig_size_tuple)
        fixed_positions = {med_id: (-5, -1.5), dz_id_list[0]: (5, 1.5)}
        fixed_nodes = fixed_positions.keys()
        pos = nx.spring_layout(S, pos=fixed_positions, fixed=fixed_nodes)
        nx.draw(S, pos, node_color=node_colors, labels=labels, with_labels=True, alpha=0.8, ax=ax)
        #ax.set_title(f'Shortest Path {ix}')
        plt.margins(x=0.2)
        
        figs.append(fig)
    
    return figs