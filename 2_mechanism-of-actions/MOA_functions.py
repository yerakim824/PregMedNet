import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

import matplotlib.patches as mpatches

def node_color_info():
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
    return node_color_dict


def node_color_dict_ptn_specific_info():
     node_color_dict_ptn_specific= {'node_type': {0: 'gene/protein',1: 'drug',3: 'disease',},
                                    'color': {0: '#16AEEF',1: '#946BE1',3: '#5DC264',
                                    }}
     return node_color_dict_ptn_specific
 

def legend_handles(node_color_df):
    legend_handles=[]
    for ix,row in node_color_df.iterrows():
        type_label = row['node_type']
        type_color = row['color']
        patch = mpatches.Patch(color=type_color, label=type_label)
        legend_handles.append(patch)
    return legend_handles


def make_node_list(final_kg,node_color_df):
    ## make `node_list` based on `final_kg`
    kg_node = final_kg.drop(columns=['relation','display_relation'],axis=1)

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



def make_edge_list(final_kg):
    edge_df = final_kg[['relation','display_relation','x_id','y_id']].drop_duplicates()
    edge_dict = edge_df.to_dict(orient='index')
    edge_list = []
    imp_edges = [('DB00904', 1576), (1576, 1051), (1051, 3553), (3553, '19091')] #['DB00904', '1576', '1051', '3553', '19091']
    # imp_edges = [('DB00904', '1576'), ('1576', '1051'), ('1051', '3553'), ('3553', '19091')] #['DB00904', '1576', '1051', '3553', '19091']

    for key in edge_dict.keys():
        test_edge = (edge_dict[key]['x_id'],edge_dict[key]['y_id'])
        test_edge2 = (edge_dict[key]['y_id'],edge_dict[key]['x_id'])
        if (test_edge in imp_edges) or (test_edge2 in imp_edges):
            each_edge = (edge_dict[key]['x_id'],edge_dict[key]['y_id'],{'relation':edge_dict[key]['relation'],'display_relation':edge_dict[key]['display_relation'],'weight':3,'color':'#000000'})
        else:
            each_edge = (edge_dict[key]['x_id'],edge_dict[key]['y_id'],{'relation':edge_dict[key]['relation'],'display_relation':edge_dict[key]['display_relation'],'weight':0.3,'color':'#ADADAD'})
        edge_list.append(each_edge)
    return edge_list



def construct_graph(G,med_id,dz_id_list,dz_name,med_name):
    if len(dz_id_list)==2:
        G = nx.contracted_nodes(G, dz_id_list[0], dz_id_list[1], self_loops=True, copy=True)
    elif len(dz_id_list)==3:
        G = nx.contracted_nodes(G, dz_id_list[0], dz_id_list[1], self_loops=True, copy=True)
        G = nx.contracted_nodes(G, dz_id_list[0], dz_id_list[2], self_loops=True, copy=True)
    else:
        pass
    
    num = 0
    subgraph_list = []
    shortest_paths_dict = {}
    all_shortest_paths = nx.all_shortest_paths(G, source=med_id, target=dz_id_list[0])
    for path in all_shortest_paths:
        subgraph_list+=path
        shortest_paths_dict[num]=path
        num+=1

    T= G.subgraph(subgraph_list)
    return T, shortest_paths_dict, num


def plot_subgraph(T,dz_name,med_name,dz_id_list,med_id,node_color_df): #,fig_size_tuple=(20,12)
    fig = plt.figure(figsize=(20,12))
    node_color = [i['node_color'] for i in dict(T.nodes).values()]
    edge_width = list(nx.get_edge_attributes(T,'weight').values())
    edge_colors = list(nx.get_edge_attributes(T,'color').values())
    labels = nx.get_node_attributes(T, 'node_name') 
    plt.title('{} and {}'.format(dz_name,med_name))
    fixed_positions = {med_id:(-10,0),dz_id_list[0]:(10,0)}
    fixed_nodes = fixed_positions.keys()
    pos = nx.spring_layout(T,pos=fixed_positions, fixed = fixed_nodes)
    nx.draw(T,pos, labels=labels,with_labels=True,font_size=20,edge_color=edge_colors,width=edge_width,node_color=node_color,node_size=[T.degree(n)*100 for n in T.nodes()])
    plt.legend(handles=legend_handles(node_color_df))
    plt.margins(x=0.3)
    return fig

  
def plot_shortest_paths(T,shortest_paths_dict,med_id,dz_id_list):
    for ix in shortest_paths_dict:
        imp_nodes = shortest_paths_dict[ix]
        S = T.subgraph(imp_nodes)
        node_colors = list(nx.get_node_attributes(S,'node_color').values())
        labels = nx.get_node_attributes(S, 'node_name')
        
        fig = plt.figure(figsize=(10,3))
        fixed_positions = {med_id:(-5,-1.5),dz_id_list[0]:(5,1.5)}
        fixed_nodes = fixed_positions.keys()
        pos = nx.spring_layout(T,pos=fixed_positions, fixed = fixed_nodes)
        nx.draw(S,pos,node_color=node_colors,labels=labels,with_labels=True,alpha=0.8)
        plt.margins(x=0.2)
        plt.show()
        # return fig

 
def plot_one_shortest_path(G,shortest_path,med_id,dz_id_list):
    imp_nodes = shortest_path
    S = G.subgraph(imp_nodes)
    node_colors = list(nx.get_node_attributes(S,'node_color').values())
    labels = nx.get_node_attributes(S, 'node_name')
    
    fig = plt.figure(figsize=(10,3))
    fixed_positions = {med_id:(-5,-1.5),dz_id_list[0]:(5,1.5)}
    fixed_nodes = fixed_positions.keys()
    # pos = nx.spring_layout(T,pos=fixed_positions, fixed = fixed_nodes)
    pos = nx.spring_layout(S,pos=fixed_positions, fixed = fixed_nodes)

    nx.draw(S,pos,node_color=node_colors,labels=labels,with_labels=True,alpha=0.8)
    plt.margins(x=0.2)
    return fig


## Functions for plotting the shortest paths as circular layout
def community_layout_2(g, partition,fig_x,fig_y):
    """
    Arguments:
    g -- networkx.Graph or networkx.DiGraph instance
        graph to plot
    partition -- dict mapping int node -> int community
        graph partitions
        
    Returns:
    pos -- dict mapping int node -> (float x, float y)
        node positions
    """
    pos_communities = _position_communities_2(g, partition,fig_x,fig_y, scale=3.)
    pos_nodes = _position_nodes_2(g, partition, scale=1.)

    # combine positions
    pos = dict()
    for node in g.nodes():
        pos[node] = pos_communities[node] + pos_nodes[node]

    return pos


## Functions for plotting the shortest paths as circular layout
def _position_communities_2(g,partition,fig_x,fig_y, **kwargs):
    """
    1. Fix two nodes of interests at the end
    2. Put all the other nodes in the middle of the figure
    3. Divide communities by the number of the nodes
    """
    
    ## counting the number of nodes per each community `num_nodes`, store them in pos_communities
    pos_communities = dict()
    total_nodes = len(g.nodes)
    types = set(partition.values())
    pos_communities = nx.circular_layout(types)
           
    # Scale pos_communities to each node in communities
    pos = dict()
    for node, community in partition.items():
        pos[node] = pos_communities[community]
        
    return pos


## Functions for plotting the shortest paths as circular layout
def _position_nodes_2(g, partition, **kwargs):
    """
    Positions nodes within communities.
    """
    communities = dict()
    for node, community in partition.items():
        try:
            communities[community] += [node]
        except KeyError:
            communities[community] = [node]
    
    scales = dict()
    types = set(partition.values())
    total_nodes = len(g.nodes)
    for t in types:
        num_nodes = len([x for x,y in g.nodes(data=True) if y['node_type']==t])
        if num_nodes<10:
            scales[t]=num_nodes/total_nodes*1.5
        else:
            scales[t]=num_nodes/total_nodes
    
    pos = dict()
    for ci, nodes in communities.items():
        subgraph = g.subgraph(nodes)
        pos_subgraph = nx.circular_layout(subgraph,scale=scales[ci])#**kwargs
        pos.update(pos_subgraph)
    return pos