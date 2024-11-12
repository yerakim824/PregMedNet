# import module
import streamlit as st
import pandas as pd
import numpy as np
from bokeh.plotting import figure
from bokeh.models import Band, ColumnDataSource
from bokeh.layouts import column, gridplot
import base64
import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches ## Added it (Add it to the github code later)
import pickle ## Added it (Add it to the github code later)
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

from PregMedNet_Functions import RAW_ODDS_RATIOS, ADJ_ODDS_RATIOS, Interactive_Plot, DDI_Plot, make_node_list, make_edge_list
from PregMedNet_Functions import MoA_node_color_df, MoA_legend_handles, MoA_make_node_list, MoA_make_edge_list, MoA_construct_graph, MoA_plot_subgraph, MoA_plot_shortest_paths, MoA_final_kg ## Add this to the Github

st.set_page_config(layout='wide')



st.markdown("""
    <style>
    .title {
        font-size:55px;
        font-weight:bold;
        text-align:center;
    }
    .centered-text {
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
st.markdown('<div class="title">PregMedNet</div>', unsafe_allow_html=True)
st.markdown('<div class="centered-text">PregMedNet is a tool for assessing the safety of maternal medications during pregnancy and their impact on neonatal outcomes. Discover more details in our paper [link]! ✨</div>', unsafe_allow_html=True)

st.markdown('#')

tab1, tab2, tab3 = st.tabs(["Maternal Medication Effects", "Drug-Drug Interactions","Mechanism of Action"])
#### tab1, tab2, tab3, tab4 = st.tabs(["Maternal Medication Effects", "Drug-Drug Interactions","Mechanism of Action", "Select & Calculate"])


with tab1:
    col1, col2 = st.columns([1.6, 2.4]) #1,4
    with col1:
        st.subheader('Select Analysis')        
        option = st.selectbox('Select Analysis Type (Default: Raw Odds Ratios)',     
                              ('Raw Odds Ratios', 'Adjusted Odds Ratios'))
        gg = st.slider("Select p-value limit (Default=0.050)", 0.010, 0.100, value=0.050 ,step=0.005)
        min_limit = st.number_input('Odds Ratio Limit: Insert min odds ratio limit (Default=No Limit)',value=np.nan)
        max_limit = st.number_input('Odds Ratio Limit: Insert max odds ratio limit (Default=No Limit)',value=np.nan)
        if np.isnan(min_limit):
            min_display='No Min Limit'
        else:
            min_display=min_limit
        if np.isnan(max_limit):
            max_display='No Max Limit'
        else:
            max_display=max_limit

        if st.button("Display"):
            if option=='Raw Odds Ratios':
                dataframe = RAW_ODDS_RATIOS()
            else:
                dataframe = ADJ_ODDS_RATIOS()

            dataframe['95% CI (LL)']=round(dataframe['95% CI (LL)'],4)
            dataframe['95% CI (UL)']=round(dataframe['95% CI (UL)'],4)
            dataframe['95% CI']=list(zip(dataframe['95% CI (LL)'],dataframe['95% CI (UL)']))
            if ~np.isnan(min_limit):
                dataframe=dataframe[dataframe['odds ratio']>=min_limit]
            if ~np.isnan(max_limit):
                dataframe=dataframe[dataframe['odds ratio']<=max_limit]

            st.markdown("""---""")

            st.subheader("Correlation List")
            st.caption('Total Number of significant correlations: {}'.format(dataframe.shape[0]))
            data = st.dataframe(dataframe[['Disease','Medication','odds ratio','95% CI','p-val']])#,'Count'
            st.markdown("""---""")
            # st.subheader("Selected Correlations")
            # st.caption('Selected Correlations from the Network Graph')

            with col2:
                st.subheader("PregMedNet Network Graph")
                st.write('Analysis Type: {} | P-Value limit: {} | Odds Ratio range: {} ~ {} | Total Number of Correlations: {}'.format(option,gg,min_display,max_display,dataframe.shape[0]))
                p = Interactive_Plot(dataframe)
                show(p)
                st.bokeh_chart(p, use_container_width=True)
                st.subheader("Selected Correlations")
                st.caption('Selected Correlations from the Network Graph')
        

with tab2:
    col1, col2 = st.columns([1.6, 2.4]) #1,4
    with col1:
        st.subheader('Select Database for the Analysis')        
        option = st.selectbox('Select Dataset (Default: Both Cohorts)',     
                              ('Both Cohorts', 'Main Analysis Cohort', 'Validation Cohort'))        
        if st.button("Display DDI"):
            if option=='Both Cohorts':
                file_path_ddi_edge = Path(__file__).parents[0] / '2024_reference_tables/drug-drug-interactions/ddi_edge_both_cohort_df.csv'
                ddi_edge = pd.read_csv(file_path_ddi_edge).drop(columns=['Unnamed: 0'], axis=1)
            elif option =='Main Analysis Cohort':
                file_path_ddi_edge = Path(__file__).parents[0] / '2024_reference_tables/drug-drug-interactions/ddi_edge_discovery_cohort_df.csv'
                ddi_edge = pd.read_csv(file_path_ddi_edge).drop(columns=['Unnamed: 0'], axis=1)
            else:
                file_path_ddi_edge = Path(__file__).parents[0] / '2024_reference_tables/drug-drug-interactions/ddi_edge_validation_cohort_df.csv'
                ddi_edge = pd.read_csv(file_path_ddi_edge).drop(columns=['Unnamed: 0','Unnamed: 0.1'], axis=1)

            file_path_ddi_node = Path(__file__).parents[0] / '2024_reference_tables/drug-drug-interactions/ddi_node_df.csv'
            ddi_node = pd.read_csv(file_path_ddi_node)
            data = st.dataframe(ddi_edge[['Disease','Med1','Med2','b3','pval(b3)','OR_1','OR_2','OR_12']])
            with col2:
                st.subheader("Drug-Drug Interaction Graph")
                p = DDI_Plot(ddi_node,ddi_edge)
                show(p)
                st.bokeh_chart(p, use_container_width=True)
        
with tab3:
    st.subheader('Select the maternal medication and neonatal complications')
    file_path_pair =  Path(__file__).parents[0] / '2024_reference_tables/mechanism_dz_med_df.csv'
    pair_id_df = pd.read_csv(file_path_pair).drop(columns=['Unnamed: 0'])
    #kg = pd.read_csv('2024_reference_tables/kg.csv')
    disease_display = st.selectbox(
                            '(1) Select the neonatal complication',
                            tuple(pair_id_df['dz_name_display'].unique()))
    med_name = st.selectbox(
        '(2) Select the maternal medication',
        tuple(pair_id_df[pair_id_df['dz_name_display']==disease_display]['Medication'].unique())
    )
    if st.button("Display the Mechanism of Action"):
        dz_diplay_crosswalk_df = pair_id_df[['Disease','dz_name_display']].drop_duplicates()
        dz_name = dz_diplay_crosswalk_df[dz_diplay_crosswalk_df['dz_name_display']==disease_display]['Disease'].iloc[0]
        ## Read the relevant file location
        file_path_location = '2024_reference_tables/THIRD_TAB/'
        medication_crosswalk = pd.read_csv(Path(__file__).parents[0] / '2024_reference_tables/THIRD_TAB/medication_crosswalk.csv').drop(columns=['Unnamed: 0'],axis=1)
        # Step 4: Load the dictionary from the file
        pickle_file_path = Path(__file__).parents[0] / '2024_reference_tables/THIRD_TAB/disease_crosswalk.pkl'
        with open(pickle_file_path, 'rb') as f:
            loaded_dict = pickle.load(f)
        disease_crosswalk= pd.DataFrame.from_dict(loaded_dict)
        disease_medication_pairs = pd.read_csv(Path(__file__).parents[0] / '2024_reference_tables/THIRD_TAB/disease_medication_pair.csv').drop(columns=['Unnamed: 0'],axis=1)
        

        med_id = medication_crosswalk[medication_crosswalk['Medication']==med_name]['DrugBank ID'].iloc[0]
        num_dz_list = list(disease_crosswalk[disease_crosswalk['MarketScan']==dz_name]['node_id'].unique())
        dz_id_list = num_dz_list.copy()
        """
        Data Loading in Progress...
        """
        final_kg_final = MoA_final_kg(dz_name, dz_id_list,med_id)
        node_color_df = MoA_node_color_df()
        node_list = MoA_make_node_list(final_kg_final,node_color_df)
        edge_list = MoA_make_edge_list(final_kg_final)
        ### Construct a Graph! ###
        G = nx.Graph()
        G.add_nodes_from(node_list)
        G.add_edges_from(edge_list)

        T, shortest_paths_dict, num = MoA_construct_graph(G,med_id,dz_id_list,dz_name,med_name)
        
        subgraph_plot = MoA_plot_subgraph(T,node_color_df,dz_name,med_name)
        each_path_figs = MoA_plot_shortest_paths(T,shortest_paths_dict,med_id,dz_id_list)
        col1, col2 = st.columns([1,1]) #1,4
        with col1:
            st.subheader('The Subgraph of All Shortest Paths')
            st.pyplot(subgraph_plot)
        with col2:
            st.subheader('Each Shortest Path from the Subgraph of All Shortest Paths')
            for fig in each_path_figs:
                st.pyplot(fig)

        
# with tab4:
#     st.markdown("""
#     <style>
#     .subtitle {
#         font-size:25px;
#         text-align:center;
#     }
#     .centered-text {
#         text-align: center;
#     }
#     </style>
#     """, unsafe_allow_html=True)
#     st.markdown('<div class="subtitle">Select the list of confounders and calculate</div>', unsafe_allow_html=True)
#     st.markdown('<div class="centered-text"> In this section, you can select maternal medications, neonatal complications, and a list of potential covariates of interest. The adjusted odds ratios will be calculated using LassoNoExp.</div>', unsafe_allow_html=True)
#     disease = st.selectbox(
#                             '(1) Select the neonatal complication',
#                             ('Kernicterus',
#                                 'Transient Tachypnea of Newborn (TTN)',
#                                 'Respiratory Distress Syndrome (RDS)',
#                                 'Sepsis',
#                                 'Intraventricular Hemorrhage (IVH)',
#                                 'Gestational Alloimmune Liver Disease (GALD)',
#                                 'Seizures',
#                                 'Necrotizing Enterocolitis(NEC)',
#                                 'Persistent Pulmonary Hypertension of Newborn (PPHN)',
#                                 'Hypoglycemia',
#                                 'Neonatal Abstinence Syndrome (NAS)',
#                                 'Arrhythmia',
#                                 'Pneumonia',
#                                 'Urinary Tract Infection (UTI)',
#                                 'Jaundice',
#                                 'Small for Gestational Age (SGA)',
#                                 'Large for Gestational Age (LGA)',
#                                 'Postmaturity',
#                                 'Neonatal Death',
#                                 'Retinopathy of Prematurity (ROP)',
#                                 'Bronchopulmonary Dysplasia (BPD)',
#                                 'Anemia of Prematurity',
#                                 'Anemia (All)',
#                                 'Other Respiratory Diseases of Newborns'))
#     med_class = st.selectbox(
#                             '(2) Select the maternal medication class',
#                             ('Anti-Infective Agents',
#                                 'Immunosuppressants', 'Hormones & Synthetic Subst',
#                                 'Cardiovascular Agents', 'Central Nervous System',
#                                 'Electrolytic, Caloric, Water', 'Eye, Ear, Nose Throat',
#                                 'Antihistamines & Comb.', 'Skin & Mucous Membrane',
#                                 'Autonomic Drugs', 'Gastrointestinal Drugs',
#                                 'Blood Form/Coagul Agents', 'Vitamins & Comb',
#                                 'Respiratory Tract Agents', 'Pharmaceutical Aids/Adjuvants',
#                                 'Antineoplastic Agents', 'Serums, Toxoids, Vaccines','Other Medications'))
    
#     file_path_medication = Path(__file__).parents[0] / '2024_reference_tables/node_tsne.csv'
#     med_class_df = pd.read_csv(file_path_medication)
#     med_tuple = tuple(med_class_df[med_class_df['New_Med_Group']==med_class]['node'])
    
#     confounder_list = ['SEX',
#                             'EGEOLOC',
#                             'GESTATIONAL_AGE',
#                             'AGE_MOM',
#                             'Anemia_Mom',
#                             'Asthma_Mom',
#                             'SUD_Alcohol_Mom',
#                             'Anxiety_Mom',
#                             'Bipolar_Disorder_Mom',
#                             'Cesarean_Section_Mom',
#                             'PTB_Mom',
#                             'Autoimmune_Mom',
#                             'APLS_Mom',
#                             'STD_Mom',
#                             'Hyperemesis_Gravidarum_Mom',
#                             'Headache_Mom',
#                             'Migraine_Mom',
#                             'ADHD_Mom',
#                             'Alcohol_Withdrawal_Mom',
#                             'Catatonic_Mom',
#                             'Chronic_Pain_Mom',
#                             'SUD_Cocaine_Mom',
#                             'Depression_Mom',
#                             'Eating_Disorder_Mom',
#                             'Eclampsia_Mom',
#                             'Epilepsy_Mom',
#                             'Infertility_Mom',
#                             'GDM_Mom',
#                             'SUD_Hallucinogen_Mom',
#                             'SUD_Marijuana_Mom',
#                             'Obesity_Mom',
#                             'SUD_Opi_Heroin_Mom',
#                             'SUD_Opi_Methadone_Mom',
#                             'SUD_Opioid_All_Mom',
#                             'Other_Psy_Analgesics_Mom',
#                             'Other_Psy_Anesthetics_Mom',
#                             'Other_Psy_antiparkinson_Mom',
#                             'Other_Psy_antidepressants_Mom',
#                             'Other_Psy_antiepileptics_Mom',
#                             'Other_Psy_antipsychotics_Mom',
#                             'Other_Psy_other_psychotropic_Mom',
#                             'Placenta_Abruption_Mom',
#                             'DM_Mom',
#                             'HTN_Mom',
#                             'SUD_Psychostimulant_all_Mom',
#                             'Renal_disease_Mom',
#                             'Schizophrenia_Mom',
#                             'Score_Alcohol_Mom',
#                             'Score_Chronic_Renal_Mom',
#                             'Score_Congestive_HF_Mom',
#                             'Score_Chronic_IHD_Mom',
#                             'Score_Congenital_Heart_Mom',
#                             'Score_Drug_Abuse_Mom',
#                             'Score_Eclampsia_Mom',
#                             'Score_Gestational_HTN_Mom',
#                             'Score_HIV_Mom',
#                             'Score_Mild_Preeclampsia_Mom',
#                             'Score_Placenta_Previa_Mom',
#                             'Score_Previous_Cesarean_Mom',
#                             'Score_Pul_HTN_Mom',
#                             'Score_SCD_Mom',
#                             'Score_SLE_Mom',
#                             'Score_Valvular_HD_Mom',
#                             'SUD_Sedative_All_Mom',
#                             'SUD_Sedtv_Barbiturate_Mom',
#                             'SUD_Sedtv_BZD_Mom',
#                             'Sleep_Disorders_Mom',
#                             'SUD_Smoking_Mom']
#     medication = st.selectbox(
#     '(3) Select the maternal medication',med_tuple)
#     covariates = st.multiselect(
#     'Select the list of covariates that will be used to adjust the odds ratios',
#     confounder_list,
#     ['GESTATIONAL_AGE','AGE_MOM'])
#     if st.button("Calculate"):
#         st.write('You need to upload your file!')    


