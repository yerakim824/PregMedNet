# import module
import streamlit as st
import pandas as pd
import numpy as np
from bokeh.plotting import figure
from bokeh.models import Band, ColumnDataSource
from bokeh.layouts import column, gridplot
import base64

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

from PregMedNet_Functions import RAW_ODDS_RATIOS, ADJ_ODDS_RATIOS, Interactive_Plot, DDI_Plot



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

tab1, tab2, tab3, tab4 = st.tabs(["Maternal Medication Effects", "Drug-Drug Interactions", "Select & Calculate","Mechanism of Action"])

with tab1:
    col1, col2 = st.columns([1.6, 2.4]) #1,4
    with col1:
        st.subheader('Select Analysis')        
        option = st.selectbox('Select Analysis Type (Default: Raw Odds Ratios)',     
                              ('Raw Odds Ratios', 'Adjusted Odds Ratios'))

        gg = st.slider("Select p-value limit (Default=0.050)", 0.010, 0.100, value=0.050 ,step=0.005)
        
        if option=='Raw Odds Ratios':
            dataframe = RAW_ODDS_RATIOS()
        else:
            dataframe = ADJ_ODDS_RATIOS()

        dataframe['95% CI (LL)']=round(dataframe['95% CI (LL)'],4)
        dataframe['95% CI (UL)']=round(dataframe['95% CI (UL)'],4)
        dataframe['95% CI']=list(zip(dataframe['95% CI (LL)'],dataframe['95% CI (UL)']))


        min_limit = st.number_input('Odds Ratio Limit: Insert min odds ratio limit (Default=No Limit)',value=np.nan)
        max_limit = st.number_input('Odds Ratio Limit: Insert max odds ratio limit (Default=No Limit, Max ={} )'.format(math.ceil(dataframe['odds ratio'].max())),value=np.nan)
        if np.isnan(min_limit):
            min_display='No Min Limit'
        else:
            min_display=min_limit
        if np.isnan(max_limit):
            max_display='No Max Limit'
        else:
            max_display=max_limit

        if ~np.isnan(min_limit):
            dataframe=dataframe[dataframe['odds ratio']>=min_limit]
        if ~np.isnan(max_limit):
            dataframe=dataframe[dataframe['odds ratio']<=max_limit]

        st.markdown("""---""")

        st.subheader("Correlation List")
        st.caption('Total Number of significant correlations: {}'.format(dataframe.shape[0]))
        data = st.dataframe(dataframe[['Disease','Medication','odds ratio','95% CI','p-val']])#,'Count'
        st.markdown("""---""")
        st.subheader("Selected Correlations")
        st.caption('Selected Correlations from the Network Graph')

    with col2:
        st.subheader("PregMedNet Network Graph")
        st.write('Analysis Type: {} | P-Value limit: {} | Odds Ratio range: {} ~ {} | Total Number of Correlations: {}'.format(option,gg,min_display,max_display,dataframe.shape[0]))
        p = Interactive_Plot(dataframe)
        show(p)
        st.bokeh_chart(p, use_container_width=True)
        

with tab2:
    col1, col2 = st.columns([1.6, 2.4]) #1,4
    with col1:
        st.subheader('Select Database for the Analysis')        
        option = st.selectbox('Select Dataset (Default: Both Cohorts)',     
                              ('Both Cohorts', 'Main Analysis Cohort', 'Validation Cohort'))        

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
    st.markdown("""
    <style>
    .subtitle {
        font-size:25px;
        text-align:center;
    }
    .centered-text {
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Select the list of confounders and calculate</div>', unsafe_allow_html=True)
    st.markdown('<div class="centered-text"> In this section, you can select maternal medications, neonatal complications, and a list of potential covariates of interest. The adjusted odds ratios will be calculated using LassoNoExp.</div>', unsafe_allow_html=True)
    disease = st.selectbox(
                            '(1) Select the neonatal complication',
                            ('Kernicterus',
                                'Transient Tachypnea of Newborn (TTN)',
                                'Respiratory Distress Syndrome (RDS)',
                                'Sepsis',
                                'Intraventricular Hemorrhage (IVH)',
                                'Gestational Alloimmune Liver Disease (GALD)',
                                'Seizures',
                                'Necrotizing Enterocolitis(NEC)',
                                'Persistent Pulmonary Hypertension of Newborn (PPHN)',
                                'Hypoglycemia',
                                'Neonatal Abstinence Syndrome (NAS)',
                                'Arrhythmia',
                                'Pneumonia',
                                'Urinary Tract Infection (UTI)',
                                'Jaundice',
                                'Small for Gestational Age (SGA)',
                                'Large for Gestational Age (LGA)',
                                'Postmaturity',
                                'Neonatal Death',
                                'Retinopathy of Prematurity (ROP)',
                                'Bronchopulmonary Dysplasia (BPD)',
                                'Anemia of Prematurity',
                                'Anemia (All)',
                                'Other Respiratory Diseases of Newborns'))
    med_class = st.selectbox(
                            '(2) Select the maternal medication class',
                            ('Anti-Infective Agents',
                                'Immunosuppressants', 'Hormones & Synthetic Subst',
                                'Cardiovascular Agents', 'Central Nervous System',
                                'Electrolytic, Caloric, Water', 'Eye, Ear, Nose Throat',
                                'Antihistamines & Comb.', 'Skin & Mucous Membrane',
                                'Autonomic Drugs', 'Gastrointestinal Drugs',
                                'Blood Form/Coagul Agents', 'Vitamins & Comb',
                                'Respiratory Tract Agents', 'Pharmaceutical Aids/Adjuvants',
                                'Antineoplastic Agents', 'Serums, Toxoids, Vaccines','Other Medications'))
    medication = st.selectbox(
    '(3) Select the maternal medication',
    ('Email', 'Home phone', 'Mobile phone'))
    covariates = st.multiselect(
    'Select the list of covariates that will be used to adjust the odds ratios',
    ['Green', 'Yellow', 'Red', 'Blue'],
    ['Yellow', 'Red'])




# aesthetics
p.toolbar.logo = None
p.toolbar_location = None
p.background_fill_alpha = 0
p.border_fill_alpha = 0
p.axis.axis_label_text_font_style = 'bold'
p.axis.axis_label_text_font_size = "16pt"
p.axis.major_label_text_font_size = "16pt"
p.legend.visible=False
