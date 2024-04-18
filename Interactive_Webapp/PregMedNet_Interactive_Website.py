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

from PregMedNet_Functions import RAW_ODDS_RATIOS, ADJ_ODDS_RATIOS, Interactive_Plot



st.set_page_config(layout='wide')


# split upper columns
st.header("PregMedNet: Associations between Neonatal Complications and Maternal Medications")
col1, col2 = st.columns([1.4, 3.6])



with col1:
    st.subheader("Select Analysis")
    option = st.selectbox('Select Analysis Type (Default: Raw Odds Ratios)',     
                          ('Raw Odds Ratios', 'Adjusted Odds Ratios'))
    
    st.subheader("Select p-value")
    gg = st.slider("p-value (Default=0.050)", 0.010, 0.100, value=0.050 ,step=0.005)
    st.write('p-value limit: ', gg)

    if option=='Raw Odds Ratios':
        dataframe = RAW_ODDS_RATIOS()
    else:
        dataframe = ADJ_ODDS_RATIOS()
        
    dataframe['95% CI (LL)']=round(dataframe['95% CI (LL)'],4)
    dataframe['95% CI (UL)']=round(dataframe['95% CI (UL)'],4)
    dataframe['95% CI']=list(zip(dataframe['95% CI (LL)'],dataframe['95% CI (UL)']))

        
    st.subheader("Select Odds Ratio Range")
    min_limit = st.number_input('Insert min odds ratio limit (default=No Limit)',value=np.nan)
    max_limit = st.number_input('Insert max odds ratio limit (max ={}) (default=No Limit)'.format(math.ceil(dataframe['odds ratio'].max())),value=np.nan)
    if np.isnan(min_limit):
        st.write('minimum odds ratio: No Limit')
        min_display='No Min Limit'
    else:
        st.write('minimum odds ratio: {}'.format(min_limit))
        min_display=min_limit
    if np.isnan(max_limit):
        st.write('maximum odds ratio: No Limit')
        max_display='No Max Limit'
    else:
        st.write('maximum odds ratio: {}'.format(max_limit))
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




# aesthetics
p.toolbar.logo = None
p.toolbar_location = None
p.background_fill_alpha = 0
p.border_fill_alpha = 0
p.axis.axis_label_text_font_style = 'bold'
p.axis.axis_label_text_font_size = "16pt"
p.axis.major_label_text_font_size = "16pt"
p.legend.visible=False
