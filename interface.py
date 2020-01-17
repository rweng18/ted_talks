# Import Packages

import streamlit as st
import numpy as np
import pandas as pd

# NLP Packages
import nltk
from nltk.corpus import wordnet
# nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Plotting Package
import plotly.graph_objects as go

# ---------------------------------------------------------------------------- #
st.title('Topic Modeling TED Talk Transcripts')

# ---------------------------------------------------------------------------- #
# READ IN DATA
# ---------------------------------------------------------------------------- #
talk_df = pd.read_csv('has_transcript_clean.csv', index_col = 0)

# ---------------------------------------------------------------------------- #
# PROCESS DATA
# ---------------------------------------------------------------------------- #

# Convert dates to datetime objects
talk_df.date_recorded = talk_df.date_recorded.replace('--', np.nan)
talk_df.date_recorded = pd.to_datetime(talk_df.date_recorded, format = '%Y-%m-%d')

talk_df.upload_date = talk_df.upload_date.replace('--', np.nan)
talk_df.upload_date = pd.to_datetime(talk_df.upload_date, format = '%Y-%m-%d')

talk_df.drop(['lemmas'], inplace = True, axis = 1)

# ---------------------------------------------------------------------------- #
# MAKE HISTOGRAMS FOR ENTIRE DATASET
# ---------------------------------------------------------------------------- #
# Histogram of View Count
fig1 = go.Figure(data=[go.Histogram(x = talk_df.views,
                                   marker_color = '#d62728',
                                   opacity = 0.75)])
fig1.update_layout(title_text = 'Histogram of Ted Talk View Counts (n = 3646)',
                  xaxis_title_text = 'View Count',
                  yaxis_title_text = 'Ted Talk Count',
                  bargap = 0.1)

# Histogram of Log(View Count)
views_log = [x for x in talk_df.views if x != 0]
fig2_title = 'Histogram of Log(Ted Talk View Counts) (n = ' + str(len(views_log)) + ')'

fig2 = go.Figure(data=[go.Histogram(x = np.log(views_log),
                                    marker_color = '#d62728',
                                    opacity = 0.75)])
fig2.update_layout(title_text = fig2_title,
                   xaxis_title_text = 'Log(View Count)',
                   yaxis_title_text = 'Number of Ted Talks',
                   bargap = 0.1)

# Histogram of Talk Duration (Seconds)
fig3 = go.Figure(data = go.Histogram(x = [dur/60 for dur in talk_df.duration],
                                     xbins=dict(start=0, end=max([dur/60 for dur in talk_df.duration]), size=1),
                                     marker_color = '#d62728',
                                     opacity = 0.75))
fig3.update_layout(title_text = 'Histogram of Talk Duration (n = 3646)',
                   xaxis_title_text = 'Duration (Minutes)',
                   yaxis_title_text = 'Number of Ted Talks',
                   bargap = 0.1)

# Histogram of Date Recorded
fig4_title = 'Histogram of Date Recorded (n = ' + str(len(talk_df[talk_df.date_recorded.notnull()])) + ')'
fig4 = go.Figure(data = go.Histogram(x = talk_df[talk_df.date_recorded.notnull()].date_recorded,
                                     marker_color = '#d62728',
                                     opacity = 0.75))
fig4.update_layout(title_text = fig4_title,
                   xaxis_title_text = 'Date Recorded',
                   yaxis_title_text = 'Number of Ted Talks',
                   bargap = 0.1)

# Histogram of Date Uploaded
fig5_title = 'Histogram of Date Uploaded (n = ' + str(len(talk_df[talk_df.upload_date.notnull()])) + ')'
fig5 = go.Figure(data = go.Histogram(x = talk_df[talk_df.upload_date.notnull()].upload_date,
                                     marker_color = '#d62728',
                                     opacity = 0.75))
fig5.update_layout(title_text = fig5_title,
                   xaxis_title_text = 'Date Uploaded',
                   yaxis_title_text = 'Number of Ted Talks',
                   bargap = 0.1)

# Calculate lag in upload
upload_lag = talk_df[talk_df.upload_date.notnull()].upload_date - talk_df[talk_df.upload_date.notnull()].date_recorded
upload_lag_days = [x.days for x in upload_lag if x.days >= 0]

# Histogram of Upload Lag
fig6 = go.Figure(data = go.Histogram(x = upload_lag_days,
                                     xbins=dict(start=0, end=max(upload_lag_days), size=90),
                                     marker_color = '#d62728',
                                     opacity = 0.75))
fig6.update_layout(title_text = 'Histogram of Upload Lag (n = 3541)',
                   xaxis_title_text = 'Days Since Recorded',
                   yaxis_title_text = 'Number of Ted Talks',
                   bargap = 0.1)

# Histogram of Comments
fig7 = go.Figure(data = go.Histogram(x = talk_df.comments,
                                     marker_color = '#d62728',
                                     opacity = 0.75))
fig7.update_layout(title_text = 'Histogram of Comments (n = 3008)',
                   xaxis_title_text = 'Number of Comments',
                   yaxis_title_text = 'Number of Ted Talks',
                   bargap = 0.1)

# Histogram of Tag Length
fig8 = go.Figure(data = go.Histogram(x = talk_df.tag_len,
                                     marker_color = '#d62728',
                                     opacity = 0.75))
fig8.update_layout(title_text = 'Histogram of Number of Tags (n = 3646)',
                   xaxis_title_text = 'Number of Tags',
                   yaxis_title_text = 'Number of Ted Talks',
                   bargap = 0.1)

# Histogram of Transcript Word Count
fig10 = go.Figure(data = go.Histogram(x = talk_df.transcript_wc,
                                      marker_color = '#d62728',
                                      opacity = 0.75))
fig10.update_layout(title_text = 'Histogram of Transcript Word Count (n = 3646)',
                   xaxis_title_text = 'Transcript Word Count',
                   yaxis_title_text = 'Number of Ted Talks',
                   bargap = 0.1)

# Histgram of Number of Distinct Tokens
fig11 = go.Figure(data = go.Histogram(x = talk_df.lemma_len,
                                      marker_color = '#d62728',
                                      opacity = 0.75))
fig11.update_layout(title_text = 'Histogram of Distinct Tokens (n = 3646)',
                   xaxis_title_text = 'Number of Distinct Tokens',
                   yaxis_title_text = 'Number of Ted Talks',
                   bargap = 0.1)

# ---------------------------------------------------------------------------- #
# Create different project sections
# Select Project Section
page = st.sidebar.selectbox('Project Section',
                           ('Overview',
                            'Exploratory Data Analysis',
                            'Topic Modeling',
                            'TED Talk Recommender'))

if page == 'Overview':
    st.write(talk_df)
    st.write('Description of Project')

if page == 'Exploratory Data Analysis':

    option = st.sidebar.selectbox(
        'Features to Explore',
         ('Audience Engagement', 'Linguistic', 'Temporal'))

    if option == 'Audience Engagement':
        st.plotly_chart(fig1) # views
        st.plotly_chart(fig2) # log(views)
        st.plotly_chart(fig7) # comments

    elif option == 'Linguistic':
        st.plotly_chart(fig8)  # ted tags
        st.plotly_chart(fig10) # transcript word count
        st.plotly_chart(fig11) # distinct tokens

    else:
        st.plotly_chart(fig3) # duration
        st.plotly_chart(fig4)
        st.plotly_chart(fig5)
        st.plotly_chart(fig6)

if page == 'Topic Models':
    st.write('Topic modeling to be updated')

if page == 'TED Talk Recommender':
    st.write('Front end for recommendation system')
