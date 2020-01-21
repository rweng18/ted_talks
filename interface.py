# Import Packages

import streamlit as st

from collections import Counter
import numpy as np
import pandas as pd
import pickle

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
# Load in data on TED talks
talk_df = pd.read_csv('has_transcript_clean.csv', index_col = 0)

# Load in tokenized data
with open('all_tok.pkl', 'rb') as file:
    tok_corpus = pickle.load(file)

# Create dictionary of token counts for entire tok_corpus
word_bank = dict(Counter(tok_corpus))
words_less_than_100 = {key: value for key, value in word_bank.items() if value < 100}
words_more = {key: value for key, value in word_bank.items() if value >= 100}

# Load in number of documents tokens appear in
with open('doc_tok_counts.pkl', 'rb') as file:
    doc_counts = pickle.load(file)
docs_less_than_100 = {key: value for key, value in doc_counts.items() if value < 100}
docs_more = {key: value for key, value in doc_counts.items() if value >= 100}

# Load in tokens by year
with open('year_tok.pkl', 'rb') as file:
    tok_year_corpus = pickle.load(file)

# Load in data on final LDA model
lda_top_words = pd.read_csv('lda_15_topics_words.csv', index_col = 0)

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

# Histogram of Token Occurrence in Entire Corpus
fig12a = go.Figure(data = go.Histogram(x = list(words_less_than_100.values()),
                                      marker_color = '#d62728',
                                      opacity = 0.75))
fig12a.update_layout(title_text = 'Histogram of Token Frequency in Corpus (n = 48,935)',
                   xaxis_title_text = 'Token Frequency in Corpus',
                   yaxis_title_text = 'Number of Tokens',
                   bargap = 0.1)

fig12b = go.Figure(data = go.Histogram(x = list(words_more.values()),
                                       xbins=dict(start=0, end=max(list(words_more.values())), size=200),
                                       marker_color = '#d62728',
                                       opacity = 0.75))
fig12b.update_layout(title_text = 'Histogram of Token Frequency in Corpus (n = 3,102)',
                   xaxis_title_text = 'Token Frequency in Corpus',
                   yaxis_title_text = 'Number of Tokens',
                   bargap = 0.1)

# ---------------------------------------------------------------------------- #
# PROJECT SECTIONS
# ---------------------------------------------------------------------------- #

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

    eda = st.sidebar.selectbox('Corpus Level', ('All Talks', 'Talks By Year'))

    if eda == 'All Talks':
        feat = st.sidebar.selectbox('Features to Explore',
                                   ('Audience Engagement',
                                    'Linguistic',
                                    'Temporal',
                                    'Token Frequency'))

        if feat == 'Audience Engagement':
            st.plotly_chart(fig1) # views
            st.plotly_chart(fig2) # log(views)
            st.plotly_chart(fig7) # comments

        elif feat == 'Linguistic':
            st.plotly_chart(fig8)   # ted tags
            st.plotly_chart(fig10)  # transcript word count
            st.plotly_chart(fig11)  # distinct tokens

        elif feat == 'Token Frequency':
            st.plotly_chart(fig12a) # token freq in corpus 0-100
            st.plotly_chart(fig12b) # token freq in corpus 100+

            # Slider for top nmin to nmax tokens
            min_val = st.text_input('Minimum Ranked Token (1 to 52037)', 1)
            max_val = st.text_input('Maximum Ranked Token (1 to 52037)', 300)
            values = st.slider("nmin to nmax", int(min_val), int(max_val), (int(min_val), int(int(max_val)/6)))
            top_n_corpus = sorted(word_bank, key=word_bank.get, reverse=True)[values[0]-1:values[1]-1]
            fig13_title = 'Top ' + str(values[0]) + ' to ' + str(values[1]) + ' Tokens in Corpus'

            # Top nmin to nmax tokens in corpus
            fig13 = go.Figure(data = go.Bar(x = top_n_corpus,
                                    y = [word_bank[word] for word in top_n_corpus],
                                    marker_color = '#d62728',
                                    opacity = 0.75))
            fig13.update_layout(title_text = fig13_title,
                        yaxis_title_text = 'Number of Occurrences in Corpus',
                        xaxis_tickangle = -45,
                        bargap = 0.1)
            st.plotly_chart(fig13)

            # Histogram of tokens appearing in < 100 documents
            fig14a = go.Figure(data = go.Histogram(x = list(docs_less_than_100.values()),
                                       marker_color = '#d62728',
                                       opacity = 0.75))
            fig14a.update_layout(title_text = 'Histogram of Tokens Appearing In < 100 Documents',
                   xaxis_title_text = 'Number of Documents Token Appears In',
                   yaxis_title_text = 'Number of Tokens',
                   bargap = 0.1)
            st.plotly_chart(fig14a)

            # Histogram of tokens appearing in 100+ documents
            fig14b = go.Figure(data = go.Histogram(x = list(docs_more.values()),
                                       marker_color = '#d62728',
                                       opacity = 0.75))
            fig14b.update_layout(title_text = 'Histogram of Tokens Appearing In 100+ Documents',
                   xaxis_title_text = 'Number of Documents Token Appears In',
                   yaxis_title_text = 'Number of Tokens',
                   bargap = 0.1)
            st.plotly_chart(fig14b)

            # Slider for top nmin to nmax tokens
            min_val_doc = st.text_input('Min. Ranked Token (1 to 52037)', 1)
            max_val_doc = st.text_input('Max. Ranked Token (1 to 52037)', 300)
            values_doc = st.slider("min to max", int(min_val_doc), int(max_val_doc), (int(min_val_doc), int(int(max_val_doc)/6)))
            top_n_doc = sorted(doc_counts, key=doc_counts.get, reverse=True)[values_doc[0]-1:values_doc[1]-1]
            fig15_title = 'Top ' + str(values_doc[0]) + ' to ' + str(values_doc[1]) + ' Tokens Appearing in Most Number of Documents'

            # Top nmin to nmax tokens in corpus
            fig15 = go.Figure(data = go.Bar(x = top_n_doc,
                                    y = [doc_counts[word] for word in top_n_doc],
                                    marker_color = '#d62728',
                                    opacity = 0.75))
            fig15.update_layout(title_text = fig15_title,
                        yaxis_title_text = 'Number of Documents Token Appears In',
                        xaxis_tickangle = -45,
                        bargap = 0.1)
            st.plotly_chart(fig15)

        elif feat == 'Temporal':
            st.plotly_chart(fig3) # duration
            st.plotly_chart(fig4) # date recorded
            st.plotly_chart(fig5) # date uploaded
            st.plotly_chart(fig6) # upload lag

    elif eda == 'Talks By Year':
        year_data_type = st.sidebar.selectbox('Year Recorded',
                                        ('Audience Engagement',
                                         'Linguistic',
                                         'Temporal'))
        # dict_keys(['lemmas', 'lemma_freq', 'lemma_unique', 'doc_lemmas', 'lemma_num_docs'])
        year_of_interest = st.text_input('Year Recorded', 2019)
        year_str = [str(year) for year in sorted(tok_year_corpus.keys())]
        st.write('Years available: ' + ', '.join(year_str))

        # Add data for given year
        year_data = tok_year_corpus[int(year_of_interest)]['lemma_freq']
        year_tok_range = st.slider("min to max", 1, 300, (1, 50))
        year_data_sort = sorted(year_data, key = year_data.get, reverse = True)[year_tok_range[0]-1:year_tok_range[1]-1]

        # Histogram of
        fig16 = go.Figure(data = go.Bar(x = year_data_sort,
                                        y = [year_data[x] for x in year_data_sort]))
        fig16.update_layout(title_text = 'Test ',
                    yaxis_title_text = 'Number of Occurrences',
                    xaxis_tickangle = -45,
                    bargap = 0.1)
        st.plotly_chart(fig16)

if page == 'Topic Modeling':
    st.write('Topic modeling to be updated')
    st.write(lda_top_words)

if page == 'TED Talk Recommender':
    st.write('Front end for recommendation system')
