# ---------------------------------------------------------------------------- #
# IMPORT PACKAGES
# ---------------------------------------------------------------------------- #

import streamlit as st

from collections import Counter
import numpy as np
import pandas as pd
import pickle

# Plotting Package
import plotly.graph_objects as go

# Import custom functions
from process_lda import show_topic_distr
from recommender import get_rec_random, get_rec_title

# Import figures
import figures

# ---------------------------------------------------------------------------- #
st.title('Topic Modeling TED Talk Transcripts')
st.subheader('Rebecca Weng | Flatiron School Data Science Immersive | Jan. 2020')
# ---------------------------------------------------------------------------- #
# READ IN DATA
# ---------------------------------------------------------------------------- #

# Load final dataset
with open('Data/final_raw_data.pkl', 'rb') as file:
    talk_df = pickle.load(file)

# Load in tokenized data
with open('Data/all_tok.pkl', 'rb') as file:
    tok_corpus = pickle.load(file)

# Create dictionary of token counts for entire tok_corpus
word_bank = dict(Counter(tok_corpus))
words_less_than_100 = {key: value for key, value in word_bank.items() if value < 100}
words_more = {key: value for key, value in word_bank.items() if value >= 100}

# Load in number of documents tokens appear in
with open('Data/doc_tok_counts.pkl', 'rb') as file:
    doc_counts = pickle.load(file)
docs_less_than_100 = {key: value for key, value in doc_counts.items() if value < 100}
docs_more = {key: value for key, value in doc_counts.items() if value >= 100}

# Load in tokens by year
with open('Data/year_tok.pkl', 'rb') as file:
    tok_year_corpus = pickle.load(file)

# Load in data on final LDA model
with open('Models/final_lda_words.pkl', 'rb') as file:
    top_15_words = pickle.load(file)

# Load final LDA document-topic matrix
with open('Models/final_lda_dtm.pkl', 'rb') as file:
    final_lda_dtm = pickle.load(file)

matrix = final_lda_dtm[['01_general', '02_science', '03_technology',
                        '04_politics', '05_problems', '06_personal',
                        '07_AI', '08_miscellaneous', '09_healthcare',
                        '10_linguistics/humanities', '11_space',
                        '12_agriculture/nature', '13_gender/sexuality',
                        '14_audio/visual', '15_urban_planning/design']]

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
    st.header('PROJECT GOALS')
    st.write('1. Explaratory analysis of TED talk features.')
    st.write('2. Use natural language processing and topic modeling to understand latent groupings of TED talks based on transcripts.')
    st.write('3. Create a content-based recommender to showcase a simple application of topic modeling.')
    st.write('4. Create an interactive interface to explore the data and findings. All the figures are interactive!')

    st.header('DATA OVERVIEW')
    st.markdown('* 4,200+ talks online')
    st.markdown('* 3,600+ talks with transcripts')
    st.markdown('* 52,000+ unique tokens')
    st.markdown('* 15,000+ tokens used for modeling')
    st.markdown('* Recorded from 1984 to present, uploaded from 2006 to present')
    st.markdown('* TED assigns 5+ tags to most talks')
    st.write(talk_df) # Print dataframe

    st.header('PROJECT PIPELINE')
    st.markdown('1. Tokenize transcripts')
    st.markdown('2. Vectorize transcripts')
    st.markdown('3. Use Latent Dirichlet Allocation (LDA) for topic modeling')
    st.markdown('4. Use Jensen-Shannon divergence to recommend most similar and dissimilar talks')

    st.header('INSIGHTS & TAKEAWAYS')
    st.markdown('* TED talks use a lot of general language to communicate niche topics')
    st.markdown('* Around the early 2000\'s you get a sense of TED\'s image')
    st.markdown('* It is incredibly important to consider human-readability when it comes to natural language processing because of our intuitive ability to determine whether words should or should not be grouped together, or whether two talks are similar or different')
    st.markdown('* There is a lot more that can be done with this data!')

if page == 'Exploratory Data Analysis':

    # Select level of EDA, all talks or talks by year
    eda = st.sidebar.selectbox('Corpus Level', ('All Talks', 'Talks By Year'))

    if eda == 'All Talks':
        feat = st.sidebar.selectbox('Features to Explore',
                                   ('Audience Engagement',
                                    'Linguistic',
                                    'Temporal',
                                    'Token Occurrences in Corpus',
                                    'Token Occurrences By Number of Documents'
                                    ))

        st.header('CORPUS-LEVEL DATA')
        st.markdown('* This data was generated by aggregating all of the talks\' information')
        st.markdown('* $n$ indicates number of talks to which each feature had recorded or engineered values')
        st.markdown('* For variables relating to token occurrences, data has been aggregated from all the talks, irrespective of other features')

        if feat == 'Audience Engagement':
            st.plotly_chart(figures.views)     # views
            st.plotly_chart(figures.views_log) # log(views)
            st.plotly_chart(figures.comm)      # comments

        elif feat == 'Linguistic':
            st.plotly_chart(figures.word_count) # transcript word count
            st.plotly_chart(figures.tag_len)    # number of distinct tags per talk
            st.plotly_chart(figures.doc_tok)    # distinct tokens

        elif feat == 'Temporal':
            st.plotly_chart(figures.dur) # duration
            st.plotly_chart(figures.recorded) # date recorded
            st.plotly_chart(figures.uploaded) # date uploaded
            st.plotly_chart(figures.lag) # upload lag

        elif feat == 'Token Occurrences in Corpus':
            st.plotly_chart(figures.corp_occur_a) # tokens in corpus < 100 times
            st.plotly_chart(figures.corp_occur_b) # tokens in corpus 100+ times

            # Slider for top nmin to nmax tokens
            min_val = st.text_input('Minimum Ranked Token (1 to 52037)', 1)
            max_val = st.text_input('Maximum Ranked Token (1 to 52037)', 300)
            values = st.slider("nmin to nmax", int(min_val), int(max_val), (int(min_val), int(int(max_val)/6)))
            top_n_corpus = sorted(word_bank, key=word_bank.get, reverse=True)[values[0]-1:values[1]-1]
            fig13_title = f'Top {values[0]} to {values[1]} Tokens in Corpus'

            # Plot top nmin to nmax tokens in corpus
            fig13 = go.Figure(data = go.Bar(x = top_n_corpus,
                                            y = [word_bank[word] for word in top_n_corpus],
                                            marker_color = '#d62728',
                                            opacity = 0.75))
            fig13.update_layout(title_text = fig13_title,
                                yaxis_title_text = 'Number of Occurrences in Corpus',
                                xaxis_tickangle = -45,
                                bargap = 0.1)
            st.plotly_chart(fig13)

        elif feat == 'Token Occurrences By Number of Documents':
            st.plotly_chart(figures.doc_occur_a) # token hist < 100 docs
            st.plotly_chart(figures.doc_occur_b) # token hist 100+ docs

            # Slider for top nmin to nmax tokens
            min_val_doc = st.text_input('Min. Ranked Token (1 to 52037)', 1)
            max_val_doc = st.text_input('Max. Ranked Token (1 to 52037)', 300)
            values_doc = st.slider("min to max", int(min_val_doc), int(max_val_doc), (int(min_val_doc), int(int(max_val_doc)/6)))
            top_n_doc = sorted(doc_counts, key=doc_counts.get, reverse=True)[values_doc[0]-1:values_doc[1]-1]
            fig15_title = f'Top {values_doc[0]} to {values_doc[1]} Tokens Appearing in Most Number of Documents'

            # Plot top nmin to nmax tokens based on number of documents tokens
            # appeared in
            fig15 = go.Figure(data = go.Bar(x = top_n_doc,
                                            y = [doc_counts[word] for word in top_n_doc],
                                            marker_color = '#d62728',
                                            opacity = 0.75))
            fig15.update_layout(title_text = fig15_title,
                                yaxis_title_text = 'Number of Documents Token Appears In',
                                xaxis_tickangle = -45,
                                bargap = 0.1)
            st.plotly_chart(fig15)


    elif eda == 'Talks By Year':
        st.header('EXPLORE TED TALK TOKENS BY YEAR RECORDED')
        st.markdown('* Compare up to 2 years of data')

        # Get all possible years in which TED talk was recorded
        year_str = [str(year) for year in sorted(tok_year_corpus.keys())]

        # Write options for years for user to see
        st.subheader('Years Available:')
        st.write(', '.join(year_str[0:int(len(year_str)/4)]))
        st.write(', '.join(year_str[int(len(year_str)/4):int(2*len(year_str)/4)]))
        st.write(', '.join(year_str[int(2*len(year_str)/4):int(3*len(year_str)/4)]))
        st.write(', '.join(year_str[int(3*len(year_str)/4):len(year_str)]))

        # User inputs year of interest
        year_of_interest_a = st.text_input('Year Recorded (1)', 2019)

        # Condition if inputted year not available
        if year_of_interest_a not in year_str:
            st.write(f'Sorry, {year_of_interest_a} not available.')

        # Add data for given year
        year_data_a = tok_year_corpus[int(year_of_interest_a)]['lemma_freq']

        # User can toggle range of tokens to search between, then range to visualize
        tok_range_a = st.slider("Range of Tokens (1)", 1, len(year_data_a), (1, 200))
        year_tok_range_a = st.slider("Min to Max (1)", tok_range_a[0], tok_range_a[1], (1, 50))

        # Sort data in range for visualization
        year_data_sort_a = sorted(year_data_a, key = year_data_a.get, reverse = True)[year_tok_range_a[0]-1:year_tok_range_a[1]-1]

        # Histogram of top tokens in talks recorded in given year
        fig16a = go.Figure(data = go.Bar(x = year_data_sort_a,
                                         y = [year_data_a[x] for x in year_data_sort_a]))
        fig16a_title = f'Top {year_tok_range_a[0]} to {year_tok_range_a[1]} Tokens in Talks Recorded in {year_of_interest_a}'
        fig16a.update_layout(title_text = fig16a_title,
                             yaxis_title_text = 'Number of Occurrences',
                             xaxis_tickangle = -45,
                             bargap = 0.1)
        st.plotly_chart(fig16a)

        # User inputs year of interest
        year_of_interest_b = st.text_input('Year Recorded (2)', 2019)

        # Condition if inputted year not available
        if year_of_interest_b not in year_str:
            st.write(f'Sorry, {year_of_interest_b} not available.')

        # Add data for given year
        year_data_b = tok_year_corpus[int(year_of_interest_b)]['lemma_freq']

        # User can toggle range of tokens to search between, then range to visualize
        tok_range_b = st.slider("Range of Tokens (2)", 1, len(year_data_b), (1, 200))
        year_tok_range_b = st.slider("Min to Max (2)", tok_range_b[0], tok_range_b[1], (1, 50))

        # Sort data in range for visualization
        year_data_sort_b = sorted(year_data_b, key = year_data_b.get, reverse = True)[year_tok_range_b[0]-1:year_tok_range_b[1]-1]

        # Histogram of top tokens in talks recorded in given year
        fig16b = go.Figure(data = go.Bar(x = year_data_sort_b,
                                         y = [year_data_b[x] for x in year_data_sort_b]))
        fig16b_title = f'Top {year_tok_range_b[0]} to {year_tok_range_b[1]} Tokens in Talks Recorded in {year_of_interest_b}'
        fig16b.update_layout(title_text = fig16b_title,
                             yaxis_title_text = 'Number of Occurrences',
                             xaxis_tickangle = -45,
                             bargap = 0.1)
        st.plotly_chart(fig16b)

if page == 'Topic Modeling':

    tm = st.sidebar.selectbox('Topic Modeling Section', ('Overview', 'Topic Distribution'))

    if tm == 'Overview':
        st.header('LATENT DIRICHLET ALLOCATION')
        st.markdown('* Assumes that documents are made up of a distribution of topics')
        st.markdown('* Assumes that topics are made up of a distribution of words')
        st.markdown('* Assumes that there are k topics, which you distribute across a document by assigning each word w to a topic')
        st.markdown('* For each word w in a document, assume topic is wrong but all the other words in teh document were assigned the correct topic')
        st.markdown('* Probabilistically assign word w to a topic based on 1) topics in the document, 2) how many times the word was assigned a topic across all documents')
        st.markdown('* Repeat!')
        st.markdown('* Evaluate using log-likelihood and perplexity, but most important for this project was human-readability')

        st.header('LDA TOPICS')
        st.markdown('* These are the top 15 words in each topic that my final model generated')
        st.markdown('* The leftmost column shows the labels I assigned the topics')
        st.write(top_15_words) # top 15 words in topics

        # st.markdown('[CLICK HERE FOR SEPARATE VISUAL!](file:///Users/rweng/Desktop/Flatiron/Projects/Final_20190124/final_lda.html)', unsafe_html = True)
        st.header('MOST PREVALENT TOPICS')
        st.markdown('* The first figure visualize the distribution of topics based on number of times they were the most prevalent (dominant topic), second-most prevalent (secondary topic), and third-most prevalent (tertiary topic)')
        st.markdown('* The following figures breakdown the frequencies that the dominant, secondary, and tertiary topics co-occurred in the same talks')
        st.markdown('* For the figures on co-occurrences, I suggest double-clicking on your topic of choice in the legend, or clicking once to remove that trace')
        st.plotly_chart(figures.lda_topics) # histogram of top 3 topics
        st.plotly_chart(figures.topic_cooc_12) # co-occurrences of dom and sec topics
        st.plotly_chart(figures.topic_cooc_13) # co-occurrences of dom and ter topics
        st.plotly_chart(figures.topic_cooc_23) # co-occurrences of sec and ter topics
    else:
        talk_index = st.text_input('Talk Index (1 to 3646)', 1)
        tm_test, summ_, tags_ = show_topic_distr(talk_df, final_lda_dtm, int(talk_index)-1)
        st.plotly_chart(tm_test)
        st.subheader('SUMMARY:')
        st.write(summ_)
        st.subheader('CURRENT TED TAGS:')
        tag_str = ', '.join(tags_)
        st.write(tag_str)

if page == 'TED Talk Recommender':

    rec = st.sidebar.selectbox('Recommender Section', ('By Title', 'Random'))

    st.header('TED TALK RECOMMENDER')
    st.markdown('* This serves as both an application and evaluation metric of the LDA topic modeling')
    st.markdown('* The Jensen-Shannon divergence determines how different two probability distributions are, based on the Kullback-Leibler divergence')
    st.markdown('* The recommender takes in the document-topic matrix generated by the LDA model, determines the Jensen-Shannon divergence between the talk of interest and all the talks, then sorts it, and returns the most simiilar and most dissimilar talks')
    st.markdown('* The dropdown bar also allows you to type so if a word is in the title, it will come up!')

    if rec == 'By Title':
        title = st.selectbox('Talk Title', tuple(talk_df.title))

        if title not in list(talk_df.title):
            st.write('TALK NOT FOUND')

        else:
            index, topic_distr, summ, tags, most_sim, most_dif = get_rec_title(matrix, final_lda_dtm, talk_df, title, 5)
            st.plotly_chart(topic_distr)
            st.subheader('SUMMARY:')
            st.write(summ)
            url = 'https://www.ted.com' + talk_df.iloc[index]['url']
            st.markdown(f'[WATCH THE TALK HERE]({url})')
            st.subheader('CURRENT TED TAGS:')
            tag_str_ = ', '.join(tags)
            st.write(tag_str_)

            st.write('--------------------------------------------------------')

            st.subheader('MOST SIMILAR TALKS:')
            for talk in most_sim:
                title = talk_df.iloc[talk]['title']
                url = 'https://www.ted.com' + talk_df.iloc[talk]['url']
                st.markdown(f'[{title}]({url})')
                st.write(talk_df.iloc[talk]['summ'])

            st.write('--------------------------------------------------------')

            st.subheader('MOST DIFFERENT TALKS:')
            for talk in most_dif:
                title = talk_df.iloc[talk]['title']
                url = 'https://www.ted.com' + talk_df.iloc[talk]['url']
                st.markdown(f'[{title}]({url})')
                st.write(talk_df.iloc[talk]['summ'])

    elif rec == 'Random':
        index, rand_topic_distr, summ, tags, most_sim, most_dif = get_rec_random(matrix, final_lda_dtm, talk_df, 5)
        st.plotly_chart(rand_topic_distr)
        st.subheader('SUMMARY:')
        st.write(summ)
        st.subheader('CURRENT TED TAGS:')
        tag_str_ = ', '.join(tags)
        st.write(tag_str_)

        st.write('--------------------------------------------------------')

        st.subheader('MOST SIMILAR TALKS:')
        for talk in most_sim:
            title = talk_df.iloc[talk]['title']
            url = 'https://www.ted.com' + talk_df.iloc[talk]['url']
            st.markdown(f'[{title}]({url})')
            st.write(talk_df.iloc[talk]['summ'])

        st.write('--------------------------------------------------------')

        st.subheader('MOST DIFFERENT TALKS:')
        for talk in most_dif:
            title = talk_df.iloc[talk]['title']
            url = 'https://www.ted.com' + talk_df.iloc[talk]['url']
            st.markdown(f'[{title}]({url})')
            st.write(talk_df.iloc[talk]['summ'])
