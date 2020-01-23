# ---------------------------------------------------------------------------- #
# IMPORT PACKAGES
# ---------------------------------------------------------------------------- #

from collections import Counter
import numpy as np
import pandas as pd
import pickle
import plotly.offline as py
import plotly.graph_objects as go

# ---------------------------------------------------------------------------- #
# LOAD DATA
# ---------------------------------------------------------------------------- #

# Load final dataset
with open('Data/final_raw_data.pkl', 'rb') as file:
    talk_df = pickle.load(file)

# Load in tokenized documents
with open('final_tok.pkl', 'rb') as file:
    tok_doc = pickle.load(file)

# Load in corpus of tokens
with open('Data/all_tok.pkl', 'rb') as file:
    tok_corpus = pickle.load(file)

# Create dictionary of token counts for entire tok_corpus
word_bank = dict(Counter(tok_corpus))
words_less = {key: value for key, value in word_bank.items() if value < 100}
words_more = {key: value for key, value in word_bank.items() if value >= 100}

# Load in number of documents tokens appear in
with open('Data/doc_tok_counts.pkl', 'rb') as file:
    doc_counts = pickle.load(file)

# Create dictionary of token counts by appearances in documents
docs_less = {key: value for key, value in doc_counts.items() if value < 100}
docs_more = {key: value for key, value in doc_counts.items() if value >= 100}

# Load final LDA document-topic matrix
with open('Models/final_lda_dtm.pkl', 'rb') as file:
    all_lda_output = pickle.load(file)

all_top_topics = all_lda_output[['dominant_topic', 'secondary_topic',
                                 'tertiary_topic']]

all_top_topics = all_top_topics.replace({0: 'General', 1: 'Science', 2: 'Tech',
                                         3: 'Politics', 4: 'Problems', 5: 'Personal',
                                         6: 'AI', 7: 'Miscellaneous', 8: 'Healthcare',
                                         9: 'Linguistics/Humanities', 10: 'Space', 11: 'Agriculture/Nature',
                                         12: 'Gender/Sexuality', 13: 'Audio/Visual', 14: 'Urban Planning/Design'})

# ---------------------------------------------------------------------------- #
# CREATE FIGURES FOR FRONTEND AND EDA
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
# AUDIENCE ENGAGEMENT
# ---------------------------------------------------------------------------- #

# Histogram of View Count
views = go.Figure(data=[go.Histogram(x = talk_df.views,
                                   marker_color = '#d62728',
                                   opacity = 0.75)])
views_title = f'Histogram of TED Talk View Counts (n = {len(talk_df.views)})'
views.update_layout(title_text = views_title,
                  xaxis_title_text = 'View Count',
                  yaxis_title_text = 'Number of TED Talks',
                  bargap = 0.1)

# Histogram of Log(View Count)
views_log_data = [x for x in talk_df.views if x != 0]
views_log_title = f'Histogram of Log(TED Talk View Counts) (n = {len(views_log_data)})'

views_log = go.Figure(data=[go.Histogram(x = np.log(views_log_data),
                                         marker_color = '#d62728',
                                         opacity = 0.75)])
views_log.update_layout(title_text = views_log_title,
                        xaxis_title_text = 'Log(View Count)',
                        yaxis_title_text = 'Number of TED Talks',
                        bargap = 0.1)

# Histogram of Comments
comm = go.Figure(data = go.Histogram(x = talk_df.comments,
                                     marker_color = '#d62728',
                                     opacity = 0.75))
comm_title = f'Histogram of Comments (n = {sum(talk_df.comments.notnull())})'
comm.update_layout(title_text = comm_title,
                   xaxis_title_text = 'Number of Comments',
                   yaxis_title_text = 'Number of TED Talks',
                   bargap = 0.1)

# ---------------------------------------------------------------------------- #
# LINGUISTIC
# ---------------------------------------------------------------------------- #

# Histogram of Transcript Word Count
word_count = go.Figure(data = go.Histogram(x = talk_df.transcript_wc,
                                           marker_color = '#d62728',
                                           opacity = 0.75))
word_count_title = f'Histogram of Transcript Word Count (n = {sum(talk_df.transcript_wc.notnull())})'
word_count.update_layout(title_text = 'Histogram of Transcript Word Count (n = 3646)',
                         xaxis_title_text = 'Transcript Word Count',
                         yaxis_title_text = 'Number of TED Talks',
                         bargap = 0.1)

# Histogram of Tag Length
tag_len = go.Figure(data = go.Histogram(x = talk_df.tag_len,
                                        marker_color = '#d62728',
                                        opacity = 0.75))
tag_len_title = f'Histogram of Number of TED Assigned Tags (n = {sum(talk_df.tag_len.notnull())})'
tag_len.update_layout(title_text = tag_len_title,
                      xaxis_title_text = 'Number of Tags',
                      yaxis_title_text = 'Number of TED Talks',
                      bargap = 0.1)

# Histogram of Number of Distinct Tokens
doc_tok_len = [len(set(doc_tok)) for doc_tok in tok_doc]
doc_tok = go.Figure(data = go.Histogram(x = doc_tok_len,
                                      marker_color = '#d62728',
                                      opacity = 0.75))
doc_tok_title = f'Histogram of Distinct Tokens (n = {len(doc_tok_len)})'
doc_tok.update_layout(title_text = doc_tok_title,
                      xaxis_title_text = 'Number of Distinct Tokens',
                      yaxis_title_text = 'Number of TED Talks',
                      bargap = 0.1)

# ---------------------------------------------------------------------------- #
# TEMPORAL
# ---------------------------------------------------------------------------- #

# Histogram of Talk Duration (Seconds)
dur = go.Figure(data = go.Histogram(x = [dur/60 for dur in talk_df.duration],
                                    xbins=dict(start=0,
                                    end=max([dur/60 for dur in talk_df.duration]),
                                    size=1),
                                    marker_color = '#d62728',
                                    opacity = 0.75))
dur_title = f'Histogram of Talk Duration (n = {len(talk_df.duration)})'
dur.update_layout(title_text = dur_title,
                  xaxis_title_text = 'Duration (Minutes)',
                  yaxis_title_text = 'Number of TED Talks',
                  bargap = 0.1)

# Histogram of Date Recorded
recorded = go.Figure(data = go.Histogram(x = talk_df[talk_df.date_recorded.notnull()].date_recorded,
                                         marker_color = '#d62728',
                                         opacity = 0.75))
recorded_title = f'Histogram of Date Recorded (n = {len(talk_df[talk_df.date_recorded.notnull()])})'
recorded.update_layout(title_text = recorded_title,
                       xaxis_title_text = 'Date Recorded',
                       yaxis_title_text = 'Number of TED Talks',
                       bargap = 0.1)

# Histogram of Date Uploaded
uploaded = go.Figure(data = go.Histogram(x = talk_df[talk_df.upload_date.notnull()].upload_date,
                                         marker_color = '#d62728',
                                         opacity = 0.75))
uploaded_title = f'Histogram of Date Uploaded (n = {len(talk_df[talk_df.upload_date.notnull()])})'
uploaded.update_layout(title_text = uploaded_title,
                       xaxis_title_text = 'Date Uploaded',
                       yaxis_title_text = 'Number of TED Talks',
                       bargap = 0.1)

# Calculate lag in upload
upload_lag = talk_df[talk_df.upload_date.notnull()].upload_date - talk_df[talk_df.upload_date.notnull()].date_recorded
upload_lag_days = [x.days for x in upload_lag if x.days >= 0]

# Histogram of Upload Lag
lag = go.Figure(data = go.Histogram(x = upload_lag_days,
                                    xbins=dict(start=0, end=max(upload_lag_days), size=90),
                                    marker_color = '#d62728',
                                    opacity = 0.75))
lag_title = f'Histogram of Upload Lag (n = {len(upload_lag_days)})'
lag.update_layout(title_text = lag_title,
                   xaxis_title_text = 'Days Since Recorded',
                   yaxis_title_text = 'Number of TED Talks',
                   bargap = 0.1)

# ---------------------------------------------------------------------------- #
# TOKEN OCCURRENCE (CORPUS)
# ---------------------------------------------------------------------------- #

# Histogram of Token Occurrence in Entire Corpus
corp_occur_a = go.Figure(data = go.Histogram(x = list(words_less.values()),
                                             marker_color = '#d62728',
                                             opacity = 0.75))
corp_occur_a_title = f'Histogram of Tokens Appearing < 100 Times in Corpus (n = {len(words_less)})'
corp_occur_a.update_layout(title_text = corp_occur_a_title,
                           xaxis_title_text = corp_occur_a_title,
                           yaxis_title_text = 'Number of Tokens',
                           bargap = 0.1)

corp_occur_b = go.Figure(data = go.Histogram(x = list(words_more.values()),
                                             xbins=dict(start=0, end=max(list(words_more.values())), size=200),
                                             marker_color = '#d62728',
                                             opacity = 0.75))
corp_occur_b_title = f'Histogram of Tokens Appearing 100+ Times in Corpus (n = {len(words_more)})'
corp_occur_b.update_layout(title_text = corp_occur_b_title,
                           xaxis_title_text = 'Token Frequency in Corpus',
                           yaxis_title_text = 'Number of Tokens',
                           bargap = 0.1)

# ---------------------------------------------------------------------------- #
# TOKEN OCCURRENCE (DOC)
# ---------------------------------------------------------------------------- #

# Histogram of tokens appearing in < 100 documents
doc_occur_a = go.Figure(data = go.Histogram(x = list(docs_less.values()),
                                            marker_color = '#d62728',
                                            opacity = 0.75))
doc_occur_a_title = f'Histogram of Tokens Appearing In < 100 Documents (n = {len(docs_less)})'
doc_occur_a.update_layout(title_text = doc_occur_a_title,
                          xaxis_title_text = 'Number of Documents Token Appears In',
                          yaxis_title_text = 'Number of Tokens',
                          bargap = 0.1)

# Histogram of tokens appearing in 100+ documents
doc_occur_b = go.Figure(data = go.Histogram(x = list(docs_more.values()),
                                       marker_color = '#d62728',
                                       opacity = 0.75))
doc_occur_b_title = f'Histogram of Tokens Appearing In 100+ Documents (n = {len(docs_more)})'
doc_occur_b.update_layout(title_text = doc_occur_b_title,
                     xaxis_title_text = 'Number of Documents Token Appears In',
                     yaxis_title_text = 'Number of Tokens',
                     bargap = 0.1)

# ---------------------------------------------------------------------------- #
# HELPER FUNCTION FOR CO-OCCURRENCES
# ---------------------------------------------------------------------------- #

# Helper function to get co-occurrences
def get_cooccur(topic_type_1, topic_type_2, topic_df):
    cooccur_dict = {}

    for index in range(0, len(topic_df)):

        type_1 = topic_df[topic_type_1][index]
        type_2 = topic_df[topic_type_2][index]

        if type_1 not in cooccur_dict.keys():
            cooccur_dict[type_1] = {'General': 0, 'Science': 0, 'Tech': 0,
                                    'Politics': 0, 'Problems': 0, 'Personal': 0,
                                    'AI': 0, 'Miscellaneous': 0, 'Healthcare': 0,
                                    'Linguistics/Humanities': 0, 'Space': 0, 'Agriculture/Nature': 0,
                                    'Gender/Sexuality': 0, 'Audio/Visual': 0, 'Urban Planning/Design': 0}
        cooccur_dict[type_1][type_2] += 1

    return pd.DataFrame.from_dict(cooccur_dict, orient='index')

# Find co-occurrences of dominant and secondary topics
topics_df_12 = get_cooccur('dominant_topic', 'secondary_topic', all_top_topics)

topics_df_13 = get_cooccur('dominant_topic', 'tertiary_topic', all_top_topics)

topics_df_23 = get_cooccur('secondary_topic', 'tertiary_topic', all_top_topics)

# ---------------------------------------------------------------------------- #
# LDA TOPICS
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
# FIGURES OF TOPIC HISTOGRAMS
# ---------------------------------------------------------------------------- #

# Histogram of top 3 topics
lda_topics = go.Figure()
lda_topics.add_trace(go.Histogram(x = all_lda_output.dominant_topic,
                                  marker_color = '#d62728',
                                  opacity = 0.75,
                                  histnorm = 'probability',
                                  name = 'Dominant Topic'))
lda_topics.add_trace(go.Histogram(x = all_lda_output.secondary_topic,
                                  marker_color = 'mediumblue',
                                  opacity = 0.75,
                                  histnorm = 'probability',
                                  name = 'Secondary Topic'))
lda_topics.add_trace(go.Histogram(x = all_lda_output.tertiary_topic,
                                  opacity = 0.75,
                                  histnorm = 'probability',
                                  name = 'Tertiary Topic'))
lda_topics.update_layout(title_text = 'Histogram of Dominant, Secondary, and Tertiary Topics',
                         xaxis_title_text = '',
                         yaxis_title_text = 'Number of Ted Talks',
                         xaxis = dict(tickmode = 'array',
                                      tickvals = np.arange(0,15,1),
                                      ticktext = ['General', 'Science', 'Tech', 'Politics', 'Problems', 'Personal',
                                                  'AI', 'Miscellaneous', 'Healthcare', 'Linguistics/Humanities', 'Space',
                                                  'Agriculture/Nature', 'Gender/Sexuality', 'Audio/Visual', 'Urban Planning/Design'],
                                      tickangle = -45),
                         bargap = 0.1)

# Plot histogram of co-occurrences of dominant and secondary topics
topic_cooc_12 = go.Figure()
for topic in topics_df_12.index:
    topic_cooc_12.add_trace(go.Bar(x = topics_df_12.columns,
                                   y = topics_df_12.loc[topic],
                                   name = topic))
topic_cooc_12.update_layout(barmode='group', xaxis_tickangle=-45,
                            xaxis_title_text = 'Secondary Topic',
                            yaxis_title_text = 'Number of TED Talks',
                            title = 'Frequency of Co-occurring Dominant and Secondary Topics')

# Plot histogram of co-occurrences of dominant and tertiary topics
topic_cooc_13 = go.Figure()
for topic in topics_df_13.index:
    topic_cooc_13.add_trace(go.Bar(x = topics_df_13.columns,
                                   y = topics_df_13.loc[topic],
                                   name = topic))

# Here we modify the tickangle of the xaxis, resulting in rotated labels.
topic_cooc_13.update_layout(barmode='group', xaxis_tickangle=-45,
                            xaxis_title_text = 'Tertiary Topic',
                            yaxis_title_text = 'Number of Ted Talks',
                            title = 'Frequency of Co-occurring Dominant and Tertiary Topics')

# Plot histogram of co-occurrences of dominant and tertiary topics
topic_cooc_23 = go.Figure()
for topic in topics_df_23.index:
    topic_cooc_23.add_trace(go.Bar(x = topics_df_23.columns,
                                   y = topics_df_23.loc[topic],
                                   name = topic))

# Here we modify the tickangle of the xaxis, resulting in rotated labels.
topic_cooc_23.update_layout(barmode='group', xaxis_tickangle=-45,
                            xaxis_title_text = 'Tertiary Topic',
                            yaxis_title_text = 'Number of Ted Talks',
                            title = 'Frequency of Co-occurring Secondary and Tertiary Topics')
