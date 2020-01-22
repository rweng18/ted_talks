# --------------------------------------------------------------------------- #
import numpy as np
import pandas as pd
import plotly.offline as py
import plotly.graph_objects as go
# --------------------------------------------------------------------------- #

# Returns dataframe of top n words in an LDA model
# given feature names from vectorizer
def topic_top_words(model, feature_names, n):
    topic_word_dict = {}

    for topic_idx, topic in enumerate(model.components_):

        topic_word_dict['Topic ' + str(topic_idx + 1)] = [feature_names[i] for i in topic.argsort()[:-n - 1:-1]]

    return pd.DataFrame.from_dict(topic_word_dict, orient='index', columns = range(1, n + 1))

# Figure of topic distribution for given index
def show_topic_distr(talk_df, lda_dtm, index):
    talk_title = talk_df.iloc[index]['title']
    if pd.isnull(talk_df.iloc[index]['date_recorded']):
        talk_date = 'Unknown'
    else:
        talk_date = pd.to_datetime(talk_df.iloc[index]['date_recorded'])
    fig_title = talk_title + ' (' + talk_date.strftime('%b') + ' ' + str(talk_date.year) + ')'

    topic_distr = go.Figure(data = go.Bar(x = np.arange(0,15),
                                          y = lda_dtm.iloc[index]['01_general':'dominant_topic'],
                                          marker_color = '#d62728',
                                          opacity = 0.75))
    topic_distr.update_layout(title_text = fig_title,
                              yaxis_title_text = 'Proportion of Talk',
                              xaxis = dict(tickmode = 'array',
                                           tickvals = np.arange(0,15,1),
                                           ticktext = ['General', 'Science', 'Tech', 'Politics', 'Problems', 'Personal',
                                                       'AI', 'Miscellaneous', 'Healthcare', 'Linguistics/Humanities', 'Space',
                                                       'Agriculture/Nature', 'Gender/Sexuality', 'Audio/Visual', 'Urban Planning/Design'],
                                           tickangle = -45))
    topic_distr.update_yaxes(range=[0, 0.75])

    return topic_distr, talk_df.iloc[index]['summ'], talk_df.iloc[index]['tags']

# Return document topic matrix of topic model
def print_dtm(topic_model, dtm):
    # Create Document - Topic Matrix
    output = topic_model.transform(dtm)

    # column names
    topicnames = ["Topic" + str(i) for i in range(topic_model.n_components)]

    # index names
    docnames = ["Doc" + str(i) for i in range(dtm.shape[0])]

    # Make the pandas dataframe
    df_document_topic = pd.DataFrame(output, columns=topicnames, index=docnames)

    # Get dominant topic for each document
    dominant_topic = np.argmax(df_document_topic.values, axis=1)

    # Get next dominant topic for each document
    sorted_topics = np.argsort(df_document_topic.values, axis = 1)
    secondary_topic = [doc[-2] for doc in sorted_topics]

    # Get third dominant topic for each document
    tertiary_topic = [doc[-3] for doc in sorted_topics]

    df_document_topic['dominant_topic'] = dominant_topic
    df_document_topic['secondary_topic'] = secondary_topic
    df_document_topic['tertiary_topic'] = tertiary_topic

    return df_document_topic

# Helper functions to style output of print_dtm
# dtm.head(100).style.applymap(color_red).applymap(make_bold)
def color_red(val):
    color = 'red' if val > .1 else 'black'
    return 'color: {col}'.format(col=color)

def make_bold(val):
    weight = 700 if val > .1 else 400
    return 'font-weight: {weight}'.format(weight=weight)
