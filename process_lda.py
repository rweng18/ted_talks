# --------------------------------------------------------------------------- #

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #

# Returns dataframe of top n words in an LDA model
# given feature names from vectorizer
def topic_top_words(model, feature_names, n):
    topic_word_dict = {}

    for topic_idx, topic in enumerate(model.components_):

        topic_word_dict['Topic ' + str(topic_idx + 1)] = [feature_names[i] for i in topic.argsort()[:-n - 1:-1]]

    return pd.DataFrame.from_dict(topic_word_dict, orient='index', columns = range(1, n + 1))


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
# dtm.head(100).style.applymap(color_green).applymap(make_bold)
def color_green(val):
    color = 'green' if val > .1 else 'black'
    return 'color: {col}'.format(col=color)

def make_bold(val):
    weight = 700 if val > .1 else 400
    return 'font-weight: {weight}'.format(weight=weight)
