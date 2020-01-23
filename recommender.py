# ---------------------------------------------------------------------------- #
# IMPORT PACKAGES
# ---------------------------------------------------------------------------- #

import numpy as np
from scipy.stats import entropy
from process_lda import show_topic_distr
import random

# ---------------------------------------------------------------------------- #
# FUNCTIONS FOR IMPLEMENTING RECOMMENDER
# ---------------------------------------------------------------------------- #

# Calculate Jensen_Shannon Similarity
def jensen_shannon(query, matrix):
    """
    This function implements a Jensen-Shannon similarity
    between the input query (an LDA topic distribution for a document)
    and the entire corpus of topic distributions.
    It returns an array of length M where M is the number of documents in the corpus
    """
    p = query[None,:].T # take transpose
    q = matrix.T # transpose matrix
    m = 0.5*(p + q)
    return np.sqrt(0.5*(entropy(p,m) + entropy(q,m)))

# Get k most similar documents
def get_most_similar_documents(query, matrix, k=10):
    """
    This function implements the Jensen-Shannon distance above
    and retruns the top k indices of the smallest jensen shannon distances
    """
    sims = jensen_shannon(query,matrix) # list of jensen shannon distances
    return sims.argsort()[1:k+1] # the top k positional index of the smallest Jensen Shannon distances

# Get k most different documents
def get_most_diff_documents(query,matrix,k=10):
    """
    This function implements the Jensen-Shannon distance above
    and retruns the top k indices of the smallest jensen shannon distances
    """
    sims = jensen_shannon(query,matrix) # list of jensen shannon distances
    return sims.argsort()[-k-1:-1] # the top k positional index of the largest Jensen Shannon distances

# Get n most similar and most different talks for random talk
def get_rec_random(matrix, all_lda_output, final_data, n):

    # Get random index
    index = random.randint(0, len(all_lda_output))

    query = matrix.iloc[index]

    print('Getting recommendations for talk #' + str(index))

    # Get figure showing topic distribution for talk in question
    rand_topic_distr, summ, tags = show_topic_distr(final_data, all_lda_output, index)

    # rand_topic_distr.show()

    # Get most similar and most different talks based on jensen-shannon distance
    most_sim = get_most_similar_documents(query, matrix, k = n)
    most_dif = get_most_diff_documents(query, matrix, k = n)

    return index, rand_topic_distr, summ, tags, most_sim, most_dif

# Get n most similar and most different talks for random talk
def get_rec_title(matrix, all_lda_output, final_data, title, n):

    # Check if title exists in title
    if title not in list(final_data.title):
        return 'No talk found.'

    else:

        # Find index of first TED talk whose title matches
        index = final_data[final_data.title == title].index.tolist()[0]
        query = matrix.iloc[index]

        print('Getting recommendations for talk #' + str(index))

        # Get figure showing topic distribution for talk in question
        topic_distr, summ, tags = show_topic_distr(final_data, all_lda_output, index)

        # rand_topic_distr.show()

        # Get most similar and most different talks based on jensen-shannon distance
        most_sim = get_most_similar_documents(query, matrix, k = n)
        most_dif = get_most_diff_documents(query, matrix, k = n)

        return index, topic_distr, summ, tags, most_sim, most_dif
