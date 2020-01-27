# Topic Modeling TED Talks
## Project Goals
1. Explore natural language processing techniques by creating a topic model using TED talk transcripts
2. Develop an application for topic modeling by creating a simple TED talk recommender
3. Develop an interactive front end to showcase the data, exploratory data analysis, topic modeling, and recommender

## Project Overview
### Part I: Topic Modeling and Natural Language Processing
TED talks are currently categorized under hundreds of topics. In fact, on the TED website itself, a wide range of topics are listed [here](https://www.ted.com/topics), from niche topics like "biomimicry" to general ideas like "big problems." This project began by using natural language processing and unsupervised learning to create a smaller set of topics with which to categorize Ted Talks.

### Part II: Recommendation System
Next, I created and deployed a simple recommendation systems using the topics modeled above to recommend TED talks based on  Jensen-Shannon divergence. Currently, you can input a title or an index and the system will generate the topic distribution, summary, link, and existing tags provided by TED for the talk, along with the 5 most similar and 5 most dissimilar talks.

### Part III: Interactive Front End
Finally, I used Streamlit to develop an interactive interface to view exploratory data analysis and deliverables of project. I then used Heroku to deploy the app here: [TED Talk Project](https://tedrecommender.herokuapp.com/). The user can use the sidebar to explore the algorithms used, exploratory data analysis, and customize some visualizations as well. The user can also generate talk topic distributions and TED talk recommendations based on the talk title or random index.

## Data & Methods
* Information about 4,200+ TED talks were web scraped from the official TED website. The dataset only includes the TED talks available from the quicklist of all TED talks [here](https://www.ted.com/talks/quick-list?page=1).
    * Topic modeling focused on the 3,600+ TED talks with transcripts
* Created a custom tokenizer (see tokenizer.py) utilizing NLTK and SpaCy packages.
    * Added missing spaces after punctuation
    * Removed parenthetical phrases about "applause," "laughter," and "music"
    * Handled cases with numbers in hyphened words and with commas
    * Removed punctuation, stop words, and lower-cased all tokens
    * Exclusively looked at unigrams
* After applying the tokenizer to the 3,600+ TED talks, had 2,300,000+ tokens (52,000+ unique tokens).
* Used a Count Vectorizer to exclude tokens that occurred in more than 70% of the transcripts, and in fewer than 4 transcripts.
    * 15,000+ unique tokens (15,000+ features) used in the final model
* After vectorizing the tokenized transcripts, input data into Latent Dirichlet Allocation (LDA)
    * Tested 10, 13, 15, 16, 17, 20, and 25 topics
    * Evaluated using 80:20 train-test-split, log-likelihood, perplexity, and human readability, with an emphasis on the latter
    * Final model used 15 topics
* Transformed entire data set using LDA model that was fit on training data
* Used resulting document-topic matrix as the input to calculate Jensen-Shannon divergence, which takes in probability distributions, to calculate similarity between TED talks

## Deliverables & Insights


## Future Work
