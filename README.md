# Topic Modeling TED Talks
## Project Goals
1. Explore natural language processing techniques by creating a topic model using TED talk transcripts
2. Develop an application for topic modeling by creating a simple TED talk recommender
3. Develop an interactive front end to showcase the data, exploratory data analysis, topic modeling, and recommender

## Project Overview

* Final presentation hosted on Google Slides [here](https://docs.google.com/presentation/d/1-l7kfdeJ5Y_BKlocZLmCj8We8QADfvC04qNAR0BIL4k/edit?usp=sharing).
* Final deliverable is an interactive app deployed on Heroku, created using Streamlit here [(https://tedrecommender.herokuapp.com/)](https://tedrecommender.herokuapp.com/). The app allows you to explore the data, exploratory data analysis, algorithms, and use the recommender.
      * You can also run locally using `streamlit run interface.py`.

### Part I: Topic Modeling and Natural Language Processing
TED talks are currently categorized under hundreds of topics. In fact, on the TED website itself, a wide range of topics are listed [here](https://www.ted.com/topics), from niche topics like "biomimicry" to general ideas like "big problems." This project began by using natural language processing and unsupervised learning to create a smaller set of topics with which to categorize Ted Talks.

![Histogram of TED Assigned Tags](https://github.com/rweng18/tedtalks/blob/master/EDA_static/fig01_TED_tags_hist.png)

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

![Histogram of Distinct Tokens](https://github.com/rweng18/tedtalks/blob/master/EDA_static/fig02_distinct_tokens_hist.png)

* Used a Count Vectorizer to exclude tokens that occurred in more than 70% of the transcripts, and in fewer than 4 transcripts.
    * 15,000+ unique tokens (15,000+ features) used in the final model
* After vectorizing the tokenized transcripts, input data into Latent Dirichlet Allocation (LDA)
    * Tested 10, 13, 15, 16, 17, 20, and 25 topics
    * Evaluated using 80:20 train-test-split, log-likelihood, perplexity, and human readability
          * Although a low log-likelihood and high perplexity is what many aim for, they have been found at times to result in topics that are counter to human readability. For this project, I focused on the latter.

| Number of Topics | Train Log-Likelihood | Test Log-Likelihood | Train Perplexity | Test Perplexity |
| ---------------- | -------------------- | ------------------- | ---------------- | --------------- |
|               10 |            -1.26e+07 |           -3.27e+06 |          2700.88 |         3764.29 |
|               15 |            -1.26e+07 |           -3.28e+06 |          2722.52 |         3853.54 |
|               20 |            -1.26e+07 |           -3.28e+06 |          2694.63 |         3895.46 |
|               25 |            -1.26e+07 |           -3.29e+06 |          2716.94 |         3940.12 |
          
* Final model used 15 topics
          * The following table shows the label for each topic that I assigned based on the most salient words, along with the 5 most salient words associated with each topic

|    Assigned Topic Name |      #1 Word |      #2 Word |      #3 Word |      #4 Word |      #5 Word |
| ---------------------- | ------------ | ------------ | ------------ | ------------ | ------------ |
|                General |         life |         tell |        brain |        human |         talk |
|                Science |        light |       energy |        earth |       planet |        space |
|             Technology |         cell |        brain |        human |   technology |       system |
|               Politics |    political |   government |        power |       medium |          war |
|               Problems |      country |      percent |       change |      company |      problem |
|               Personal |          day |         tell |        story |         life |         love |
|                     AI |        robot |      machine |     computer |        build |         game |
|          Miscellaneous |       socket |           tk |     amputate |   prosthesis |      amputee |
|             Healthcare |      patient |       cancer |       health |      disease |       doctor |
| Linguistics/Healthcare |     language |         word |         book |        write |         read |
|                  Space |    satellite |       rocket |        orbit |        space |       launch |
|     Agriculture/Nature |         food |          eat |        plant |       farmer |       animal |
|       Gender/Sexuality |        woman |          man |          sex |       female |         male |
|           Audio/Visual |        music |        sound |         play |        voice |         hear |
|  Urban Planning/Design |         city |       design |     building |        build |        place |

* Transformed entire data set using LDA model that was fit on training data

![pyLDA Visualization](https://github.com/rweng18/tedtalks/blob/master/EDA_static/pyLDAvis.gif)

![Histogram of Most Frequent Important Topics](https://github.com/rweng18/tedtalks/blob/master/EDA_static/fig04_topic_freq.png)
 
* Dominant Topic: largest topic in a talk
* Secondary Topic: second largest topic in a talk
* Tertiary Topic: third largest topic in a talk

* Used resulting document-topic matrix as the input to calculate Jensen-Shannon divergence, which takes in probability distributions, to calculate similarity between TED talks

![Example Document-Topic Matrix](https://github.com/rweng18/tedtalks/blob/master/EDA_static/fig03_topic_distr.png)

## Deliverables & Insights

* TED talks are usually framed around solving a personal problem
* TED talks are meant to appeal to a general audience
* Although TED has been around for several decades, around early 2000’s developed a strong, wide-known image
* Tokenizing and processing transcripts is error-prone
* Can get TED talk recommendations by using a dropdown to search for titles


![Search talk from dropdown](https://github.com/rweng18/tedtalks/blob/master/EDA_static/rec_01.gif)


* The app will then show you the topic distribution, summary, assigned TED tags, and link to watch the talk, as displayed in the gifs


![Get topic distribution and URL](https://github.com/rweng18/tedtalks/blob/master/EDA_static/rec_02.gif)


* The app will then show the top 5 most similar and most similar talks, along with their summaries, tags, and links


![Get recommendations and URLs](https://github.com/rweng18/tedtalks/blob/master/EDA_static/rec_03.gif)


## Future Work

1. Word Embeddings & n-grams
2. Recommend based on topic or keywords
3. Applications of topic modeling
     1. Semantic analysis
     2. TED talks over time
     3. Predicting views or audience engagement
4. Improved frontend

