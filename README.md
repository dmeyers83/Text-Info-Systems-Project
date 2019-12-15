# Text-Info-Systems-Project
 
## Overview

Reverse Resume helps users learn what are the most relevant skillsets and keywords that hiring managers are seeking.  Users can use these insights to tailor their resume so they are more attractive to job market or build out the desired skillset through training or higher education.

Users simply enter in the job position they are interested in pursuing (‘Python Developer”), click search and Reverse Resume will scrape 100+ job descriptions from Indeed.com and run various text analysis models to return a summary and visualization of the most relevant sentences, key topics and analytics on how related the topics are.  


## Team Members and Contributions

- Doug Meyers: Modified the class HW webscraping code for Indeed.com, coded procedure for data cleaning, adopted test code for LDA and Text Rank from Gensim package, Flask integration with Chart JS, and contributed to documenation and video.
- James Robertson: Migrated codebase from Jupyter notebook into Class object and API, contributed to documentation, created video and made all final cuts
- Brian Yoo: Performed EDA for LDA model, created PCA/probability diagrams and data pipelines for Chart JS figures, contributed to documentation and video

Note: All group members contributed 20+ hours to the project.

## Dependencies

Webscraping packages
- [beautiful soup](https://www.crummy.com/software/BeautifulSoup/)
- [selenium](https://selenium-python.readthedocs.io/installation.html)
- [chromedriver](https://chromedriver.chromium.org/)

Data Wrangling packages
- [pandas](https://pandas.pydata.org/)

Text mining/ML packages
- [scikit-learn](https://scikit-learn.org/stable/) - package used for machine learning
- [nltk](https://www.nltk.org/install.html) - package used for natural language processing and document processing
- [gensim](https://radimrehurek.com/gensim/) - package used for topic modeling

Visualization packages
- [pprint](https://docs.python.org/3/library/pprint.html)
- [matplotlib](https://matplotlib.org/)

Web Framework packages
- [flask](https://www.palletsprojects.com/p/flask/)
- [materlize css](https://materializecss.com/)
- [jquery](https://jquery.com/)

## Installation

### Conda Installation
```
conda create --name reverseresume python==3.6
conda install --file requirements.txt 
source activate reverseresume
```

### Pip Installation
```
pip install -r requirements.txt --no-index --find-links file:///tmp/packages
```


### Download Wordnet Lexical Database
Open up your python console
```
import nltk
nltk.download('wordnet')
```

## Software Implementation   

<img src="./static/RRDirectory.PNG">

- main.py: This is the main python application that runs the flask website and calls the reverseresume class which contains the majority of the business logic.  PCA was also implemented in this file. [add more]
- reverseresume.py:  This python file class contains the code used to scrape Indeed.com and implementes the various gensim text models to run text rank and LDA.  [add more]
- templates dir:  This directory contains the html and javascript files used to render to search landing page and the results.
- browser dir:  This directory contains chrome drivers for linux, windows and mac operating systems.
- static dir:  This directory contains static html and image files.
## Usage

Run the app:
```
python main.py
```
Access the page via URL: http://127.0.0.1:5000/

** If you get a selenium web driver or chrome error after your first search you may need to upgrade your chrome driver using this link:
https://chromedriver.chromium.org/downloads
The chrome drivers are in the browser directory


Search a job (e.g., python developer)

<p align="center">
<img src="./static/search_box.png" width="700">
</p>

<br/>
And that's it! Keep in mind that the query will take some time (usually several minutes) to load.

<br/>

## Query Results

The query results will return a 250 word limit summary of the most relevant sentences obtained from the job postings. Gensim's TextRank summary is created by first placing all sentences from the corpus into a graph data structure, where each sentence represents a node on the graph. The edges of nodes are then generated using a similarity function, such as those used in text retrieval processes (for instance, BM25, BM25+ and cosine similarity). Once this graph structure is created, Google’s PageRank algorithm is used to score each sentence. The sentences that have the highest score are the ones that show up on the summary. More detail on the algorithm can be found [here](https://arxiv.org/abs/1602.03606) . 

<p align="center">
<img src="./static/summary.png" width="700">
</p>

<br/>

  Reverse Resume uses gensim's Latent Dirichlet Allocation (LDA) to determine the most likely topic a document is categorized as. LDA is a generative model and the Bayesian form of the Probabilistic Latent Semantic Analysis model, which can be used to categorize documents based on their topics, where each topic is defined as a distribution of words. Keep in mind that documents can be associated with multiple topics, although for the following chart, we only label each document based on their most probable topic. Here we perform principle component analysis (PCA) of the topic coverage to provide some idea as to how each topic varies from one another. Each bubble in the chart represents a topic. The size of each bubble is determined based on the number of job postings for a given topic.

<p align="center">
<img src="./static/pca_bubble_chart.png" width="700">
</p>


<br/>

Finally, Reverse Resume provide a chart of word distributions of relevant keywords for each topic. Here we show the product of the conditional probability of words given the topic (p(w|theta)) with the probability of the topic (p(theta)), and compare them to the probability of the word (p(w)) for the entire corpus. The point of these charts it to show which words tend to be most relevant for each topic.


<p align="center">
<img src="./static/topic_charts.png" width="700">
</p>
