# Text-Info-Systems-Project
 
Indeed webscraper that returns matching summary and visualization of topic word distributions for a user-provided job search query.

## Team Members

- Doug Meyers
- James Robertson
- Brian Yoo

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


## Installation

### Conda Installation
```
conda create --name reverseresume python==3.6
conda install --file requirements.txt 
source activate reverseresume
```

### pip Installation
```
pip install -r requirements.txt --no-index --find-links file:///tmp/packages
```

## Usage

Run the app:
```
python main.py
```

Search a job (e.g., python developer)

![](./static/search_box.png | width=200) 

<br/>
And that's it! Keep in mind that the query will take some time (usually several minutes) to load.

<br/>

## Query Results

The query results will return a 250 word limit summary of the most relevant sentences obtained from the job postings.


<img src="./static/summary.png" width=300 >

<br/>

Reverse Resume uses Latent Dirichlet Allocation (LDA) to create its topic model. As such, we perform a principle component analysis (PCA) of the topic coverage to provide some idea as to how each topic varies from one another.


![](./static/pca_bubble_chart.png | width=200)

<br/>

Finally, Reverse Resume provide a chart of word distributions of relevant keywords for each topic.



![](./static/topic_charts.png | width=200) 