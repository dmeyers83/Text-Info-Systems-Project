import sys
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import re
import urllib
import pickle
import re
import urllib
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from gensim.summarization import keywords
from gensim.summarization.summarizer import summarize
from gensim.summarization import mz_keywords
from gensim.models import Phrases
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.models import LsiModel
from gensim.models import HdpModel
from gensim.models.wrappers import LdaMallet
from gensim.parsing.preprocessing import remove_stopwords
from pprint import pprint
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import sys

class ReverseResume():
    def __init__(self):
        print("Reverse Resume Beta")
        self.importPackages()
        self.platform=sys.platform
        self.browser=self.getBrowser(self.platform)


    def __del__(self):
        cleanClose(self)

    def importPackages(self):
        try:
            import sys
            from bs4 import BeautifulSoup
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            import re
            import urllib
            import pickle
            import re
            import urllib
            import nltk
            from nltk.tokenize import RegexpTokenizer
            from nltk.stem.wordnet import WordNetLemmatizer
            from nltk.corpus import stopwords
            from gensim.summarization import keywords
            from gensim.summarization.summarizer import summarize
            from gensim.summarization import mz_keywords
            from gensim.models import Phrases
            from gensim.corpora import Dictionary
            from gensim.models import LdaModel
            from gensim.models import LsiModel
            from gensim.models import HdpModel
            from gensim.models.wrappers import LdaMallet
            from gensim.parsing.preprocessing import remove_stopwords
            from pprint import pprint
            from gensim.models import Word2Vec
            from sklearn.decomposition import PCA
            from gensim.models import Word2Vec
            from sklearn.manifold import TSNE
            from sklearn.decomposition import PCA
            import pandas as pd
            import seaborn as sns
            import numpy as np
            import matplotlib.pyplot as plt
            import sys

        except ImportError as e:
            print(e)

    def getBrowser(self, platform):
        options = Options()
        options.headless = True

        if platform == "linux" or platform == "linux2":
            return (webdriver.Chrome('./browser/chromedriverlinux',options=options))
        elif platform == "darwin":
            return (webdriver.Chrome('./browser/chromedriver',options=options))
        elif platform == "win32":
            return (webdriver.Chrome('./browser/chromedriver.exe',options=options))

    def cleanClose(self):
        self.browser.close()
        self.browser.quit()

    def get_js_soup(self,url,browser):
        self.browser.get(url)
        res_html = self.browser.execute_script('return document.body.innerHTML')
        soup = BeautifulSoup(res_html,'html.parser') #beautiful soup object to be used for parsing html content
        return soup

    def process_bio(self,bio):
        bio = bio.encode('ascii',errors='ignore').decode('utf-8')       #removes non-ascii characters
        bio = re.sub('\s+',' ',bio)       #repalces repeated whitespace characters with single space
        return bio

    def remove_script(self,soup):
        for script in soup(["script", "style"]):
            script.decompose()
        return soup

    def write_lst(self,lst,file_):
        with open(file_,'w') as f:
            for l in lst:
                f.write(l)
                f.write('\n')


    def scrape_search_result_page(self,dir_url,page_result,browser):
        print ('-'*20,'Scraping indeed search result page '+ str(page_result)+'','-'*20)
        indeed_links = []
        soup = self.get_js_soup(dir_url,self.browser)
        for link_holder in soup.find_all('div',class_='title'): #get list of all <div> of class 'photo nocaption'
            rel_link = link_holder.find('a')['href'] #get url
            if rel_link != '':
                indeed_links.append('https://www.indeed.com' + rel_link)
        print ('-'*20,'Found {} indeed search urls'.format(len(indeed_links)),'-'*20)
        return indeed_links

    def run(self,query, location):

        q = 'Python Developer'
        l = 'New+York+State' #location of job
        numPage = 20 #num pages to scrap links from
        allLinks = [] # list to capture
        start = 0 #pagnigation variable, page 1 = 0, page 2 = 10, page 3 = 30, etc

        # loop over n number of pages
        for page_result in range(numPage):
            start = page_result* 10 #increment the variable used to denote the next page
            search_result_url = 'https://www.indeed.com/jobs?q='+ q +'&l='+ l +'&start='+str(start) #build query string
            print(search_result_url)
            jobSearchResult = self.scrape_search_result_page(search_result_url,page_result, self.browser) # call scraper function
            allLinks.extend(jobSearchResult) #add to link



        # ### write to file for debugging

        # In[4]:


        #Remove Duplicates
        print(len(allLinks))
        allLinks = list(set(allLinks))
        print (len(allLinks))


        # In[5]:


        print(allLinks)
        job_urls_file = 'jobSearchResult' +q+'.txt'
        # write to file
        self.write_lst(allLinks,job_urls_file)


        # ## Scrape page data for each link

        # In[6]:





        # In[7]:


        homepage_found = False
        page_data = ''
        page_data_list = []
        for link_num, indeed_url in enumerate(allLinks):
            print("Accessing link",link_num+1,"of",len(allLinks))
            try:
                page_soup = self.remove_script(self.get_js_soup(indeed_url,self.browser))

            except:
                print ('Could not access {}'.format(indeed_url))

            page_data = self.process_bio(page_soup.get_text(separator=' '))  #helper function from bio hw to clean up text

            #remove header
            page_data = page_data[189:] #the 189 slice removes the header of the indeed pages

            #remove footer
            footer_position = page_data.find('save job') #find the position of 'save job' which starts the footer
            trimStringBy = footer_position - len(page_data) #returns a negative number to trim the string by
            page_data = page_data[:trimStringBy] #drop footer
            page_data = remove_stopwords(page_data)
            page_data_list.append(page_data)



        # ## Print page data and write to file for debug

        # **Footer still has some text at the end which isn't properly cleaned

        # In[8]:


        print(page_data_list[1])
        document_set = page_data_list
        page_data_file = 'pageText' +q+'.txt'
        self.write_lst(page_data_list,page_data_file)



        # Create single document by concatenating all documents
        all_documents = ""

        for doc in page_data_list:
            all_documents += doc


        # In[11]:


        #keywords
        keywords(all_documents).split('\n')


        # In[12]:


        print(summarize(all_documents, word_count  = 250))


        # In[13]:


        print(mz_keywords(all_documents,scores=True,threshold=0.001))


        # # Topic Modeling

        # ### import libraries

        # In[14]:





        # ### tokenize the documents

        # In[15]:


        docs = page_data_list

        # Split the documents into tokens.
        tokenizer = RegexpTokenizer(r'\w+')
        for idx in range(len(docs)):
            docs[idx] = docs[idx].lower()  # Convert to lowercase.
            docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.

        # Remove numbers, but not words that contain numbers.
        docs = [[token for token in doc if not token.isnumeric()] for doc in docs]

        # Remove words that are less than 3 characters.
        docs = [[token for token in doc if len(token) > 3] for doc in docs]


        # must download wordnet!!!

        # In[16]:


        # nltk.download('wordnet')


        # ### lemmatize the documents

        # In[17]:


        lemmatizer = WordNetLemmatizer()
        docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]


        # ### compute bigrams

        # In[18]:


        # Add bigrams and trigrams to docs (only ones that appear 20 times or more).
        bigram = Phrases(docs, min_count=10)
        for idx in range(len(docs)):
            for token in bigram[docs[idx]]:
                if '_' in token:
                    # Token is a bigram, add to document.
                    docs[idx].append(token)


        # ### remove rare and common tokens

        # In[19]:


        # Create a dictionary representation of the documents.
        dictionary = Dictionary(docs)

        # Filter out words that occur less than 20 documents, or more than 50% of the documents.
        dictionary.filter_extremes(no_below=20, no_above=0.75)


        # In[20]:


        # Bag-of-words representation of the documents.
        corpus = [dictionary.doc2bow(doc) for doc in docs]


        # In[21]:


        print('Number of unique tokens: %d' % len(dictionary))
        print('Number of documents: %d' % len(corpus))


        # ## Build LDA Model

        # In[22]:


        # Set training parameters.
        num_topics = 20
        chunksize = 2000
        passes = 20
        iterations = 400
        eval_every = None  # Don't evaluate model perplexity, takes too much time.

        # Make a index to word dictionary.
        temp = dictionary[0]  # This is only to "load" the dictionary.
        id2word = dictionary.id2token

        lda_model = LdaModel(
            corpus=corpus,
            id2word=id2word,
            chunksize=chunksize,
            alpha='auto',
            eta='auto',
            iterations=iterations,
            num_topics=num_topics,
            passes=passes,
            eval_every=eval_every
        )


        # In[23]:


        num_top_words = 10
        top_topics = lda_model.top_topics(corpus, topn=num_top_words) #, num_words=10)
        # print(top_topics)
        # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
        avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
        print('Average topic coherence: %.4f.' % avg_topic_coherence)

        for idx, topic in lda_model.print_topics(-1):
            print('Topic: {} \nWords: {}'.format(idx, topic))


        # ## Visualize Topics

        # In[266]:





        # In[339]:


        #visualize without sorting topics

        # initialize
        topic_v = np.zeros(num_topics)
        p_word_topic = []
        word = []
        topic_number_idx = []

        # create dataframe
        for i in range(num_topics):

            word_id, prob, = zip(*lda_model.get_topic_terms(i))
            for j in range(len(word_id)):
                p_word_topic.append(prob[j])
                word.append(id2word[word_id[j]])
                topic_number_idx.append(i)
        dict = {'p_word_topic': p_word_topic,
                'word': word,
                'topic_number_idx': topic_number_idx,
               }
        df = pd.DataFrame(dict)
        df['p_total'] = df.groupby(['word'])['p_word_topic'].transform('sum')
        # print(df)
        # sorted by top topics
        for n in df.topic_number_idx.unique():
            df_sparse = df[df['topic_number_idx']==n]
            df.drop_duplicates(subset=['word', 'p_word_topic'])
            df_sparse_total = df[df['topic_number_idx']==n]

            plt.figure()
            plt.title(r'$p(\theta_{'+str(n)+'})=$'+str('%0.3f'%(df_sparse.p_word_topic.sum())))
            sns.set_color_codes("pastel")
            sns.barplot(x='p_total',y='word',data=df_sparse_total,color='b',alpha = 1,label=r'$p(w)$')
            sns.set_color_codes("pastel")
            sns.barplot(x='p_word_topic',y='word',data=df_sparse,color='r',alpha =1,label=r'$p(w|\theta_{'+str(n)+'})$')

            plt.legend(bbox_to_anchor=[1.25, 1.0])

            plt.xlabel('likelihood')
        plt.show()


        # ### use coherence to sort dataframes (deprecated since the topic numbers get re-indexed)

        # In[268]:


        def create_df_from_top_topics(top_topics):
            ''' function that coverts lda_model.top_topics() data into pandas dataframe
            '''
            # initialize empty lists
            p_word_topic = []
            word = []
            coherence = []
            topic_number_idx = []

            # loop through topics
            for t, topic in enumerate(top_topics):

                # obtain probability and word vectors
                p_vec,w_vec = zip(*topic[0])
                for p,w in zip(p_vec,w_vec):
                    p_word_topic.append(p)
                    word.append(w)
                    coherence.append(topic[1])
                    topic_number_idx.append(t)

            # covert lists to dict
            dict = {'p_word_topic': p_word_topic,
                    'word': word,
                    'coherence': coherence,
                    'topic_number_idx': topic_number_idx,
                   }

            return pd.DataFrame(dict)

        df = create_df_from_top_topics(top_topics)


        # In[288]:


        # # sorted by top topics (coherence measure)
        # for n in df.topic_number_idx.unique():
        #     df_sparse = df[df['topic_number_idx']==n]

        #     plt.title('topic '+str(n)+', coherence='+str('%4.3f'%df_sparse.coherence.mean()))
        #     sns.barplot(x='p_word_topic',y='word',data=df_sparse,color='b',alpha = 0.6)
        # #     print(df_sparse)
        #     plt.xlabel('likelihood')
        # #     plt.xlim([0, max(df.p_word_topic)])
        # plt.show()



        # ### PCA

        # https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/

        # plot topics based on principle components

        # In[296]:





        # PCA by topic weight
        # shape = num_docs * number_topics
        topic_weights = []

        print(lda_model[corpus])
        for i, row_list in enumerate(lda_model[corpus]):
            # row list contains list of topic number and probability of topic as a tuple
            # note that each doc can have multiple topics

            # initialize zero list
            r = np.zeros(num_topics)

            topic_n, p_topic =zip(*row_list)

            # store topic prob into r
            for i in range(len(row_list)):
                r[topic_n[i]]=p_topic[i]
            topic_weights.append(r)

        # print(np.shape(topic_weights))

        # Array of topic weights
        X = pd.DataFrame(topic_weights).fillna(0).values
        # X = lda_model.get_topics() # returns term-topic matrix

        # only look at the first two components
        pca = PCA(n_components=2)
        result = pca.fit_transform(X)

        # Dominant topic number in each doc
        topic_num = np.argmax(X, axis=1)

        plt.figure(figsize=(8,8))

        print('number of documents:',len(lda_model[corpus]))
        labels=['topic_'+"%02d" %(x) for x in range(num_topics+1)]

        # df_topic_weight = pd.DataFrame({'x':result[:,0],'y':result[:,1]})
        # sns.scatterplot(x=df_topic_weight.x, y=df_topic_weight.y, data =df_topic_weight)

        df_topic_weight = pd.DataFrame({'x':result[:,0],'y':result[:,1],'label':[labels[x] for x in topic_num]}).sort_values(by='label')
        sns.scatterplot(x=df_topic_weight.x, y=df_topic_weight.y, hue=df_topic_weight.label, data =df_topic_weight)

        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend( bbox_to_anchor=[1.2, 1.0])

        plt.show()


        # plot topics based on T-SNE

        # In[297]:




        # TSNE by topic weight

        # Dominant topic number in each doc
        topic_num = np.argmax(X, axis=1)

        # tSNE Dimension Reduction
        tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
        tsne_lda = tsne_model.fit_transform(X)

        # Plot the Topic Clusters
        print('number of documents:',len(tsne_lda))
        plt.figure(figsize=(8,8))
        df_topic_weight = pd.DataFrame({'x':tsne_lda[:,0],'y':tsne_lda[:,1],'label':[labels[x] for x in topic_num]}).sort_values(by='label')
        sns.scatterplot(x=df_topic_weight.x, y=df_topic_weight.y, hue=df_topic_weight.label, data =df_topic_weight)
        plt.legend( bbox_to_anchor=[1.2, 1.0])
        plt.xlabel('')
        plt.ylabel('')
        plt.show()


        # In[107]:


        # PCA by LDA -> word2vec

        # TODO: might need to re-weigh the sentence based on their probabilities
        topic_sentence = []
        for topic in top_topics:
            p_vec,w_vec = zip(*topic[0])
            topic_sentence.append(w_vec)

        # train model by creating word2vec neural network
        model = Word2Vec(topic_sentence, min_count=1)

        # fit a 2d PCA model to the vectors
        X = model[model.wv.vocab]
        print(np.shape(X))

        # only look at the first two components
        pca = PCA(n_components=2)
        result = pca.fit_transform(X)

        # create a scatter plot of the projection
        plt.figure(figsize=(8,8))

        pca1 = result[:, 0]
        pca2 = result[:, 1]


        plt.scatter(pca1, pca2)
        words = list(model.wv.vocab)

        for i, word in enumerate(words):
            plt.annotate(word, xy=(result[i, 0], result[i, 1]))

        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.show()
