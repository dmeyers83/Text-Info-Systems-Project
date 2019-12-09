#import flask library
from flask import Flask, escape, url_for, render_template, request
import reverseresume as rr
import pandas as pd
import numpy as np


#initialize flask app
app = Flask(__name__)

#Path to render the html to display the search page


@app.route('/')
def displaySearch():
  return render_template('landingpage.html')

#path to render the results page.  the searchQuery parameter is passed from the landing page.


@app.route('/result/<searchQuery>')
def diplayresult(searchQuery):

  # call Reverse Resume
  resume = rr.ReverseResume()
#   resume.verbose = True
  resume.run(searchQuery)
  lda_model = resume.lda
  # print(resume.lda)

  summary = resume.summary
  print(summary)

  # get dataframe of lda probability results
  df = resume.get_lda_word_topic_probs()

  # store word topic prob results from df to html amenable format
  labelsList = []
  dataList = []
  dataList_tot = []
  numTopics = int(resume.num_topics)

  for n in df.topic_number_idx.unique():
    # create dataframe containing only n topic data
    df_sparse = df[df['topic_number_idx'] == n]

    labelsList.append(list(df_sparse['word']))
    # float32 cant be serialized to JSON unless converted to regular float
    temp = list([float(x) for x in df_sparse['p_word_topic']])
    dataList.append(temp)
    temp = list([float(x) for x in df_sparse['p_total']])
    dataList_tot.append(temp)

  # PCA
  labels=['Topic '+str(x) for x in range(numTopics)]
  topic_num = np.argmax(resume.lda_topic_coverage, axis=1)
  # store into df
  df_pca = pd.DataFrame({'x': resume.pca[:, 0], 'y': resume.pca[:, 1], 'label': [labels[x] for x in topic_num]}).sort_values(by='label')
  # get centroids of xy points
  df_pca_grouped = df_pca.groupby(['label']).mean()
  # store counts for each topic
  df_pca_grouped['count'] = df_pca.groupby(['label']).count().x
  df_pca_grouped.reset_index(inplace=True)
  # set max radius for plot
  max_radius = 30
  # convert into lists
  pca_x=list(["%0.3f"%(x) for x in df_pca_grouped.x])
  pca_y=list(["%0.3f"%(x) for x in df_pca_grouped.y])
  # rescale radius based on counts and max_radius
  pca_r = np.array(df_pca_grouped['count']) / df_pca_grouped['count'].max() * max_radius
  pca_r = list([int(x) for x in pca_r])
  
  # create PCA Object for rendering
  PCA_Object = [{"label": labels[k]+' (number of docs: '+str(np.array(df_pca_grouped['count'])[k])+')',
                 "data": [{"x": pca_x[k],
                           "y": pca_y[k],
                           "r": pca_r[k]
                           }],
                 "backgroundColor": "#1e88e5",
                 "hoverBackgroundColor": "#fb8c00"}
                for k in range(len(pca_x))]

  return render_template('results.html',
                         searchQuery=searchQuery,
                         labels=labelsList,
                         values=dataList,
                         values_tot=dataList_tot,
                         numTopics=numTopics,
                         summary=summary,
                         pca_data=PCA_Object)


#run flask app
if __name__ == '__main__':
    app.run()
