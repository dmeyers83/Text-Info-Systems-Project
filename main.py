#import flask library
from flask import Flask, escape, url_for, render_template, request
import reverseresume as rr

Test_PCA_Object = [
    {
        "label": "Topic 0 - Word 1, 2, 3",
        "data": [{
            "x": 6,
            "y": 2,
            "r": 30
        }],
        "backgroundColor": "#1e88e5",
        "hoverBackgroundColor": "#fb8c00"
    },
    {
        "label": "Topic 1 - Word 4, 5, 6",
        "data": [{
            "x": 1,
            "y": 3,
            "r": 20
        }],
        "backgroundColor": "#1e88e5",
        "hoverBackgroundColor": "#fb8c00"
    },
    {
        "label": "Topic 2 - 7,8,9",
        "data": [{
            "x": 0,
            "y": 0,
            "r": 40
        }],
        "backgroundColor": "#1e88e5",
        "hoverBackgroundColor": "#fb8c00"
    },
    {
        "label": "Topic 3 - Word 9,10,12",
        "data": [{
            "x": 8,
            "y": 9,
            "r": 20
        }],
        "backgroundColor": "#1e88e5",
        "hoverBackgroundColor": "#fb8c00"
    }
]



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
  # resume.verbose=True
  resume.run(searchQuery)
  lda_model = resume.lda
  # print(resume.lda)

  summary = resume.summary
  print(summary)

  # get dataframe of lda probability results
  df = resume.get_lda_word_topic_probs()



  labelsList = []
  dataList= []
  dataList_tot = []
  numTopics=int(resume.num_topics)
  
  for n in df.topic_number_idx.unique():
    # create dataframe containing only n topic data
    df_sparse = df[df['topic_number_idx']==n]
    

    labelsList.append(list(df_sparse['word']))
    # float32 cant be serialized to JSON unless converted to regular float
    temp = list([float(x) for x in df_sparse['p_word_topic']])
    dataList.append(temp)  
    temp = list([float(x) for x in df_sparse['p_total']])
    dataList_tot.append(temp)



  return render_template('results.html', searchQuery=searchQuery, labels =labelsList, values=dataList, values_tot=dataList_tot,numTopics=numTopics, summary =summary, pca_data = Test_PCA_Object )

#run flask app
if __name__ == '__main__':
    app.run(debug=True)