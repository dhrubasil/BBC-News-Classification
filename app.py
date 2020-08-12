import cherrypy
import pickle
import pandas as pd
# https://www.amazon.in/s?k=led+tv&page=5&qid=1595995683&ref=sr_pg_3

class ArticlePrediction(object):
   def __init__(self):
      self.tfidf_tr = None
      self.model = None
      with open("news_classification_tfidf_vectorizer", 'rb') as data:
         self.tfidf_tr = pickle.load(data)
      with open("news_classification_rf_model", 'rb') as data:
         self.model = pickle.load(data)

   @cherrypy.expose
   def index(self):      
   	return "Welcome to the News Analysis Page. Still Improving..."

   @cherrypy.expose
   def generate(self, article_text=None):
      if article_text == None:
         return "This is not a valid news article!!"
      else:
         article_text = article_text.lower()
         article_frame = pd.DataFrame({"Text":[article_text]})
         article_feature = self.tfidf_tr.transform(article_frame.Text)
         prediction = self.model.predict(article_feature)

         if prediction[0] == 0:
            return "Category is Business"
         elif prediction[0] == 1:
            return "Category is Tech"
         elif prediction[0] == 2:
            return "Category is Politics"
         elif prediction[0] == 3:
            return "Category is Sports"
         else:
            return "Category is Entertainment"
   @cherrypy.expose                 
   def addition_random(self,first_number):
      return "Input is some number"
		
if __name__ == '__main__':
   cherrypy.quickstart(ArticlePrediction ())