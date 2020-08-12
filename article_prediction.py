"""
Usage Instructions:
--- Create an ArticlePrediction Object
--- Call generate Function
--- Generate Function
------Input:
-------- An article text
-------- Data Type = String
------Output:
-------- Category of the input article
Category Reference:
--- This model can predict following categories:
-----Business
-----Sports
-----Technology
-----Entertainment
-----Politics
"""


import pickle
import pandas as pd

class ArticlePrediction:
   def __init__(self):
      self.tfidf_tr = None
      self.model = None
      with open("news_classification_tfidf_vectorizer", 'rb') as data:
         self.tfidf_tr = pickle.load(data)
      with open("news_classification_rf_model", 'rb') as data:
         self.model = pickle.load(data)

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