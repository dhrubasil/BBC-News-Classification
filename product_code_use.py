from article_prediction import ArticlePrediction 


if __name__ == "__main__":
	art_pred = ArticlePrediction()
	article_text = input("Kindly Enter Your Article Text:")
	predicted_category = art_pred.generate(article_text)


	print (predicted_category)