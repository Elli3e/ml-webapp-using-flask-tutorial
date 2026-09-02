from pickle import load 
from flask import Flask, request, render_template

app = Flask(__name__)

model = load(open(
    "C:/Users/e_bab/Documents/4geeks/python-hello/ml-webapp-using-flask-tutorial/dense_keras_youtubr_spam.sav",
    "rb"
))
vectorizer = load(open(
    "C:/Users/e_bab/Documents/4geeks/python-hello/ml-webapp-using-flask-tutorial/vectorize_youtubr_spam.sav",
    "rb"
))
class_dict= {"0":"not_spam","1":"spam"}           

@app.route("/", methods = ["GET", "POST"])

def predict_spam():

    if request.method == "POST":
       
       comment = request.form["comment"]
       comment_tfidf = vectorizer.transform([comment])
       pred_class= model.predict(comment_tfidf)
       pred_class = int(pred_class [0][0]> 0.5)
       return class_dict[str(pred_class)]
    else:
     return render_template("index.html")

    
if __name__ == "__main__":
    app.run(debug=True)