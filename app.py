from pickle import load 
from flask import Flask, request, render_template

app = Flask(__name__)

model = load(open(
    "support_vector_youtube_spam.sav",
    "rb"
))
vectorizer = load(open(
    "vectorize_youtubr_spam.sav",
    "rb"
))
class_dict= {"0":"not_spam","1":"spam"}           

@app.route("/", methods = ["GET", "POST"])

def predict_spam():

    if request.method == "POST":
       
       comment = request.form["comment"]
       comment_tfidf = vectorizer.transform([comment])
       pred_class= model.predict(comment_tfidf)
       pred_class = int(pred_class[0])
       return class_dict[str(pred_class)]
    else:
     return render_template("index.html")

    
if __name__ == "__main__":
    app.run(debug=True)