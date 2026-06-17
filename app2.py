import nltk
from flask import Flask, request, render_template_string
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.util import ngrams
from collections import defaultdict, Counter

app = Flask(__name__)

def setup_nltk():
    packages = [
        "punkt",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng"
    ]

    for pkg in packages:
        try:
            nltk.data.find(pkg)
        except:
            try:
                nltk.download(pkg)
            except:
                pass

setup_nltk()

training_text = """
the plant is growing fast
the plant needs water
this plant is healthy
the soil is dry
the plant needs sunlight
plants need nutrients
plants need water regularly
"""

tokens = word_tokenize(training_text.lower())

bigram_model = defaultdict(Counter)
for w1, w2 in ngrams(tokens, 2):
    bigram_model[w1][w2] += 1

trigram_model = defaultdict(Counter)
for w1, w2, w3 in ngrams(tokens, 3):
    trigram_model[(w1, w2)][w3] += 1


def predict_bigram(word):
    if word in bigram_model:
        return bigram_model[word].most_common(1)[0][0]
    return "No prediction available"


def predict_trigram(w1, w2):
    if (w1, w2) in trigram_model:
        return trigram_model[(w1, w2)].most_common(1)[0][0]
    return "No prediction available"


HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>POS Tagger and N-Gram Predictor</title>
    <style>
        body{
            font-family:Arial;
            margin:40px;
        }
        input{
            padding:8px;
            width:300px;
        }
        button{
            padding:8px 15px;
        }
        .box{
            margin-bottom:30px;
        }
    </style>
</head>
<body>

<h1>POS Tagger + Next Word Predictor</h1>

<div class="box">
<h2>POS Tagger</h2>
<form method="POST">
    <input type="hidden" name="action" value="pos">
    <input type="text" name="sentence" placeholder="Enter sentence">
    <button type="submit">Analyze</button>
</form>

{% if tags %}
<h3>Tags</h3>
<ul>
{% for word, tag in tags %}
<li>{{word}} → {{tag}}</li>
{% endfor %}
</ul>
{% endif %}
</div>

<div class="box">
<h2>N-Gram Predictor</h2>
<form method="POST">
    <input type="hidden" name="action" value="predict">
    <input type="text" name="context" placeholder="Enter one or two words">
    <button type="submit">Predict</button>
</form>

{% if prediction %}
<h3>Prediction: {{prediction}}</h3>
{% endif %}
</div>

</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():
    tags = None
    prediction = None

    if request.method == "POST":

        action = request.form.get("action")

        if action == "pos":
            sentence = request.form.get("sentence", "").strip()

            if sentence:
                words = word_tokenize(sentence)
                tags = pos_tag(words)

        elif action == "predict":
            context = request.form.get("context", "").lower().strip()

            if context:
                words = context.split()

                if len(words) == 1:
                    prediction = predict_bigram(words[0])

                elif len(words) >= 2:
                    prediction = predict_trigram(
                        words[-2],
                        words[-1]
                    )

    return render_template_string(
        HTML,
        tags=tags,
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
