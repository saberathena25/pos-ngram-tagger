# ==========================================
# POS Tagger + N-Gram Next Word Predictor
# Flask Web Application (Single Project)
# ==========================================

import nltk
from flask import Flask, render_template_string, request
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.util import ngrams
from collections import defaultdict, Counter

# Download required NLTK data (run once)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')

# ------------------------------------------
# Train N-gram Model (Sample Dataset)
# ------------------------------------------
text = """
the plant is growing fast the plant needs water
this plant is healthy the soil is dry the plant needs sunlight
"""

tokens = word_tokenize(text.lower())

# Bigram model
bigram_model = defaultdict(Counter)
for w1, w2 in ngrams(tokens, 2):
    bigram_model[w1][w2] += 1

# Trigram model
trigram_model = defaultdict(Counter)
for w1, w2, w3 in ngrams(tokens, 3):
    trigram_model[(w1, w2)][w3] += 1

# ------------------------------------------
# Prediction Functions
# ------------------------------------------

def predict_bigram(word):
    if word in bigram_model:
        return bigram_model[word].most_common(1)[0][0]
    return "No prediction"


def predict_trigram(w1, w2):
    if (w1, w2) in trigram_model:
        return trigram_model[(w1, w2)].most_common(1)[0][0]
    return "No prediction"

# ------------------------------------------
# Flask App
# ------------------------------------------
app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>POS Tagger + N-Gram Predictor</title>
</head>
<body>
    <h2>POS Tagger</h2>
    <form method="post">
        <input type="text" name="sentence" placeholder="Enter sentence" required>
        <button type="submit">Analyze</button>
    </form>

    {% if tags %}
        <h3>POS Tags:</h3>
        <ul>
        {% for word, tag in tags %}
            <li>{{word}} → {{tag}}</li>
        {% endfor %}
        </ul>
    {% endif %}

    <h2>Next Word Prediction</h2>
    <form method="post">
        <input type="text" name="context" placeholder="Enter one or two words" required>
        <button type="submit">Predict</button>
    </form>

    {% if prediction %}
        <h3>Prediction: {{prediction}}</h3>
    {% endif %}

</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    tags = None
    prediction = None

    if request.method == 'POST':
        # POS Tagging
        sentence = request.form.get('sentence')
        if sentence:
            words = word_tokenize(sentence)
            tags = pos_tag(words)

        # Prediction
        context = request.form.get('context')
        if context:
            words = context.lower().split()
            if len(words) == 1:
                prediction = predict_bigram(words[0])
            elif len(words) >= 2:
                prediction = predict_trigram(words[-2], words[-1])

    return render_template_string(HTML, tags=tags, prediction=prediction)

# ------------------------------------------
# Run App
# ------------------------------------------
if __name__ == '__main__':
    # use_reloader=False avoids Windows/OneDrive issues with Flask's debug reloader.
    host, base_port = '127.0.0.1', 8080
    for port in range(base_port, base_port + 10):
        try:
            print(f'\n>>> Open http://{host}:{port}/ in your browser (Ctrl+C to stop).\n')
            app.run(host=host, port=port, debug=True, use_reloader=False)
            break
        except OSError as e:
            if getattr(e, 'winerror', None) == 10048 or getattr(e, 'errno', None) in (98, 48):
                print(f'Port {port} is in use, trying {port + 1}...')
                continue
            raise

# ==========================================
# HOW TO RUN
# ==========================================
# 1. Install dependencies:
#    pip install flask nltk
#
# 2. Run:
#    python app.py
#
# 3. Open browser (URL is printed when the server starts; default port 8080):
#    http://127.0.0.1:8080
# ==========================================
