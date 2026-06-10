#Import Libraries
import numpy
import random 
import json 
import pickle
import nltk 
from nltk.stem.lancaster import LancasterStemmer
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import NotFittedError

app = Flask(__name__)
CORS(app)

stemmer = LancasterStemmer() #Creates an instance of Lancaster Stemmer 

#loads intents file 
with open("intents.json") as file:
    data = json.load(file)

#Process Data
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)

#Exception for when data cannot be found        
except Exception as e:
    print(f"Error loading data: {e}")

    words, labels, docs_x, docs_y = [], [], [], []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern) #breaks apart string text into individual text
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"] #Creates a list of root words
    words = sorted(list(set(words)))
    labels = sorted(labels)

    #training data preparation 
    training, output = [], []
    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = [1 if w in [stemmer.stem(wd.lower()) for wd in doc] else 0 for w in words]
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

except Exception as e:
    print(f"Error loading model: {e}")
    model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=10000)
    model.fit(training, output)

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)


#Convert user input into a bag-of-words vector for the model
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for n in s_words:
        for i, x in enumerate(words):
            if x in n:
                bag[i] = 1
    return numpy.array(bag)

#Chat Function
@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json.get('message', '')
        if user_input.lower() == "quit":
            return jsonify({'response': 'Chat session ended', 'end_session': True})

        input_data = bag_of_words(user_input, words).reshape(1, -1)
        try:
            output_data = model.predict_proba(input_data)[0]
        except NotFittedError:
            return jsonify({'error': 'Model not fitted yet.'}), 500
        
        output_index = numpy.argmax(output_data)
        tag = labels[output_index]

        if output_data[output_index] > 0.7:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
            bot_response = random.choice(responses)

        else: 
            bot_response = "Sorry, I didn't get that. Could you rephrase the question?"

        return jsonify({
            'response': bot_response,
            'tag': tag,
            'confidence': float(output_data[output_index])
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
