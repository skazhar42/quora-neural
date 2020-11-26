import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import xgboost as xgb
from keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
features_model = pickle.load(open('features_model.pkl', 'rb'))
bst = pickle.load(open('bst.pkl', 'rb'))
counts_vectorizer_vocab = pickle.load(open('counts_vectorizer.vocabulary_.pkl', 'rb'))
words_tokenizer = pickle.load(open('words_tokenizer.pkl', 'rb'))

def create_padded_seqs(texts, max_len=10):
    seqs = texts.apply(lambda s:
        [counts_vectorizer_vocab[w] if w in counts_vectorizer_vocab else 9999
         for w in words_tokenizer.findall(s.lower())])
    return pad_sequences(seqs, maxlen=max_len)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    input_questions = [str(x) for x in request.form.values()]
    input_df = pd.DataFrame({'question1': input_questions[0] , 'question2': input_questions[1]}, index=[0])
    q1 = create_padded_seqs(input_df['question1'])
    q2 = create_padded_seqs(input_df['question2'])
    F_test = features_model.predict([q1, q2], batch_size=128)
    dTest = xgb.DMatrix(F_test)
    pred = bst.predict(dTest, ntree_limit=bst.best_ntree_limit)
    output=''
    if pred[0]<0.2:
        output='Questions dont match '
    else:
        output = 'Questions match  '
    return render_template('index.html', prediction_text=output)

#port = int(os.getenv("PORT"))
if __name__ == "__main__":
    app.run(debug=True)
    #app.run(host='0.0.0.0', port=port)