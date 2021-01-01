from flask import Flask,request,render_template
import  pickle
import numpy as np
import feather
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer


def feature_matrix(xTrain,voc_train):
    X =np.zeros((xTrain.shape[0],len(voc_train)))
    for i in range(xTrain.shape[0]):
        subj = xTrain[i]
        for name in subj.split():
            if name in voc_train:
                X[i,voc_train[name]] +=1
    return X


def tf_idf(X_train_mat):
    tfidf = TfidfTransformer(norm='l2')
    tfidf.fit(X_train_mat)
    tf_idf_matrix = tfidf.transform(X_train_mat)
    tf_idf_mat = tf_idf_matrix.todense()
    return tf_idf_mat


app = Flask(__name__)
mode = pickle.load(open('mode.pkl','rb'))


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/Result',methods= ['POST'])
def home():
    inp_features  = str(request.form['mail'])
    x_model = pd.DataFrame([inp_features])
    x_model =x_model[0]
    vocab = pickle.load(open('voc.pkl','rb'))
    X = feature_matrix(x_model,vocab)
    tf_df_x = tf_idf(X)
    model_svm = pickle.load(open('mode.pkl','rb'))
    result_svm = model_svm.predict(tf_df_x)
    if result_svm == 1:
        return render_template('home.html',prediction_text = ['Spam'])
    else:
        return render_template('home.html',prediction_text = ['Ham'])



if __name__ == '__main__':
    app.run(debug =True)
