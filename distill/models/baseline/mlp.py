from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

train = fetch_20newsgroups(
    subset='train',
    shuffle=True,
    random_state=42,
    data_home='/app/distill/data'
)
test = fetch_20newsgroups(
    subset='test',
    shuffle=True,
    random_state=42,
    data_home='/app/distill/data'
)

print('size of training set: %s' % (len(train['data'])))
print('size of validation set: %s' % (len(test['data'])))
print('classes: %s' % (train.target_names))

count_vect = CountVectorizer()
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MLPClassifier(
        hidden_layer_sizes=(128, 64, 64)
    )),
])

text_clf.fit(train.data, train.target)
predicted = text_clf.predict(test.data)
print(classification_report(test.target, predicted, target_names=test.target_names))

"""
took too long, lost patience, stopped training after 5 min.

                          precision    recall  f1-score   support

             alt.atheism       0.80      0.72      0.76       319
           comp.graphics       0.60      0.83      0.70       389
 comp.os.ms-windows.misc       0.78      0.63      0.69       394
comp.sys.ibm.pc.hardware       0.61      0.83      0.70       392
   comp.sys.mac.hardware       0.85      0.80      0.83       385
          comp.windows.x       0.87      0.73      0.79       395
            misc.forsale       0.87      0.83      0.85       390
               rec.autos       0.88      0.93      0.90       396
         rec.motorcycles       0.98      0.93      0.95       398
      rec.sport.baseball       0.88      0.95      0.91       397
        rec.sport.hockey       0.98      0.94      0.96       399
               sci.crypt       0.97      0.86      0.91       396
         sci.electronics       0.83      0.74      0.78       393
                 sci.med       0.91      0.77      0.83       396
               sci.space       0.89      0.85      0.87       394
  soc.religion.christian       0.81      0.92      0.86       398
      talk.politics.guns       0.75      0.84      0.79       364
   talk.politics.mideast       0.98      0.83      0.90       376
      talk.politics.misc       0.66      0.61      0.64       310
      talk.religion.misc       0.55      0.70      0.62       251

                accuracy                           0.82      7532
               macro avg       0.82      0.81      0.81      7532
            weighted avg       0.83      0.82      0.82      7532
"""
