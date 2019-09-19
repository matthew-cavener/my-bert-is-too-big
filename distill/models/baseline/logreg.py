from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
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

text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='log', penalty='l2',
                          alpha=1e-3, random_state=42,
                          max_iter=5, tol=None)),
])

text_clf.fit(train.data, train.target)
predicted = text_clf.predict(test.data)
print(classification_report(test.target, predicted, target_names=test.target_names))

"""
                          precision    recall  f1-score   support

             alt.atheism       0.76      0.48      0.59       319
           comp.graphics       0.73      0.69      0.71       389
 comp.os.ms-windows.misc       0.69      0.77      0.73       394
comp.sys.ibm.pc.hardware       0.70      0.69      0.70       392
   comp.sys.mac.hardware       0.85      0.71      0.77       385
          comp.windows.x       0.80      0.72      0.76       395
            misc.forsale       0.63      0.88      0.73       390
               rec.autos       0.87      0.87      0.87       396
         rec.motorcycles       0.88      0.93      0.90       398
      rec.sport.baseball       0.83      0.84      0.84       397
        rec.sport.hockey       0.86      0.96      0.91       399
               sci.crypt       0.79      0.89      0.84       396
         sci.electronics       0.81      0.52      0.63       393
                 sci.med       0.89      0.66      0.76       396
               sci.space       0.82      0.87      0.85       394
  soc.religion.christian       0.44      0.94      0.60       398
      talk.politics.guns       0.66      0.85      0.74       364
   talk.politics.mideast       0.87      0.89      0.88       376
      talk.politics.misc       0.98      0.39      0.56       310
      talk.religion.misc       0.84      0.10      0.18       251

                accuracy                           0.75      7532
               macro avg       0.79      0.73      0.73      7532
            weighted avg       0.78      0.75      0.74      7532
"""
