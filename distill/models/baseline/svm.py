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
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-3, random_state=42,
                          max_iter=5, tol=None)),
])

text_clf.fit(train.data, train.target)
predicted = text_clf.predict(test.data)
print(classification_report(test.target, predicted, target_names=test.target_names))

"""
                          precision    recall  f1-score   support

             alt.atheism       0.73      0.71      0.72       319
           comp.graphics       0.78      0.72      0.75       389
 comp.os.ms-windows.misc       0.73      0.78      0.75       394
comp.sys.ibm.pc.hardware       0.74      0.67      0.70       392
   comp.sys.mac.hardware       0.81      0.83      0.82       385
          comp.windows.x       0.84      0.76      0.80       395
            misc.forsale       0.84      0.90      0.87       390
               rec.autos       0.91      0.90      0.90       396
         rec.motorcycles       0.93      0.96      0.95       398
      rec.sport.baseball       0.88      0.90      0.89       397
        rec.sport.hockey       0.88      0.99      0.93       399
               sci.crypt       0.84      0.96      0.90       396
         sci.electronics       0.83      0.62      0.71       393
                 sci.med       0.87      0.86      0.87       396
               sci.space       0.84      0.96      0.90       394
  soc.religion.christian       0.76      0.94      0.84       398
      talk.politics.guns       0.70      0.92      0.80       364
   talk.politics.mideast       0.90      0.93      0.92       376
      talk.politics.misc       0.89      0.55      0.68       310
      talk.religion.misc       0.85      0.41      0.55       251

                accuracy                           0.82      7532
               macro avg       0.83      0.81      0.81      7532
            weighted avg       0.83      0.82      0.82      7532
"""
