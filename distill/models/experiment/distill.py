import ktrain
from ktrain import text

from keras.models import Model

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier, SGDRegressor, LinearRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder

import numpy as np

def activation_choice(x):
    softmax = list(np.exp(x)/sum(np.exp(x)))
    return softmax.index(max(softmax))

train = fetch_20newsgroups(
    subset='train',
    shuffle=True,
    random_state=42,
    data_home='/app/distill/data',
    remove=('headers', 'footers', 'quotes')
)
test = fetch_20newsgroups(
    subset='test',
    shuffle=True,
    random_state=42,
    data_home='/app/distill/data',
    remove=('headers', 'footers', 'quotes')
)

print('size of training set: %s' % (len(train['data'])))
print('size of validation set: %s' % (len(test['data'])))
print('classes: %s' % (train.target_names))

x_train = train.data
y_train = train.target
x_test = test.data
y_test = test.target

(x_train,  y_train), (x_test, y_test), preproc = text.texts_from_array(x_train=x_train, y_train=y_train,
                                                                       x_test=x_test, y_test=y_test,
                                                                       class_names=train.target_names,
                                                                       preprocess_mode='bert',
                                                                       ngram_range=1, 
                                                                       maxlen=400, 
                                                                       max_features=35000)

model = text.text_classifier('bert', train_data=(x_train, y_train))
learner = ktrain.get_learner(model, train_data=(x_train, y_train), batch_size=4)
learner.load_model('/app/distill/models/trained_models/bert.h5')
print(model.summary())
print(learner.print_layers())

print('\n\n\BERT validtion ==========')
print(learner.validate(val_data=(x_test, y_test), class_names=train.target_names))


print('\n\n\nLogReg validtion =========')
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', LinearRegression()),
])

ohe = OneHotEncoder(sparse=False)
train_ohe = ohe.fit_transform(train.target.reshape(-1, 1))
test_ohe = ohe.fit_transform(test.target.reshape(-1, 1))

text_clf.fit(train.data, train_ohe)
predicted = text_clf.predict(test.data)
prediction = list(map(activation_choice, predicted))
print(classification_report(test.target, prediction, target_names=test.target_names))


# Get the logits out of the model for the training data
from math import log
intermediate_layer_model = Model(inputs=learner.model.input,
                                 outputs=learner.model.get_layer('dense_1').output)
logits = intermediate_layer_model.predict(x_train)
logits = [list(map(log, _)) for _ in logits]

print(list(logits[0]))
print(test.target[0])


# Using linear regression here since it is continous, not discreet.
print('\n\n\nLogRegDist validtion =====')
text_clf_distill = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', LinearRegression()),
])

text_clf_distill.fit(train.data, logits)
predicted = text_clf_distill.predict(test.data)

prediction = list(map(activation_choice, predicted))

print(classification_report(test.target, prediction, target_names=test.target_names))


"""
BERT validtion ==========
                          precision    recall  f1-score   support

             alt.atheism       0.54      0.47      0.50       319
           comp.graphics       0.73      0.70      0.71       389
 comp.os.ms-windows.misc       0.70      0.66      0.68       394
comp.sys.ibm.pc.hardware       0.69      0.63      0.66       392
   comp.sys.mac.hardware       0.74      0.75      0.75       385
          comp.windows.x       0.80      0.83      0.81       395
            misc.forsale       0.87      0.84      0.85       390
               rec.autos       0.55      0.76      0.64       396
         rec.motorcycles       0.73      0.75      0.74       398
      rec.sport.baseball       0.92      0.81      0.86       397
        rec.sport.hockey       0.90      0.88      0.89       399
               sci.crypt       0.80      0.73      0.76       396
         sci.electronics       0.63      0.62      0.63       393
                 sci.med       0.83      0.83      0.83       396
               sci.space       0.77      0.80      0.78       394
  soc.religion.christian       0.73      0.75      0.74       398
      talk.politics.guns       0.61      0.67      0.64       364
   talk.politics.mideast       0.91      0.78      0.84       376
      talk.politics.misc       0.50      0.48      0.49       310
      talk.religion.misc       0.33      0.40      0.36       251

                accuracy                           0.72      7532
               macro avg       0.71      0.71      0.71      7532
            weighted avg       0.72      0.72      0.72      7532


LinearRegressionBaseline==========
                          precision    recall  f1-score   support

             alt.atheism       0.55      0.41      0.47       319
           comp.graphics       0.49      0.52      0.51       389
 comp.os.ms-windows.misc       0.40      0.46      0.43       394
comp.sys.ibm.pc.hardware       0.24      0.52      0.33       392
   comp.sys.mac.hardware       0.49      0.50      0.49       385
          comp.windows.x       0.73      0.49      0.58       395
            misc.forsale       0.28      0.61      0.38       390
               rec.autos       0.39      0.64      0.48       396
         rec.motorcycles       0.56      0.56      0.56       398
      rec.sport.baseball       0.79      0.65      0.71       397
        rec.sport.hockey       0.81      0.67      0.74       399
               sci.crypt       0.78      0.54      0.64       396
         sci.electronics       0.63      0.40      0.49       393
                 sci.med       0.87      0.57      0.69       396
               sci.space       0.73      0.52      0.61       394
  soc.religion.christian       0.66      0.63      0.64       398
      talk.politics.guns       0.62      0.52      0.57       364
   talk.politics.mideast       0.87      0.61      0.72       376
      talk.politics.misc       0.59      0.35      0.44       310
      talk.religion.misc       0.37      0.24      0.29       251

                accuracy                           0.53      7532
               macro avg       0.59      0.52      0.54      7532
            weighted avg       0.60      0.53      0.55      7532


LogRegDist validtion =====
                          precision    recall  f1-score   support

             alt.atheism       0.52      0.45      0.48       319
           comp.graphics       0.57      0.64      0.60       389
 comp.os.ms-windows.misc       0.48      0.57      0.52       394
comp.sys.ibm.pc.hardware       0.60      0.57      0.58       392
   comp.sys.mac.hardware       0.51      0.62      0.56       385
          comp.windows.x       0.75      0.58      0.65       395
            misc.forsale       0.75      0.68      0.72       390
               rec.autos       0.45      0.69      0.54       396
         rec.motorcycles       0.60      0.67      0.63       398
      rec.sport.baseball       0.80      0.70      0.74       397
        rec.sport.hockey       0.82      0.77      0.79       399
               sci.crypt       0.74      0.66      0.69       396
         sci.electronics       0.58      0.53      0.56       393
                 sci.med       0.77      0.65      0.70       396
               sci.space       0.69      0.61      0.65       394
  soc.religion.christian       0.63      0.66      0.64       398
      talk.politics.guns       0.58      0.58      0.58       364
   talk.politics.mideast       0.82      0.65      0.73       376
      talk.politics.misc       0.46      0.41      0.44       310
      talk.religion.misc       0.30      0.40      0.34       251

                accuracy                           0.61      7532
               macro avg       0.62      0.60      0.61      7532
            weighted avg       0.63      0.61      0.62      7532
"""
