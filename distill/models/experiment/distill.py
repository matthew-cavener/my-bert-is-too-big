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

x_train = train.data
y_train = train.target
x_test = test.data
y_test = test.target

(x_train,  y_train), (x_test, y_test), preproc = text.texts_from_array(x_train=x_train, y_train=y_train,
                                                                       x_test=x_test, y_test=y_test,
                                                                       class_names=train.target_names,
                                                                       ngram_range=2, 
                                                                       maxlen=1000, 
                                                                       max_features=50000)

model = text.text_classifier('nbsvm', train_data=(x_train, y_train))
learner = ktrain.get_learner(model, train_data=(x_train, y_train), val_data=(x_test, y_test))
learner.load_model('/app/distill/models/trained_models/nbsvm')

print('\n\n\nNBSVM validtion ==========')
print(learner.validate(val_data=(x_test, y_test), class_names=train.target_names))
print(model.summary())
print(learner.print_layers())


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
intermediate_layer_model = Model(inputs=learner.model.input,
                                 outputs=learner.model.get_layer('activation_1').input)
logits = intermediate_layer_model.predict(x_train)

print(list(logits[0]))
print(learner.layer_output(4))
print(activation_choice(logits[0]))
print(activation_choice(learner.layer_output(4)[0]))
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
NBSVM=============
                          precision    recall  f1-score   support

             alt.atheism       0.86      0.74      0.80       319
           comp.graphics       0.74      0.74      0.74       389
 comp.os.ms-windows.misc       0.78      0.70      0.74       394
comp.sys.ibm.pc.hardware       0.70      0.76      0.73       392
   comp.sys.mac.hardware       0.84      0.85      0.84       385
          comp.windows.x       0.84      0.83      0.83       395
            misc.forsale       0.85      0.87      0.86       390
               rec.autos       0.88      0.91      0.90       396
         rec.motorcycles       0.96      0.95      0.95       398
      rec.sport.baseball       0.93      0.92      0.93       397
        rec.sport.hockey       0.94      0.98      0.96       399
               sci.crypt       0.91      0.93      0.92       396
         sci.electronics       0.81      0.75      0.78       393
                 sci.med       0.91      0.82      0.86       396
               sci.space       0.92      0.92      0.92       394
  soc.religion.christian       0.91      0.93      0.92       398
      talk.politics.guns       0.80      0.89      0.85       364
   talk.politics.mideast       0.96      0.88      0.92       376
      talk.politics.misc       0.72      0.67      0.69       310
      talk.religion.misc       0.55      0.75      0.64       251

                accuracy                           0.84      7532
               macro avg       0.84      0.84      0.84      7532
            weighted avg       0.85      0.84      0.84      7532


LinearRegressionBaseline==========
                          precision    recall  f1-score   support

             alt.atheism       0.82      0.73      0.77       319
           comp.graphics       0.56      0.70      0.62       389
 comp.os.ms-windows.misc       0.42      0.56      0.48       394
comp.sys.ibm.pc.hardware       0.43      0.67      0.53       392
   comp.sys.mac.hardware       0.83      0.75      0.79       385
          comp.windows.x       0.72      0.60      0.65       395
            misc.forsale       0.56      0.75      0.64       390
               rec.autos       0.93      0.80      0.86       396
         rec.motorcycles       0.97      0.92      0.95       398
      rec.sport.baseball       0.92      0.87      0.90       397
        rec.sport.hockey       0.97      0.91      0.94       399
               sci.crypt       0.96      0.84      0.90       396
         sci.electronics       0.69      0.70      0.70       393
                 sci.med       0.91      0.73      0.81       396
               sci.space       0.90      0.79      0.84       394
  soc.religion.christian       0.85      0.90      0.87       398
      talk.politics.guns       0.78      0.86      0.82       364
   talk.politics.mideast       0.98      0.82      0.89       376
      talk.politics.misc       0.84      0.58      0.69       310
      talk.religion.misc       0.73      0.64      0.68       251

                accuracy                           0.76      7532
               macro avg       0.79      0.76      0.77      7532
            weighted avg       0.79      0.76      0.77      7532


LinearRegressionDistilled=========
                          precision    recall  f1-score   support

             alt.atheism       0.77      0.69      0.73       319
           comp.graphics       0.73      0.61      0.67       389
 comp.os.ms-windows.misc       0.29      0.62      0.40       394
comp.sys.ibm.pc.hardware       0.64      0.62      0.63       392
   comp.sys.mac.hardware       0.81      0.69      0.75       385
          comp.windows.x       0.78      0.59      0.68       395
            misc.forsale       0.82      0.63      0.71       390
               rec.autos       0.88      0.83      0.86       396
         rec.motorcycles       0.94      0.90      0.92       398
      rec.sport.baseball       0.88      0.90      0.89       397
        rec.sport.hockey       0.93      0.91      0.92       399
               sci.crypt       0.89      0.84      0.86       396
         sci.electronics       0.75      0.59      0.66       393
                 sci.med       0.87      0.77      0.82       396
               sci.space       0.90      0.84      0.87       394
  soc.religion.christian       0.85      0.82      0.84       398
      talk.politics.guns       0.75      0.87      0.81       364
   talk.politics.mideast       0.95      0.84      0.89       376
      talk.politics.misc       0.60      0.59      0.60       310
      talk.religion.misc       0.45      0.67      0.54       251

                accuracy                           0.75      7532
               macro avg       0.77      0.74      0.75      7532
            weighted avg       0.78      0.75      0.76      7532
"""
