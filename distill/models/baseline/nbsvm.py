import ktrain
from ktrain import text

from sklearn.datasets import fetch_20newsgroups

train_b = fetch_20newsgroups(
    subset='train',
    shuffle=True,
    random_state=42,
    data_home='/app/distill/data',
    remove=('headers', 'footers', 'quotes')
)
test_b = fetch_20newsgroups(
    subset='test',
    shuffle=True,
    random_state=42,
    data_home='/app/distill/data',
    remove=('headers', 'footers', 'quotes')
)

print('size of training set: %s' % (len(train_b['data'])))
print('size of validation set: %s' % (len(test_b['data'])))
print('classes: %s' % (train_b.target_names))

x_train = train_b.data
y_train = train_b.target
x_test = test_b.data
y_test = test_b.target

(x_train,  y_train), (x_test, y_test), preproc = text.texts_from_array(x_train=x_train, y_train=y_train,
                                                                       x_test=x_test, y_test=y_test,
                                                                       class_names=train_b.target_names,
                                                                       ngram_range=2, 
                                                                       maxlen=1000, 
                                                                       max_features=50000)

model = text.text_classifier('nbsvm', train_data=(x_train, y_train))
learner = ktrain.get_learner(model, train_data=(x_train, y_train), val_data=(x_test, y_test))

learner.autofit(0.01)
learner.save_model('/app/distill/models/trained_models/nbsvm')
print(learner.validate())
learner.load_model('/app/distill/models/trained_models/nbsvm')


"""
              precision    recall  f1-score   support

           0       0.86      0.79      0.82       319
           1       0.77      0.76      0.76       389
           2       0.81      0.73      0.77       394
           3       0.72      0.74      0.73       392
           4       0.83      0.84      0.84       385
           5       0.86      0.84      0.85       395
           6       0.83      0.88      0.86       390
           7       0.90      0.91      0.90       396
           8       0.96      0.94      0.95       398
           9       0.95      0.92      0.93       397
          10       0.93      0.98      0.95       399
          11       0.89      0.91      0.90       396
          12       0.79      0.80      0.79       393
          13       0.93      0.81      0.86       396
          14       0.92      0.90      0.91       394
          15       0.91      0.94      0.93       398
          16       0.78      0.91      0.84       364
          17       0.97      0.89      0.93       376
          18       0.72      0.66      0.69       310
          19       0.60      0.75      0.67       251

    accuracy                           0.85      7532
   macro avg       0.85      0.84      0.84      7532
weighted avg       0.85      0.85      0.85      7532

NOTE: 191us/step, took 1m48s overall
"""


