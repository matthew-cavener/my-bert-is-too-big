import ktrain
from ktrain import text

from sklearn.datasets import fetch_20newsgroups

train_b = fetch_20newsgroups(
    subset='train',
    shuffle=True,
    random_state=42,
    data_home='/app/distill/data'
)
test_b = fetch_20newsgroups(
    subset='test',
    shuffle=True,
    random_state=42,
    data_home='/app/distill/data'
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
                                                                       ngram_range=1, 
                                                                       maxlen=350, 
                                                                       max_features=35000)

model = text.text_classifier('nbsvm', train_data=(x_train, y_train))
learner = ktrain.get_learner(model, train_data=(x_train, y_train), val_data=(x_test, y_test))
learner.lr_find()

learner.autofit(0.01)
print(learner.validate())

"""
Weights from best epoch have been loaded into model.
              precision    recall  f1-score   support

           0       0.78      0.74      0.76       319
           1       0.74      0.79      0.76       389
           2       0.76      0.73      0.75       394
           3       0.72      0.73      0.73       392
           4       0.85      0.87      0.86       385
           5       0.86      0.80      0.83       395
           6       0.83      0.86      0.85       390
           7       0.91      0.90      0.91       396
           8       0.95      0.94      0.95       398
           9       0.95      0.94      0.94       397
          10       0.93      0.97      0.95       399
          11       0.88      0.92      0.90       396
          12       0.78      0.78      0.78       393
          13       0.92      0.84      0.88       396
          14       0.89      0.92      0.91       394
          15       0.86      0.90      0.88       398
          16       0.73      0.90      0.80       364
          17       0.96      0.87      0.91       376
          18       0.75      0.60      0.67       310
          19       0.62      0.60      0.61       251

    accuracy                           0.84      7532
   macro avg       0.83      0.83      0.83      7532
weighted avg       0.84      0.84      0.84      7532

NOTE: 191us/step, took 44s overall
"""


