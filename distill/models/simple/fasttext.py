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

from keras.models import Sequential
from keras.layers import Dense, Embedding, GlobalAveragePooling1D
def get_model():
    model = Sequential()
    model.add(Embedding(134142+1, 50, input_length=350)) # add 1 for padding token
    model.add(GlobalAveragePooling1D())
    model.add(Dense(20, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
model = get_model()

# model = text.text_classifier('fasttext', train_data=(x_train, y_train))
learner = ktrain.get_learner(model, train_data=(x_train, y_train), val_data=(x_test, y_test))
learner.lr_find()

learner.autofit(0.01)
print(learner.validate())

"""
Weights from best epoch have been loaded into model.
              precision    recall  f1-score   support

           0       0.75      0.77      0.76       319
           1       0.75      0.79      0.77       389
           2       0.81      0.70      0.75       394
           3       0.73      0.73      0.73       392
           4       0.78      0.86      0.82       385
           5       0.87      0.80      0.83       395
           6       0.82      0.89      0.85       390
           7       0.82      0.92      0.87       396
           8       0.99      0.90      0.94       398
           9       0.91      0.94      0.93       397
          10       0.97      0.95      0.96       399
          11       0.92      0.90      0.91       396
          12       0.75      0.80      0.77       393
          13       0.90      0.85      0.87       396
          14       0.92      0.90      0.91       394
          15       0.85      0.88      0.86       398
          16       0.74      0.87      0.80       364
          17       0.97      0.84      0.90       376
          18       0.73      0.59      0.65       310
          19       0.60      0.64      0.62       251

    accuracy                           0.83      7532
   macro avg       0.83      0.83      0.83      7532
weighted avg       0.84      0.83      0.83      7532

NOTE: 2ms/step, took 8m 59s overall
"""
