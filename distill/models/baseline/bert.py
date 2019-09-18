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
                                                                       preprocess_mode='bert',
                                                                       ngram_range=1, 
                                                                       maxlen=400, 
                                                                       max_features=35000)

model = text.text_classifier('bert', train_data=(x_train, y_train))
learner = ktrain.get_learner(model, train_data=(x_train, y_train), batch_size=6)
learner.autofit(2e-5, 5)

learner.save_model('/app/distill/models/trained_models/bert')
learner.load_model('/app/distill/models/trained_models/bert')
print(learner.validate(val_data=(x_test, y_test), class_names=train_b.target_names))

"""
                          precision    recall  f1-score   support

             alt.atheism       0.72      0.72      0.72       319
           comp.graphics       0.84      0.79      0.81       389
 comp.os.ms-windows.misc       0.83      0.81      0.82       394
comp.sys.ibm.pc.hardware       0.72      0.77      0.74       392
   comp.sys.mac.hardware       0.82      0.85      0.84       385
          comp.windows.x       0.92      0.86      0.89       395
            misc.forsale       0.90      0.92      0.91       390
               rec.autos       0.90      0.90      0.90       396
         rec.motorcycles       0.95      0.86      0.90       398
      rec.sport.baseball       0.96      0.94      0.95       397
        rec.sport.hockey       0.97      0.97      0.97       399
               sci.crypt       0.95      0.90      0.93       396
         sci.electronics       0.82      0.83      0.83       393
                 sci.med       0.90      0.95      0.93       396
               sci.space       0.95      0.93      0.94       394
  soc.religion.christian       0.82      0.96      0.88       398
      talk.politics.guns       0.70      0.82      0.76       364
   talk.politics.mideast       0.93      0.92      0.93       376
      talk.politics.misc       0.69      0.58      0.63       310
      talk.religion.misc       0.66      0.55      0.60       251

                accuracy                           0.85      7532
               macro avg       0.85      0.84      0.84      7532
            weighted avg       0.85      0.85      0.85      7532

[[229   0   0   0   0   0   0   0   0   0   0   0   0   4   5  22   1  14
    3  41]
 [  0 309  16  13   3  18   4   1   1   0   1  13   3   4   0   1   0   2
    0   0]
 [  0  19 321  41   4   7   0   0   0   0   0   0   1   1   0   0   0   0
    0   0]
 [  0   6  19 303  40   1   9   0   0   0   0   0  13   0   0   1   0   0
    0   0]
 [  0   4   5  26 329   0   9   0   0   0   0   0  12   0   0   0   0   0
    0   0]
 [  1  16  25   5   3 340   3   0   0   0   0   1   0   0   0   0   1   0
    0   0]
 [  0   0   0  11   3   0 360   5   3   0   0   1   5   1   1   0   0   0
    0   0]
 [  0   1   0   0   1   0   8 355   8   1   0   1  13   1   0   0   0   0
    7   0]
 [  1   1   0   0   1   0   3  23 341   0   0   0  12   2   0   0   7   0
    7   0]
 [  1   1   0   0   0   0   2   1   3 374   9   0   0   1   0   0   2   0
    3   0]
 [  0   0   0   0   0   0   0   1   0   8 388   0   1   1   0   0   0   0
    0   0]
 [  0   1   0   4   0   3   0   1   0   3   0 358   8   0   0   0   6   0
   11   1]
 [  0   3   0  20  14   1   4   6   1   1   0   0 327  15   1   0   0   0
    0   0]
 [  2   2   0   0   1   1   0   0   0   0   0   0   1 376   0   8   2   0
    3   0]
 [  8   7   0   0   1   0   0   0   0   0   1   0   3   4 368   0   0   0
    2   0]
 [  7   0   1   0   0   0   0   0   0   0   0   0   0   2   0 382   0   1
    0   5]
 [  5   0   0   0   0   0   0   1   0   1   0   2   0   0   1   0 300   4
   35  15]
 [ 12   0   0   0   1   0   0   0   1   1   1   0   0   0   0   1   2 347
   10   0]
 [  4   0   0   0   0   0   0   0   0   0   0   1   0   4   7   2 100   1
  181  10]
 [ 46   0   0   0   0   0   0   0   1   0   0   0   0   0   5  50   5   3
    2 139]]
"""
