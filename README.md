# my-bert-is-too-big
Doing Knowledge Distillation on BERT because the inference time is too damn high!

Using a small model to learn the function that a very large, complicated one learned by getting the nuance of the dataset for free (given by the large complicated "teacher"), instead of having to learn it on its own.


# Motivation
BERT inference takes too long. I'm curious if I can reduce inference time while maintaining (or maybe improving???) performance.

Inspired by this talk on "Dark Knowledge": https://www.youtube.com/watch?v=EK61htlw8hY

"Physicists will tell you 95% of the stuff in the universe [dark matter, dark energy] is not what they were paying attention to, its something else. 95% of what a neural net learns is not what you were trying to get it to learn, and not what you were testing it on."

"You are trying to learn different painters, you could have a teacher who's just like "Oh, that's a ruebans, thats a kasitzsky, that's a so and so. And that'd be very helpful. But it'd be much better to have a teacher who says, "Oh, that's a reubans, but its a very odd reubans in that you might almost mistake it for a Tischner. That would be enormously helpful information to learning about painters."

"The training data says this is a dog. Once you've trained up your big model it's going to say: "Probability .9 its a dog, .1 its a cat, 1e-6 its a cow, 1e-9 its a car... things with small probabilities are irrelevent.. it might as well have been 0... that's what I call Dark Knowledge. One thing you won't see in particular is that car is 1000 times less likely than cow, that's where almost all the knowledge is. Almost everything it knows is about the relative probabilities of things its almost certain are wrong."

"The way the big net is generalizing, information about how to generalize is what is showing up in this dark knowledge. It is telling you "You shouldn't generalize from cow to car because cow is 1000 time more likely than car. if cow was e-6 and and horse was e-7 it would be telling you "Maybe you should be genralizing between cows and horses, they're both pretty similar bets here. So there's lots of information about how to generalize in those soft targets, and none of that information is in the hard targets. So what the big model has really done, if you take the soft targets, is its learned what's like what, and it is telling the little one what's like what, and that is making life much easier when it comes time to training."

"The net your transfer the knowledge to does worse than the big model so what's the point? The point is a.) it's an amazing effect and b.) it'll allow you to have a very little production model."

And this talk on Knowledge Distillation: https://blog.feedly.com/nlp-breakfast-8-knowledge-distillation/

Take aways of those are basically that the big network learns much richer information than the sort of one-hot encoded data fed it.
In the MNIST example, a big network can learn on its own that a 1 and a 7 are similar, but it also has a bunch of (Hinton argues mostly) so called "Dark Knowledge"

That is, the network knows a great deal about generalization between related classes, even among unrealted prediction, and we don't necessarily need a network that "has" that knowledge, so much as one that can use it.

# Experimental results
So I ended up not using BERT because training was taking entirely too long, so I used an NBSVM.
I used linear regression as my simple model because anything more complicated had about the same accuracy (~82-84).
I used the 20 news groups dataset because the actual problem I'm concerned with is in the same vein (which class does this text belong to?)
The results were... disappointing to say the least. Given the data presented in the literature and other implementations I'm inclined to say my results are due to not using a model (like BERT) that would achieve much higher accuracy, and a datasset that is both large and "easy".
Results below:

```
NBSVM============================
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
```
```
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
```
```
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
```

# Work to do for experimentation
1.) Train some simpler classifiers on some text classification problem to create a baseline.

2.) Fine Tune a BERT model for that same dataset.

3.) "Create" new dataset by having FTed BERT soft label the text from the training dataset.

4.) Train the simpler classifiers on the soft labelled dataset.

5.) Compare accuracies of BERT, baseline, and soft label learners.

6.) (Maybe) see how an ensamble of simpler methods does compared to BERT, while keeping inference time ~1/100th.

7.) Set up nice webserver that compares input text to BERT and Simpler Models, compare inference time, do cosine similarity of output probabilities to get quick idea of how close the simple ones are to BERT.

Later should use FTed BERT logits as soft targets, should implement annealing, should use BiLSTM as one of the Simple Models.

# Other references
ModelCompression (Bucila et al, 2006)
https://www.cs.cornell.edu/~caruana/compression.kdd06.pdf

Distilling the Knowledge in a Neural Network (Hinton et al, 2015)
https://arxiv.org/abs/1503.02531

Distilling Task-Specific Knowledge from BERT into Simple Neural Networks (Tang et al, 2019):
https://arxiv.org/abs/1903.12136

Born Again Neural Networks (Furlanello et al, 2018):
https://arxiv.org/abs/1805.04770

BAM! Born-Again Multi-Task Networks for Natural Language Understanding (Clark et al, 2019):
https://arxiv.org/abs/1907.04829

Awesome Knowledge Distillation (Contains most of the above as well)
https://github.com/dkozlov/awesome-knowledge-distillation

Distilling BERT — How to achieve BERT performance using Logistic Regression:
https://towardsdatascience.com/distilling-bert-how-to-achieve-bert-performance-using-logistic-regression-69a7fc14249d

🏎 Smaller, faster, cheaper, lighter: Introducing DistilBERT, a distilled version of BERT:
https://medium.com/huggingface/distilbert-8cf3380435b5

