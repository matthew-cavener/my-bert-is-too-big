# my-bert-is-too-big
Doing Knowledge Distillation on BERT because the inference time is too damn high!

Using a small model to learn the function that a very large, complicated one learned by getting the nuance of the dataset for free, instead of having to learn it on its own.


# Motivation
BERT inference takes too long. I'm curious if I can reduce inference time while maintaining (or maybe improving???) performance.

Inspired by this talk on "Dark Knowledge": `https://www.youtube.com/watch?v=EK61htlw8hY`

"Physicists will tell you 95% of the stuff in the universe [dark matter, dark energy] is not what they were paying attention to, its something else. 95% of what a neural net learns is not what you were trying to get it to learn, and not what you were testing it on."

"You are trying to learn different painters, you could have a teacher who's just like "Oh, that's a ruebans, thats a kasitzsky, that's a so and so. And that'd be very helpful. But it'd be much better to have a teacher who says, "Oh, that's a reubans, but its a very odd reubans in that you might almost mistake it for a Tischner. That would be enormously helpful information to learning about painters."

"The training data says this is a dog. Once you've trained up your big model it's going to say: "Probability .9 its a dog, .1 its a cat, 1e-6 its a cow, 1e-9 its a car... things with small probabilities are irrelevent.. it might as well have been 0... that's what I call Dark Knowledge. One thing you won't see in particular is that car is 1000 times less likely than cow, that's where almost all the knowledge is. Almost everything it knows is about the relative probabilities of things its almost certain are wrong."

"The way the big net is generalizing, information about how to generalize is what is showing up in this dark knowledge. It is telling you "You shouldn't generalize from cow to car because cow is 1000 time more likely than car. if cow was e-6 and and horse was e-7 it would be telling you "Maybe you should be genralizing between cows and horses, they're both pretty similar bets here. So there's lots of information about how to generalize in those soft targets, and none of that information is in the hard targets. So what the big model has really done, if you take the soft targets, is its learned what's like what, and it is telling the little one what's like what, and that is making life much easier when it comes time to training."

"The net your transfer the knowledge to does worse than the big model so what's the point? The point is a.) it's an amazing effect and b.) it'll allow you to have a very little production model."

And this talk on Knowledge Distillation: `https://blog.feedly.com/nlp-breakfast-8-knowledge-distillation/`

Take aways of those are basically that the big network learns much richer information than the sort of one-hot encoded data fed it.
In the MNIST example, a big network can learn on its own that a 1 and a 7 are similar, but it also has a bunch of (Hinton argues mostly) so called "Dark Knowledge"
That is, that the probability of it being an 8 is 1000 times less than that of it being a 4, but while that isn't info we really care about, the network worked very hard to learn that, and at inference time works very hard to determine that realtive probability.

So to speed things up we get this nice big complicated model to learn these nuances, and then use it to label some data, producing "soft targets".
Ideally, you'd use the logits, but you can also "temperature scale" the softmax. (Hence the term distillation)

Anyway, now you have these predictions (again, using the MNIST `1` example) from your big model like `[.001, .791, 1e-7, 1e-6, 1e-7, 1e-6, 1e-4, .210, 1e-4, 1e-5]` and instead create these soft targets from the temp scaled softmax: `[~0, .6, ~0, ~0, ~0, ~0, ~0, .4, ~0, ~0]` which is a much richer than the original `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`, and avoids having to learn "Dark Knowledge" we don't actually care about, while preserving the sorts of similarity we do.
It encodes those similarities that the big network had to work so hard to learn, and a gives them to a smaller network for free. So we can get similar results with a much smaller network, and more importantly, much faster inference time, since it doesn't have to learn these nuances for itself.


It's basically the difference between having to discover calculus for yourself and learning it from a textbook. Someone else already did the hard work figuring everything out, you just need to know how to use it.

# Work to do for experimentation
1.) Fine Tune a BERT model for some Multiclass classification problem (in progress, sub set of toxic comments dataset)

2.) Train some simpler classifiers on that same dataset to create a baseline.

3.) "Create" new dataset by having FTed BERT soft label the text from the training dataset.

4.) Train the simpler classifiers on the soft labelled dataset.

5.) Take validation set that none of the models have seen and compare performance (including inference time) of: BERT, Baseline simples, Soft labelled simples.

6.) (Maybe) see how an ensamble of simpler methods does compared to BERT, while keeping inference time ~1/100th.

7.) Set up nice webserver that compares input text to BERT and Simpler Models, compare inference time, do cosine similarity of output probabilities to get quick idea of how close the simple ones are to BERT.

Later should use FTed BERT logits as soft targets, should implement annealing, should use BiLSTM as one of the Simple Models.

# Other references
ModelCompression (Bucila et al, 2006)
`https://www.cs.cornell.edu/~caruana/compression.kdd06.pdf`

Distilling the Knowledge in a Neural Network (Hinton et al, 2015)
`https://arxiv.org/abs/1503.02531`

Distilling Task-Specific Knowledge from BERT into Simple Neural Networks (Tang et al, 2019):
`https://arxiv.org/abs/1903.12136`

Born Again Neural Networks (Furlanello et al, 2018):
`https://arxiv.org/abs/1805.04770`

BAM! Born-Again Multi-Task Networks for Natural Language Understanding (Clark et al, 2019):
`https://arxiv.org/abs/1907.04829`

Awesome Knowledge Distillation (Contains most of the above as well)
`https://github.com/dkozlov/awesome-knowledge-distillation`

Distilling BERT — How to achieve BERT performance using Logistic Regression:
`https://towardsdatascience.com/distilling-bert-how-to-achieve-bert-performance-using-logistic-regression-69a7fc14249d`

🏎 Smaller, faster, cheaper, lighter: Introducing DistilBERT, a distilled version of BERT:
`https://medium.com/huggingface/distilbert-8cf3380435b5`

