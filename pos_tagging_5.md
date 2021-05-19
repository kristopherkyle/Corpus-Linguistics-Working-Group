# Introduction to POS Tagging (Part 5  - More Machine Learning)
(Kristopher Kyle updated 2021-05-14)

Now that we have worked through the basics of training and testing machine learning-based part of speech (POS) tagging models in scikit-learn, we will test out some other machine learning algorithms.

See [this chapter](https://web.stanford.edu/~jurafsky/slp3/8.pdf) in an NLP book by [Jurafsky & Martin (2020)](https://web.stanford.edu/~jurafsky/slp3/) for information on other machine learning algorithms that have traditionally been used in POS taggers.

## Getting started (again)
In the last tutorial, we created a decision tree classifier that achieved 92.5% accuracy using a decision tree classifier and a small dataset. In this tutorial, we will add a couple of features to our feature set. We will also add a more precise accuracy analysis function from [Tutorial 3](pos_tagging_3.md). Finally, we will add a "real world" tagging function that will allow us to tag sample sentences to further explore the strengths/weaknesses of our taggers.

Note that a .zip file with all pre-trained models (except for the random forest model, which is 1.4 GB when extracted) can be [downloaded here](https://github.com/kristopherkyle/Corpus-Linguistics-Working-Group/raw/main/data/Tutorial5Data.zip).

The random forest model [can be downloaded here](https://github.com/kristopherkyle/Corpus-Linguistics-Working-Group/raw/main/data/clf_rf_features3.pickle.zip).

### Feature set
```python
def simple_features3(input_sent,idx,token): #takes a sentence as input (with word and tag specified), outputs a more feature-rich version
	features = {}
	features["word"] = token["word"]
	#features["pos"] = token["pos"]
	if idx == 0:
		features["prev_pos"] = "<start>" #no previous pos
		features["prev_prev_pos"] = "<start>_<start>" #no  previous pos_bg
		#features["prev_word"] = "<first_word>" #no previous word

	elif idx == 1:
		features["prev_pos"] = input_sent[idx-1]["pos"] #previos pos_tag
		features["prev_prev_pos"] = "<start>_"+ input_sent[idx-1]["pos"]
		#features["prev_word"] = input_sent[idx-1]["word"]

	else:
		features["prev_pos"] = input_sent[idx-1]["pos"] #
		features["prev_prev_pos"] = input_sent[idx-2]["pos"] + "_" + input_sent[idx-1]["pos"]

		#features["prev_word"] = input_sent[idx-1]["word"] #no previous word

	#features["prefix_tg"] = token["word"][:3] #get first three characters
	features["suffix_tg"] = token["word"][-3:] #get last two characters

	return(features)

def feature_extractor3(input_data): #takes list [sents] of lists [tokens] of dictionaries [token_features], outputs a flat list of dicts [features]
	feature_list = [] #flast list of token dictionaries
	for sent in input_data: #iterate through sentences
		for idx, token in enumerate(sent): #iterate through tokens
			feature_list.append(simple_features3(sent,idx,token)) #use simple_features function to add features
	return(feature_list)

### function for extracting only POS
def extract_pos(input_data):
	pos_list = []
	for sent in input_data:
		for token in sent:
			pos_list.append(token["pos"])
	return(pos_list)

```
### Preparing data for scikit-learn
You can download [the pre-processed data pickle here](https://github.com/kristopherkyle/Corpus-Linguistics-Working-Group/raw/main/docs/brown_sents_list.pickle), then place it in your working directory or you can go back to previous tutorials and build the data set from scratch.

```python
### Load data ###
full_data = pickle.load(open("brown_sents_list.pickle","rb"))

### Extract Features from data ###
flat_features3 = feature_extractor3(full_data)
train_features3 = flat_features3[:784443]
test_features3 = flat_features3[784443:]

### Extract POS from data (for training and accuracy checks) ###
flat_pos = extract_pos(full_data)
train_pos = flat_pos[:784443]
test_pos = flat_pos[784443:]

### vectorize predictors ###
from sklearn.feature_extraction import DictVectorizer
vec3 = DictVectorizer(sparse = True)

#we use .fit_transform() to create the vectors
train_features3_vec = vec3.fit_transform(train_features3) #vectorize sample of features

#and apply previously made vectors using .transform()
test_features3_vec = vec3.transform(test_features3)


### Turn POS Tags into numbers ###
#create our own pos dictionary
def pos_cats(pos_list):
	cat_d = {}
	for idx, x in enumerate(list(set(pos_list))):
		cat_d[x] = idx
	return(cat_d)

def convert_pos(pos_list,pos_d):
	converted = []
	for x in pos_list:
		converted.append(pos_d[x])
	return(converted)

def extract_pred_pos(pred_array,rev_d):
	predicted = []
	for x in pred_array:
		predicted.append(rev_d[x])
	return(predicted)

pos_d = pos_cats(flat_pos) #create pos to number dictionary
rev_pos_d = {value : key for (key, value) in pos_d.items()} # create number to POS dictionary for decoding output
train_pos_num = convert_pos(train_pos,pos_d) #convert training pos tags to numbers
```

If you DON'T want to run all of the full models, you can also load a pre-made vector dictionary and the POS dictionary that was used in my version of the full models below (provided in download link at the beginning of this tutorial).

```python
from sklearn.feature_extraction import DictVectorizer

### Load from pickles ###
vec3 = pickle.load(open("vec3_day5.pickle", "rb"))
pos_d = pickle.load(open("pos_num_dict.pickle", "rb"))
rev_pos_d = pickle.load(open("num_pos_dict.pickle", "rb"))

#we use .fit_transform() to create the vectors
train_features3_vec = vec3.fit_transform(train_features3) #vectorize sample of features

#and apply previously made vectors using .transform()
test_features3_vec = vec3.transform(test_features3)

#convert training pos tags to numbers
train_pos_num = convert_pos(train_pos,pos_d)
```

### Simple and Refined Accuracy
We will adapt our accuracy functions slightly to allow for a global indication of accuracy and a more fine-grained one.

```python
def pred_accuracy(pred,gold):
	c = 0
	f = 0

	for idx, x in enumerate(pred):
		#print(x,gold[idx]["pos"])
		if x == gold[idx]:
			c+=1
		else:
			f+=1
	return(c/(c+f))

#code for precision, recall, and F1 formulas
def prec_rec(accuracy_dict):
	accuracy_dict["TC"] = accuracy_dict["TP"] + accuracy_dict["FN"]
	if accuracy_dict["TP"] + accuracy_dict["FN"] == 0:
		accuracy_dict["recall"] = 0
	else:
		accuracy_dict["recall"] = accuracy_dict["TP"]/(accuracy_dict["TP"] + accuracy_dict["FN"])

	if accuracy_dict["TP"] +accuracy_dict["FP"] == 0:
		accuracy_dict["precision"] = 0
	else:
		accuracy_dict["precision"] = accuracy_dict["TP"]/(accuracy_dict["TP"] +accuracy_dict["FP"])
	if accuracy_dict["precision"] == 0 and accuracy_dict["recall"] == 0:
		accuracy_dict["f1"] = 0
	else:
		accuracy_dict["f1"] = 2 * ((accuracy_dict["precision"] * accuracy_dict["recall"])/(accuracy_dict["precision"] + accuracy_dict["recall"]))


#code to apply the above function to a full dataset
def tag_prec_rec_flat(tested,gold):
	tag_d = {}

	for idx, item in enumerate(gold):
		### convert formats, as needed: ###
		if type(item) == str:
			item = {"pos" : item}

		if type(tested[idx]) == str:
			tested[idx] = {"pos" : tested[idx]}

		### update tag dictionary as needed ###
		if item["pos"] not in tag_d:
			tag_d[item["pos"]] = {"TP":0,"FP":0,"FN":0}
		if tested[idx]["pos"] not in tag_d:
			tag_d[tested[idx]["pos"]] = {"TP":0,"FP":0,"FN":0}

		### tabulate accuracy ###
		if item["pos"] == tested[idx]["pos"]:
			tag_d[item["pos"]]["TP"] += 1
		else:
			tag_d[item["pos"]]["FN"] += 1
			tag_d[tested[idx]["pos"]]["FP"] += 1

	for x in tag_d:
		prec_rec(tag_d[x])

	return(tag_d)
```

### Tagging "real world" data
Tagging user-generated data is the most common end goal for the development of a POS tagger. Additionally, testing case sentences on in-development taggers can  help illuminate particular benefits and drawbacks of a particular model.

Tagging a featured (and vectorized) test set (particularly if information from previous words/tags are used as predictors) is a bit simpler than tagging raw, user-generated data. To tag user-generated data, we will need to complete the whole tagging process one word at a time.

This will include:
- adding predictor features
- vectorizing the feature set
- run the predictor algorithm
- convert the tag to a string
- add tag to the sentence level feature set (so we can retrieve it for the next word[s] feature set)

Below, we do this in a few lines, but most of the process happens in a single (complex) line.

The `tagger()` function takes 5 arguments:
- `model` is a trained predictor model
- `vec` is a trained vectorizer
- `rev_dict` is a number to POS tag dictionary
- `extractor` is a feature extractor function (such as the `feature_extractor3`) function outlined earlier in this tutorial
- `token_list` is a list of lists (sentences) of strings (words). If we integrated a tokenizer (and sentence tokenizer) we could just accept strings. For simplicity's sake, we haven't done so here.

```python
def tagger(model,vec,rev_dict,extractor,token_list):
	tagged = [] #for final word + tag output

	for sent in token_list:
		features = [] #in progress tagging list
		tagging = [] #final word-tag pairs

		for idx, token in enumerate(sent):
			features.append({"word" : token}) # add word to feature set
			#print(features,idx,features[idx])
			features[idx] = extractor(features,idx,features[idx]) #add featured token to features list
			#print(features)
			#print(vec.transform(features[idx]))
			#print(model.predict(vec.transform([features[idx]])))
			features[idx]["pos"] = rev_dict[model.predict(vec.transform([features[idx]]))[0]] #add pos to features - this happens one token at a time
			tagging.append({"word" : token, "pos" : features[idx]["pos"]}) #add word-tag pairs to sentence level output
		tagged.append(tagging) #add sentence to output

	return(tagged)
```

## Some models

### Decision trees
As briefly discussed in [Tutorial 4](pos_tagging_4.md), [decision trees](https://scikit-learn.org/stable/modules/tree.html) allow for a series of binary decisions to be made in a classification problem. To use the full version of the model (instead of training it on your computer) [you can download it here]().

Note that adding features in decision tree classifiers doesn't always increase accuracy. The (slightly more complex) feature set used in this tutorial is actually less accurate than the simpler one used in the last tutorial. If you don't want to take the time to train the model, you can use the pre-trained one.

```python
from sklearn import tree
clf_dt = tree.DecisionTreeClassifier()

#remember to iteratively test the time it takes to train the data:
#clf_dt = clf_dt.fit(train_features3_vec[:100],train_pos_num[:100])
#clf_dt = clf_dt.fit(train_features3_vec[:1000],train_pos_num[:1000])
#clf_dt = clf_dt.fit(train_features3_vec[:10000],train_pos_num[:10000])
#clf_dt = clf_dt.fit(train_features3_vec[:100000],train_pos_num[:100000])
clf_dt = clf_dt.fit(train_features_vec,train_pos_num)

#if you don't want to take the time to train the model yourself, you can load it here:
clf_dt = pickle.load(open("clf_dt_features3.pickle","rb"))
```

Now that we have a model, we can test its overall accuracy, its by-tag accuracy, and try some test sentences.

```python
### process training data and convert it from numbers to POS tags ###
clf_dt_pred = extract_pred_pos(clf_dt.predict(test_features3_vec), rev_pos_d)

### check simple accuracy ###
clf_dt_accs = (clf_dt_pred,test_pos) #90.98

### check by-tag accuracy ###
clf_dt_accbt = tag_prec_rec_flat(clf_dt_pred,test_pos)

#sort data by frequency ("TC")
from operator import *
clf_dt_accbt_sorted = sorted(clf_dt_accbt.items(),key=lambda x:getitem(x[1],'TC'), reverse = True)

#output F1 score of top ten most frequent tags
for x in clf_dt_accbt_sorted[:10]:
	print(x[0], x[1]["f1"])
```

```
NN 0.8650467153017753
IN 0.9617074144739762
DT 0.9852015693709351
JJ 0.7904728747260849
NNP 0.7389897095917319
, 0.9999479925109215
NNS 0.9184573002754821
. 0.9998891721157042
RB 0.8693959562978777
PRP 0.9824458557238235
```

As we see above, our model is pretty good at tagging some parts of speech, and not as good at tagging others. Lets try a few test cases to see how our tagger does on particular words in particular contexts.

```python
#create some sample data. Note that .split(" ") will turn our a string into a list of strings
sample1 = ["I am going on a run in my pleasant neighborhood right now".split(" ")]
print(tagger(clf_dt,vec3,rev_pos_d,simple_features3,sample1))
```

```
[[{'word': 'I', 'pos': 'PRP'},
  {'word': 'am', 'pos': 'VBP'},
  {'word': 'going', 'pos': 'VBG'},
  {'word': 'on', 'pos': 'IN'},
  {'word': 'a', 'pos': 'DT'},
  {'word': 'run', 'pos': 'NN'},
  {'word': 'in', 'pos': 'IN'},
  {'word': 'my', 'pos': 'PRP$'},
  {'word': 'pleasant', 'pos': 'JJ'},
  {'word': 'neighborhood', 'pos': 'NN'},
  {'word': 'right', 'pos': 'JJ'},
  {'word': 'now', 'pos': 'RB'}]]
```

```python
sample2 = ["I feel very happy about my decision to eat pizza from Papa Murphy's tonight".split(" "),"Tomorrow I will take my in-laws to Silver Falls State Park".split(" ")]
```

```
[[{'word': 'I', 'pos': 'PRP'},
  {'word': 'feel', 'pos': 'VBP'},
  {'word': 'very', 'pos': 'RB'},
  {'word': 'happy', 'pos': 'JJ'},
  {'word': 'about', 'pos': 'IN'},
  {'word': 'my', 'pos': 'PRP$'},
  {'word': 'decision', 'pos': 'NN'},
  {'word': 'to', 'pos': 'TO'},
  {'word': 'eat', 'pos': 'VB'},
  {'word': 'pizza', 'pos': 'NN'},
  {'word': 'from', 'pos': 'IN'},
  {'word': 'Papa', 'pos': 'NNP'},
  {'word': "Murphy's", 'pos': 'NNP'},
  {'word': 'tonight', 'pos': 'VBD'}],
 [{'word': 'Tomorrow', 'pos': 'RB'},
  {'word': 'I', 'pos': 'PRP'},
  {'word': 'will', 'pos': 'MD'},
  {'word': 'take', 'pos': 'VB'},
  {'word': 'my', 'pos': 'PRP$'},
  {'word': 'in-laws', 'pos': 'NNS'},
  {'word': 'to', 'pos': 'TO'},
  {'word': 'Silver', 'pos': 'VB'},
  {'word': 'Falls', 'pos': 'NNS'},
  {'word': 'State', 'pos': 'VB'},
  {'word': 'Park', 'pos': 'NN'}]]
```

### Random forest
One commonly employed variation on the decision tree is the [random forest]. In short, [random forests](https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees) are an [ensemble method](https://scikit-learn.org/stable/modules/ensemble.html) wherein decision trees are optimized on random samples of a training set instead of the entire dataset. There are a number of methods for [tuning the random forest parameters](https://scikit-learn.org/stable/modules/ensemble.html#parameters), such as how many random samples should be used, and how many (randomly selected) predictors should be used. To keep things simple, we will use the default values below.

```python
from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier(n_estimators=10)

#clf_rf = clf_rf.fit(train_features3_vec[:100],train_pos_num[:100])
#clf_rf = clf_rf.fit(train_features3_vec[:1000],train_pos_num[:1000])
#clf_rf = clf_rf.fit(train_features3_vec[:10000],train_pos_num[:10000])
#clf_rf = clf_rf.fit(train_features3_vec[:100000],train_pos_num[:100000])
clf_rf = clf_rf.fit(train_features3_vec,train_pos_num)

#if you don't want to take the time to train the model yourself, you can load it here:
clf_rf = pickle.load(open("clf_rf_features3.pickle","rb"))
```

Now we can check the accuracy and test some sentences:

```python
### process training data and convert it from numbers to POS tags ###
clf_rf_pred = extract_pred_pos(clf_rf.predict(test_features3_vec), rev_pos_d)

### check simple accuracy ###
clf_rf_accs = (clf_rf_pred,test_pos) #90.6%

### check by-tag accuracy ###
clf_rf_accbt = tag_prec_rec_flat(clf_rf_pred,test_pos)

#sort data by frequency ("TC")
from operator import *
clf_rf_accbt_sorted = sorted(clf_rf_accbt.items(),key=lambda x:getitem(x[1],'TC'), reverse = True)

#output F1 score of top ten most frequent tags
for x in clf_rf_accbt_sorted[:10]:
	print(x[0], x[1]["f1"])
```

```
NN 0.8516446511794974
IN 0.960549645390071
DT 0.9847889249700905
JJ 0.7775759577278732
NNP 0.7247111046778124
, 0.9999219907949137
NNS 0.9146615511098931
. 0.9999168813897432
RB 0.864562401459177
PRP 0.9829504782976509
```

```python
print(tagger(clf_rf,vec3,rev_pos_d,simple_features3,sample1))
```

```
[[{'word': 'I', 'pos': 'PRP'},
  {'word': 'am', 'pos': 'VBP'},
  {'word': 'going', 'pos': 'VBG'},
  {'word': 'on', 'pos': 'IN'},
  {'word': 'a', 'pos': 'DT'},
  {'word': 'run', 'pos': 'NN'},
  {'word': 'in', 'pos': 'IN'},
  {'word': 'my', 'pos': 'PRP$'},
  {'word': 'pleasant', 'pos': 'JJ'},
  {'word': 'neighborhood', 'pos': 'NN'},
  {'word': 'right', 'pos': 'JJ'},
  {'word': 'now', 'pos': 'RB'}]]
```

```python
print(tagger(clf_rf,vec3,rev_pos_d,simple_features3,sample2))
```

```
[[{'word': 'I', 'pos': 'PRP'},
  {'word': 'feel', 'pos': 'VBP'},
  {'word': 'very', 'pos': 'RB'},
  {'word': 'happy', 'pos': 'JJ'},
  {'word': 'about', 'pos': 'IN'},
  {'word': 'my', 'pos': 'PRP$'},
  {'word': 'decision', 'pos': 'NN'},
  {'word': 'to', 'pos': 'TO'},
  {'word': 'eat', 'pos': 'VB'},
  {'word': 'pizza', 'pos': 'NN'},
  {'word': 'from', 'pos': 'IN'},
  {'word': 'Papa', 'pos': 'NNP'},
  {'word': "Murphy's", 'pos': 'NNP'},
  {'word': 'tonight', 'pos': 'NNP'}],
 [{'word': 'Tomorrow', 'pos': 'RB'},
  {'word': 'I', 'pos': 'PRP'},
  {'word': 'will', 'pos': 'MD'},
  {'word': 'take', 'pos': 'VB'},
  {'word': 'my', 'pos': 'PRP$'},
  {'word': 'in-laws', 'pos': 'NN'},
  {'word': 'to', 'pos': 'TO'},
  {'word': 'Silver', 'pos': 'VB'},
  {'word': 'Falls', 'pos': 'NNS'},
  {'word': 'State', 'pos': 'JJ'},
  {'word': 'Park', 'pos': 'NN'}]]
```

### Maximum Entropy (Multinomial Logistic Regression)
[Maximum Entropy (Multinomial Logistic Regression)](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression) is another fairly commonly used POS predictor model. One advantage it has over decision trees is (under many circumstances) better at using a larger set of predictors (and can use them at the same time, unlike a decision tree, which must make a number of binary decisions). Below, we employ a MaxEnt model on our feature set.

Note that we use a parameters `solver` and `maxiter` to help the model run. These are not necessarily optimized - [see here for more information](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression).

```python
from sklearn.linear_model import LogisticRegression
# clf_mxe = LogisticRegression(random_state=0,solver='saga',max_iter=1000).fit(train_features3_vec[:1000],train_pos_num[:1000])
# clf_mxe = LogisticRegression(random_state=0,solver='saga',max_iter=1000).fit(train_features3_vec[:10000],train_pos_num[:10000])
# clf_mxe = LogisticRegression(random_state=0,solver='saga',max_iter=1000).fit(train_features3_vec[:100000],train_pos_num[:100000])
clf_mxe = LogisticRegression(random_state=0,solver='saga',max_iter=1000).fit(train_features3_vec,train_pos_num)

#if you don't want to take the time to train the model yourself, you can load it here:
clf_mxe = pickle.load(open("clf_mxe_features3.pickle","rb"))

```

Now we can check the accuracy and test some sentences:

```python
### process training data and convert it from numbers to POS tags ###
clf_mxe_pred = extract_pred_pos(clf_mxe.predict(test_features3_vec), rev_pos_d)

### check simple accuracy ###
clf_mxe_accs = (clf_mxe_pred,test_pos) #94.25%

### check by-tag accuracy ###
clf_mxe_accbt = tag_prec_rec_flat(clf_mxe_pred,test_pos)

#sort data by frequency ("TC")
from operator import *
clf_mxe_accbt_sorted = sorted(clf_mxe_accbt.items(),key=lambda x:getitem(x[1],'TC'), reverse = True)

#output F1 score of top ten most frequent tags
for x in clf_mxe_accbt_sorted[:10]:
	print(x[0], x[1]["f1"])
```
```
NN 0.9192649564776861
IN 0.96795577676959
DT 0.986210338306367
JJ 0.8734826115485566
NNP 0.8474950134929016
, 0.9999479925109215
NNS 0.9582309582309583
. 1.0
RB 0.9056300599468513
PRP 0.9842024689749519
```

```python
#create some sample data. Note that .split(" ") will turn our a string into a list of strings
sample1 = ["I am going on a run in my pleasant neighborhood right now".split(" ")]
print(tagger(clf_mxe,vec3,rev_pos_d,simple_features3,sample1))
```

```
[[{'word': 'I', 'pos': 'PRP'},
  {'word': 'am', 'pos': 'VBP'},
  {'word': 'going', 'pos': 'VBG'},
  {'word': 'on', 'pos': 'IN'},
  {'word': 'a', 'pos': 'DT'},
  {'word': 'run', 'pos': 'NN'},
  {'word': 'in', 'pos': 'IN'},
  {'word': 'my', 'pos': 'PRP$'},
  {'word': 'pleasant', 'pos': 'JJ'},
  {'word': 'neighborhood', 'pos': 'NN'},
  {'word': 'right', 'pos': 'NN'},
  {'word': 'now', 'pos': 'RB'}]]
```

```python
sample2 = ["I feel very happy about my decision to eat pizza from Papa Murphy's tonight".split(" "),"Tomorrow I will take my in-laws to Silver Falls State Park".split(" ")]
print(tagger(clf_mxe,vec3,rev_pos_d,simple_features3,sample2))
```

```
[[{'word': 'I', 'pos': 'PRP'},
  {'word': 'feel', 'pos': 'VBP'},
  {'word': 'very', 'pos': 'RB'},
  {'word': 'happy', 'pos': 'JJ'},
  {'word': 'about', 'pos': 'IN'},
  {'word': 'my', 'pos': 'PRP$'},
  {'word': 'decision', 'pos': 'NN'},
  {'word': 'to', 'pos': 'TO'},
  {'word': 'eat', 'pos': 'VB'},
  {'word': 'pizza', 'pos': 'NN'},
  {'word': 'from', 'pos': 'IN'},
  {'word': 'Papa', 'pos': 'NNP'},
  {'word': "Murphy's", 'pos': 'NNP'},
  {'word': 'tonight', 'pos': 'RB'}],
 [{'word': 'Tomorrow', 'pos': 'RB'},
  {'word': 'I', 'pos': 'PRP'},
  {'word': 'will', 'pos': 'MD'},
  {'word': 'take', 'pos': 'VB'},
  {'word': 'my', 'pos': 'PRP$'},
  {'word': 'in-laws', 'pos': 'NNS'},
  {'word': 'to', 'pos': 'TO'},
  {'word': 'Silver', 'pos': 'VB'},
  {'word': 'Falls', 'pos': 'NNS'},
  {'word': 'State', 'pos': 'NNP'},
  {'word': 'Park', 'pos': 'NNP'}]]
```
## Building better taggers
So far, our best tagger achieves 94.5% accuracy on unseen data, which is not bad at all!

To increase tagging accuracy, we need to a) optimize our feature set and b) tune our machine learning parameters and/or use a different ML model. Regardless of the model use, however, our feature set will likely need to be optimized in order to reach 97-98% accuracy.
