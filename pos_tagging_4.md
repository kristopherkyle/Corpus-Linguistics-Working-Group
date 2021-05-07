# Introduction to POS Tagging (Part 4  - Machine Learning)
(Kristopher Kyle - Updated 2021-05-07)

## Getting started with machine learning
Now that we have sufficient background in feature selection and a basic understanding of how POS taggers work, we can move closer to our goal of a state of the art POS tagger by using machine learning techniques in our models. There are a variety of machine learning techniques that we can employ - they only criteria is that they must allow for categorical predictions to be made (i.e., POS tags) based on the occurrence of categorical predictors (and their frequency of occurrence). In this tutorial, we will start by using [scikit-learn](https://scikit-learn.org/stable/index.html) to create a tagger using a decision tree. Most of the effort in this tutorial will be focused on formatting our data to be used with scikit-learn, but once we do that we will have access to a number of machine learning models.

## Decision Trees
Decision trees are a simple (but reasonably powerful) machine learning technique that have been used in POS taggers for some time. Schmid's [TreeTagger](https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/), for example, has been in use since the mid-1990's (corpus linguists still use it today to tag texts in a wide range of languages). In a classic decision tree, a series of binary decisions are made that minimize prediction errors. See [scikit-learn's explanation of decision trees for more information on how they work.](https://scikit-learn.org/stable/modules/tree.html)

## Loading and formatting our data

For this tutorial, we will load our data again, but this time we will load it from a python pickle. You can [download the pickle here](https://github.com/kristopherkyle/Corpus-Linguistics-Working-Group/raw/main/docs/brown_sents_list.pickle), then place it in your working directory.

```python
import pickle
full_data = pickle.load(open("brown_sents_list.pickle","rb"))
```

### Formatting our data for scikit-learn and extracting features
To train our model in scikit-learn, we need to feed it a list of tokens instead of a list of lists (sentences) of tokens (see previous tutorials to brush up on how our data was formatted). However, it is likely useful for our prediction features to be sensitive to a word's position in a sentence, so we will first add our features, and then flatten our lists.

For this tutorial, we will use a very simple set of prediction features. As we add more features, our model will take longer to train, so we will start small and leave a lot of room for model improvement. For now, our feature set will include a token's suffix (operationalized as the last three characters in a token) and the POS tag of the preceding word.

```python
def simple_features(input_sent,idx,token): #takes a sentence as input (with word and tag specified), outputs a more feature-rich version
	features = {}
	#features["word"] = token["word"]
	if idx == 0:
		features["prev_pos"] = "<start>" #no previous pos

	elif idx == 1:
		features["prev_pos"] = input_sent[idx-1]["pos"] #previos pos_tag

	else:
		features["prev_pos"] = input_sent[idx-1]["pos"] #

	features["suffix_tg"] = token["word"][-3:] #get last three characters

	return(features)
```

We will now write a function that extracts the features from a list of lists (sentences) of lists (tokens) and outputs a flattened list of dictionaries (token features). We will write functions that will output a flattened list of words and a flattened list of POS tags.

```python
def feature_extractor(input_data): #takes list [sents] of lists [tokens] of dictionaries [token_features], outputs a flat list of dicts [features]
	feature_list = [] #flast list of token dictionaries
	for sent in input_data: #iterate through sentences
		for idx, token in enumerate(sent): #iterate through tokens
			feature_list.append(simple_features(sent,idx,token)) #use simple_features function to add features
	return(feature_list)

def extract_pos(input_data):
	pos_list = []
	for sent in input_data:
		for token in sent:
			pos_list.append(token["pos"])
	return(pos_list)

def extract_words(input_data):
	word_list = []
	for sent in input_data:
		for token in sent:
			word_list.append(token["word"])
	return(word_list)

flat_words = extract_words(full_data)
flat_pos = extract_pos(full_data)
flat_features = feature_extractor(full_data)
```

If we want access the word, tag, and feature set for a token, we can use list index numbers from each of our lists:

```python
for idx, x in enumerate(flat_words[:10]):
	print(x, flat_pos[idx],flat_features[idx])
```

```
In IN {'prev_pos': '<start>', 'suffix_tg': 'In'}
tradition NN {'prev_pos': 'IN', 'suffix_tg': 'ion'}
and CC {'prev_pos': 'NN', 'suffix_tg': 'and'}
in IN {'prev_pos': 'CC', 'suffix_tg': 'in'}
poetry NN {'prev_pos': 'IN', 'suffix_tg': 'try'}
, , {'prev_pos': 'NN', 'suffix_tg': ','}
the DT {'prev_pos': ',', 'suffix_tg': 'the'}
marriage NN {'prev_pos': 'DT', 'suffix_tg': 'age'}
bed NN {'prev_pos': 'NN', 'suffix_tg': 'bed'}
is VBZ {'prev_pos': 'NN', 'suffix_tg': 'is'}
```

### Training and test sets
Previously, we made training and test sets based on sentences. We could do that here as well, but for simplicity we will make our training and test sets based on tokens. We will also use a shortcut and use slices to chose our sets instead of a random number generator.

```python
len(flat_words) * .67 #get size of training data: 784,443.37

#training data
train_words = flat_words[:784443]
train_pos = flat_pos[:784443]
train_features = flat_features[:784443]

#test data
test_words = flat_words[784443:]
test_pos = flat_pos[784443:]
test_features = flat_features[784443:]
```

### Turning categorical variables into numbers
The largest methodological hurdle to using scikit-learn for POS tagging is that scikit-learn expects variables to be numerical. Fortunately, scikit-learn has a built-in function to deal with this - it can turn a list of feature-set dictionaries and convert it to a vector-based representation.

We will also need to turn our POS tag predictions into numbers - we will do that below as well.

```python
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse = True)

#transform categorical variables to vectors

#we use .fit_transform() to create the vectors
train_features_vec = vec.fit_transform(train_features) #vectorize sample of features

#and apply previously made vectors using .transform()
test_features_vec = vec.transform(test_features)

#create our own POS conversion dictionary
def pos_cats(pos_list):
	cat_d = {}
	for idx, x in enumerate(list(set(pos_list))):
		cat_d[x] = idx
	return(cat_d)

pos_d = pos_cats(flat_pos)
#print(pos_d)

def convert_pos(pos_list,pos_d):
	converted = []
	for x in pos_list:
		converted.append(pos_d[x])
	return(converted)

train_pos_num = convert_pos(train_pos,pos_d)
```

## Creating and evaluating a decision tree model in sci-kit learn
Now that our data is properly formatted, we can train a decision tree model in sci-kit learn.

```python
from sklearn import tree #import decision tree module

clf = tree.DecisionTreeClassifier() #create classifier
clf = clf.fit(train_features_vec,train_pos_num) #train model (features, pos tags)

pred1 = clf.predict(test_features_vec) #apply model to new data

print(pred1[:10]) #print first ten items in pred1
```

```
 array([83, 11, 15, 44, 54, 34, 79, 43, 54, 53])
```

We now have predicted POS tags for our test data, but they are in numerical format, so we need to convert them back to strings.

```python
#revers \e the dictionary using a dictionary comprehension
rev_pos_d = {value : key for (key, value) in pos_d.items()}

def extract_pred_pos(pred_array,rev_d):
	predicted = []
	for x in pred_array:
		predicted.append(rev_d[x])
	return(predicted)

pred1_pos = extract_pred_pos(pred1,rev_pos_d)
pred1_pos[:10]
```
```
['PRP$', 'VBZ', 'RB', 'VB', 'IN', ',', 'DT', 'CD', 'IN', 'PRP']
```

### Checking the accuracy
Now we will check the overall accuracy of the model.

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

pred_accuracy(pred1_pos,test_pos) #0.8582905416597648
```
As we can see, we get a reasonably accurate model using a relatively poor set of predictors!

Now, we need to apply our advanced accuracy analysis skills to see the strengths and weaknesses of our tagger and add features to our set!





```python
def accuracy_cmplx_sent(gold,tested,acc_dict,feature_d):
	for idx, item in enumerate(gold):
		if item["pos"] == tested[idx]["pos"]:
			acc_dict["correct"] += 1
			if item["word"] in feature_d:
				acc_dict["known_correct"] += 1
			else:
				acc_dict["unknown_correct"] += 1

		else:
			acc_dict["incorrect"] += 1
			if item["word"] in feature_d:
				acc_dict["known_incorrect"] += 1
			else:
				acc_dict["unknown_incorrect"] += 1

def accuracy_cmplx_doc(gold,tested,feature_d):
	acc_dict = {"correct" : 0, "incorrect" : 0,"known_correct" : 0,"known_incorrect" : 0, "unknown_correct" : 0,"unknown_incorrect" : 0}

	for idx, item in enumerate(gold):
		accuracy_cmplx_sent(item,tested[idx],acc_dict,feature_d)

	acc_dict["total_acc"] = acc_dict["correct"]/(acc_dict["correct"] + acc_dict["incorrect"])
	acc_dict["known_acc"] = acc_dict["known_correct"]/(acc_dict["known_correct"] + acc_dict["known_incorrect"])
	acc_dict["unknown_acc"] = acc_dict["unknown_correct"]/(acc_dict["unknown_correct"] + acc_dict["unknown_incorrect"])

	return(acc_dict)

#check our total accuracy and the accuracy for known words and unknown words, respectively
tested_simple_4_cmplx = accuracy_cmplx_doc(test_data,test_simple_4_tagged,top_featured_freq["word"])
print(tested_simple_4_cmplx)
```

```
> {'correct': 364267, 'incorrect': 21447, 'known_correct': 357759, 'known_incorrect': 17406, 'unknown_correct': 6508, 'unknown_incorrect': 4041, 'total_acc': 0.9443966254789818, 'known_acc': 0.9536044140578146, 'unknown_acc': 0.6169305147407337}
```

As we can see, our model is doing reasonably well with known words (95.3% accuracy), but not nearly as well with unknown words (61.6% accuracy). Given that our training data set is relatively small (~670,000 words), it is likely that we will encounter a higher proportion of unknown words in real data than we do in our test set. This means that our accuracy is likely to be worse on "real" data, so we will want to improve our unknown word accuracy substantially!

## Even more precise accuracy figures: Precision and recall for each tag

In addition to knowing how well our model performs on known and unknown words, it is also helpful to know which part of speech tags we can trust, and which ones may need additional (manual) analysis. Below, we will write code that will check the accuracy for each tag using precision (how often we are correct when we assign a particular tag), recall (the degree to which we appropriately tag "all" instances of a particular tag), and F1 (the harmonic mean of precision and recall).

```python
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

def possible_tags(dataset): #create preliminary accuracy dictionary (used in tag_prec_rec_doc() below)
	acc_dict = {}
	for sent in dataset:
		for token in sent:
			if token["pos"] not in acc_dict:
				acc_dict[token["pos"]] = {"TP":0,"FP":0,"FN":0}
	return(acc_dict)

#code for compilation of true positives, false positives, and false negatives for each tag
def tag_prec_rec_sent(gold,tested,tag_d):
	for idx, item in enumerate(gold):
		if item["pos"] == tested[idx]["pos"]:
			tag_d[item["pos"]]["TP"] += 1
		else:
			tag_d[item["pos"]]["FN"] += 1
			tag_d[tested[idx]["pos"]]["FP"] += 1

#code to apply the above function to a full dataset
def tag_prec_rec_doc(gold,tested,feature_d,full_data):
	tag_d = possible_tags(full_data)

	for idx, item in enumerate(gold):
		tag_prec_rec_sent(item,tested[idx],tag_d)

	for x in tag_d:
		prec_rec(tag_d[x])

	return(tag_d)

#get detailed tag information for our model
tested_simple_4_pr = tag_prec_rec_doc(test_data,test_simple_4_tagged,top_featured_freq,full_data)

#sort the dictionary by tag frequency to make it easier to read:
from operator import *
simple_4_pr_sorted = sorted(tested_simple_4_pr.items(),key=lambda x:getitem(x[1],'TC'), reverse = True)

#Check out precision and recall for the five most frequent tags
for x in simple_4_pr_sorted[:5]:
	print(x)
```

```
> ('NN', {'TP': 49814, 'FP': 3782, 'FN': 3478, 'TC': 53292, 'recall': 0.9347369211138632, 'precision': 0.9294350324651094, 'f1': 0.9320784372427214})
('IN', {'TP': 44347, 'FP': 2365, 'FN': 803, 'TC': 45150, 'recall': 0.9822148394241418, 'precision': 0.9493706114060627, 'f1': 0.9655134876227386})
('DT', {'TP': 38124, 'FP': 589, 'FN': 569, 'TC': 38693, 'recall': 0.9852944977127646, 'precision': 0.9847854725802702, 'f1': 0.985039919386094})
('JJ', {'TP': 21700, 'FP': 2563, 'FN': 3216, 'TC': 24916, 'recall': 0.8709263124096965, 'precision': 0.8943659069364877, 'f1': 0.8824904939100022})
('NNP', {'TP': 18238, 'FP': 1657, 'FN': 2088, 'TC': 20326, 'recall': 0.8972744268424677, 'precision': 0.9167127418949484, 'f1': 0.9068894358668358})
```

As we can see, we get reasonable precision, recall, and F1 scores for the "NN" tag (f1 = .932), better scores for "IN" (f1 = .965), and particularly good scores for "DT" (f1 = .985). Our fourth most frequent tag, "JJ", however, gets lower scores (f1 = .882). It should be noted that these scores all for ALL words, though we could fairly easy distinguish between known and unknown words here as well.

## Next up: Applying machine learning algorithms
