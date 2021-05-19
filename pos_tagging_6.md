# Introduction to POS Tagging (Part 6  - Perceptron Tagger)
(Kristopher Kyle updated 2021-05-18)

In this tutorial, we will work with the Perceptron Tagger, which can approach state of the art accuracy, trains and tags quickly, and is reasonably simple. The original creator of this implementation (Mathew Honnibal, who created [Spacy](https://spacy.io/), wrote up a [nice explanation of how the Perceptron Tagger works](https://explosion.ai/blog/part-of-speech-pos-tagger-in-python) back in 2013.

The TLDR version is that a perceptron is a very simple neural net (see more in-depth explanations [here](https://towardsdatascience.com/perceptron-learning-algorithm-d5db0deab975) and [here](https://machinelearningmastery.com/perceptron-algorithm-for-classification-in-python/#:~:text=The%20Perceptron%20algorithm%20is%20a,and%20predicts%20a%20class%20label.)).

In this tutorial, we will:
- briefly discuss some features of the tagger
- format our data for use with the tagger
- test a version with a very simple feature set (the same one we used in [Tutorial 5](pos_tagging_5.md))
- test a version with a rich feature set
- add a sentence and word tokenizer to create a fully functional POS tagger

All data for this tutorial can be [downloaded here](https://github.com/kristopherkyle/Corpus-Linguistics-Working-Group/raw/main/data/Perceptron_Tagger_CWG.zip). In order to follow along with the tutorial, you will want to unzip the folder, create a new .py file in it (e.g., "part_6.py"), and set your working directory.

## Getting data ready

## Perceptron Tagger

### Features
Two versions of the Perceptron Tagger program are included in the downloaded folder. The versions are identical except for the feature set. The simple version, which includes all of the features from tutorial 5 is included below. As we can see from the lines that are commented out, we can add a lot more features. Note that this function is embedded inside the PerceptronTagger() class - it is included below for illustrative purposes.

```python
def _get_features(self, i, word, context, prev, prev2):
	'''Map tokens into a feature representation, implemented as a
	{hashable: int} dict. If the features change, a new model must be
	trained.
	'''
	def add(name, *args):
		features[' '.join((name,) + tuple(args))] += 1

	i += len(self.START)
	features = defaultdict(int)
	# It's useful to have a constant feature, which acts sort of like a prior
	#From Kris: also possible given this code are prev2 (e.g., DT if our target tag is NN: DT JJ NN)
	add('bias')
	add('i suffix', context[i][-3:]) #current word suffix
	add('i-1 tag', prev) #previous tag
	add('i tag+i-2 tag', prev, prev2)
	add('i word', context[i]) #current word
# 		add('i-1 tag+i word', prev, context[i]) #previous tag+word bigram
# 		add('i-1 word', context[i-1]) #previous word
# 		add('i+1 word', context[i+1]) #next word
# 		add('i pref1', word[0]) #first letter prefix
# 		add('i pref3', context[i][:3]) #first three letters prefix
# 		add('i-2 tag', prev2) # second previous tag
# 		add('i-1 suffix', context[i-1][-3:]) #previous suffix
# 		add('i-2 word', context[i-2]) #second previous word
# 		add('i+1 suffix', context[i+1][-3:]) #next word suffix
# 		add('i+2 word', context[i+2]) #second next word

	return features
```

### Other Variables We Can Adjust
The Perceptron Tagger takes advantage of an unambiguous word dictionary, which can be tuned based on the acceptable ambiguity levels (by default this is 97%) and the number of occurrences required before a word can possibly be considered "unambiguous" (by default this is set at 20). These parameters are set in the _make_tag_dict() function of the PerceptronTagger() class.

```python
def _make_tagdict(self, sentences):
	'''
	Make a tag dictionary for single-tag words.
	:param sentences: A list of list of (word, tag) tuples.
	'''
	counts = defaultdict(lambda: defaultdict(int))
	for sentence in sentences:
		self._sentences.append(sentence)
		for word, tag in sentence:
			counts[word][tag] += 1
			self.classes.add(tag)
	freq_thresh = 20 #frequency threshold
	ambiguity_thresh = 0.97 #ambiguity threshold
	for word, tag_freqs in counts.items():
		tag, mode = max(tag_freqs.items(), key=lambda item: item[1])
		n = sum(tag_freqs.values())
		# Don't add rare words to the tag dictionary
		# Only add quite unambiguous words
		if n >= freq_thresh and (mode / n) >= ambiguity_thresh:
			self.tagdict[word] = tag
```

## Formatting Data
To train the Perceptron Tagger, data should be formatted as a list of sentences of (word, tag) tuples. We will import the data we have used in previous tutorials, which is formatted as list of sentences of {"word": word, "pos": tag} dictionaries, and then convert it.

```python
import random
random.seed(10) #set seed so we get the same results each time

def tupler(lolod):
	outlist = [] #output
	for sent in lolod: #iterate through sentences
		outsent = []
		for token in sent: #iterate through tokens
			outsent.append((token["word"],token["pos"])) #create tuples
		outlist.append(outsent)
	return(outlist) #return list of lists of tuples

### load data: ###
full_data = tupler(pickle.load(open("brown_sents_list.pickle","rb")))

### create training and test sets ###
train_data = random.sample(full_data,34912) #create training set with 67% of sentences

test_data = [x for sent in full_data if sent not in train_data]
```

## Training and Testing the Perceptron Tagger

### Training the Perceptron Tagger (SimpleTron)
```python
from simple_perceptron import PerceptronTagger as SimpleTron #import PerceptronTagger from simple_perceptron.py as SimpleTron

tagger = SimpleTron(load=False) #define tagger

tagger.train(train_data,save_loc = "small_feature_Browntrain_perceptron.pickle") #train tagger on train_data, save the model as "small_feature_Browntrain_perceptron.pickle"

#load pretrained model (if needed)
tagger = PerceptronTagger(load = True, PICKLE = "small_feature_Browntrain_perceptron.pickle")
```

### Testing the Perceptron Tagger (SimpleTron)

```python

### strip tags if necessary, apply tagger
def test_tagger(test_sents,model,tag_strip = False, word_loc = 0):

	if tag_strip == True:
		sent_words = []
		for sent in test_sents:
			ws = []
			for token in sent:
				ws.append(token[word_loc])
			sent_words.append(ws)
	else:
		sent_words = test_sents

	tagged_sents = []

	for sent in sent_words:
		tagged_sents.append(model.tag(sent))

	return(tagged_sents)

#Check accuracy
def simple_accuracy_sent(gold,test): #this takes a hand-tagged list (gold), and a machine-tagged text (test) and calculates the simple accuracy
	correct = 0 #holder for correct count
	nwords = 0 #holder for total words count

	for sent_id, sents in enumerate(gold): #iterate through sentences. Note enumerate() adds the index. So here, we define the index as "sent_id", and the item as "sents"
		for word_id, (gold_word, gold_tag) in enumerate(sents): #now, we iterate through the words in each sentence using enumerate(). the format is now [index, [word, tag]]. We define the index as 'word_id', the word as 'word' and the tag as 'tag'
			nwords += 1
			if gold_tag == test[sent_id][word_id][1]: #if the tag is correct, add one to the correct score
				correct +=1

	return(correct/nwords)

tagged_test = test_tagger(test_data,tagger,tag_strip = True)

print(simple_accuracy_sent(tagged_test,test_data)) #0.9320666399778277
```

```python
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
def tag_prec_rec(tested,gold):
	tag_d = {}

	for sent_id, sent in enumerate(gold):
		for idx, (word, tag) in enumerate(sent):
			### update tag dictionary as needed ###
			tested_tag = tested[sent_id][idx][1]

			if tag not in tag_d:
				tag_d[tag] = {"TP":0,"FP":0,"FN":0}
			if tested_tag not in tag_d:
				tag_d[tested_tag] = {"TP":0,"FP":0,"FN":0}

			### tabulate accuracy ###
			if tag == tested_tag:
				tag_d[tag]["TP"] += 1
			else:
				tag_d[tag]["FN"] += 1
				tag_d[tested_tag]["FP"] += 1

	for x in tag_d:
		prec_rec(tag_d[x])

	return(tag_d)

cmplx_acc_simp = tag_prec_rec(tagged_test,test_data)

from operator import *

#output F1 score of top ten most frequent tags
for x in sorted(cmplx_acc_simp.items(),key=lambda x:getitem(x[1],'TC'), reverse = True)[:10]:
	print(x[0], x[1]["f1"])
```

```
NN 0.9260796954740957
IN 0.9674883035869
DT 0.9862300154718927
JJ 0.8778548600826253
NNP 0.8577973386868453
, 0.9999479220914488
NNS 0.9604092289657377
. 0.9999725101025374
RB 0.9083532219570406
PRP 0.9856693014587752
```

### Training the Testing the Perceptron Tagger (FullTron)

Now, we will use the "fully" featured version of the Perceptron Tagger to increase the accuracy of our system. To reach fully state of the art accuracy (97-98%), we would need to have more training data - but our model below comes very close (96.27% macro accuracy).

```python
from full_perceptron import PerceptronTagger as FullTron

tagger2 = FullTron(load=False)

tagger2.train(train_tup,save_loc = "full_feature_Browntrain_perceptron.pickle")

#load pretrained model (if needed)
tagger2 = FullTron(load = True, PICKLE = "full_feature_Browntrain_perceptron.pickle")

tagged_test2 = test_tagger(test_data,tagger2,tag_strip = True)

simple_accuracy_sent(test_data,tagged_test2) #test 1 (small set): 0.9627081205245337

cmplx_acc = tag_prec_rec(tagged_test2,test_data)

for x in sorted(cmplx_acc.items(),key=lambda x:getitem(x[1],'TC'), reverse = True)[:10]:
	print(x[0], x[1]["f1"])
```

```
NN 0.9523881229118629
IN 0.9777645882107253
DT 0.9907638251973182
JJ 0.9085173501577287
NNP 0.9416097300964286
, 0.9999479220914488
NNS 0.9753486863444698
. 0.9999725101025374
RB 0.929699795977849
PRP 0.9927594529364441
```

## Implementing a Fully Functional Tagger

We can easily implement a full tagging pipeline by borrowing a sentence tokenizer and word tokenizer from the Natural Language Toolkit (NLTK). There are certainly more accurate tokenizers out there, but the default implementations in NLTK are both quick and reasonably accurate.

```python
from nltk.tokenize import sent_tokenize, word_tokenize

def tag_strings(input_string,trained_tagger):
	tagged_sents = []

	sents = sent_tokenize(input_string) #use nltk sent tokenize to separate strings into sentences
	for sent in sents:
		tagged_sents.append(trained_tagger.tag(word_tokenize(sent)))

	return(tagged_sents)

for sents in tag_strings("I really love pizza. Do you love pizza?", tagger2):
	print(sents)
```
