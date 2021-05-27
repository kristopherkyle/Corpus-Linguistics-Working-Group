# Introduction to POS Tagging (Part 7  - Beyond English)
(Kristopher Kyle - Updated 2021-05-26)

In this tutorial we apply the Perceptron Tagger to new datasets and languages. We will use data from [https://universaldependencies.org/](https://universaldependencies.org/), which includes a large number of datasets in .conllu format across a wide range of languages.

It should be noted that the universal dependency datasets are tagged with ["universal" part of speech tags](https://universaldependencies.org/u/pos/index.html), which tend to be less specific than language specific tags. However, in many cases often include language-specific tags as well (such as Penn tags for English).

## Dealing with .conllu files
In order to use these datasets, we will need to extract pertinent information from .conllu files, which are formatted as follows:

```
# sent_id = weblog-juancole.com_juancole_20051126063000_ENG_20051126_063000-0006
# text = The third was being run by the head of an investment firm.
1	The	the	DET	DT	Definite=Def|PronType=Art	2	det	2:det	_
2	third	third	ADJ	JJ	Degree=Pos|NumType=Ord	5	nsubj:pass	5:nsubj:pass	_
3	was	be	AUX	VBD	Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin	5	aux	5:aux	_
4	being	be	AUX	VBG	VerbForm=Ger	5	aux:pass	5:aux:pass	_
5	run	run	VERB	VBN	Tense=Past|VerbForm=Part|Voice=Pass	0	root	0:root	_
6	by	by	ADP	IN	_	8	case	8:case	_
7	the	the	DET	DT	Definite=Def|PronType=Art	8	det	8:det	_
8	head	head	NOUN	NN	Number=Sing	5	obl	5:obl:by	_
9	of	of	ADP	IN	_	12	case	12:case	_
10	an	a	DET	DT	Definite=Ind|PronType=Art	12	det	12:det	_
11	investment	investment	NOUN	NN	Number=Sing	12	compound	12:compound	_
12	firm	firm	NOUN	NN	Number=Sing	8	nmod	8:nmod:of	SpaceAfter=No
13	.	.	PUNCT	.	_	5	punct	5:punct	_

# sent_id = weblog-juancole.com_juancole_20051126063000_ENG_20051126_063000-0007
# text = You wonder if he was manipulating the market with his bombing targets.
1	You	you	PRON	PRP	Case=Nom|Person=2|PronType=Prs	2	nsubj	2:nsubj	_
2	wonder	wonder	VERB	VBP	Mood=Ind|Tense=Pres|VerbForm=Fin	0	root	0:root	_
3	if	if	SCONJ	IN	_	6	mark	6:mark	_
4	he	he	PRON	PRP	Case=Nom|Gender=Masc|Number=Sing|Person=3|PronType=Prs	6	nsubj	6:nsubj	_
5	was	be	AUX	VBD	Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin	6	aux	6:aux	_
6	manipulating	manipulate	VERB	VBG	Tense=Pres|VerbForm=Part	2	ccomp	2:ccomp	_
7	the	the	DET	DT	Definite=Def|PronType=Art	8	det	8:det	_
8	market	market	NOUN	NN	Number=Sing	6	obj	6:obj	_
9	with	with	ADP	IN	_	12	case	12:case	_
10	his	he	PRON	PRP$	Gender=Masc|Number=Sing|Person=3|Poss=Yes|PronType=Prs	12	nmod:poss	12:nmod:poss	_
11	bombing	bombing	NOUN	NN	Number=Sing	12	compound	12:compound	_
12	targets	target	NOUN	NNS	Number=Plur	6	obl	6:obl:with	SpaceAfter=No
13	.	.	PUNCT	.	_	2	punct	2:punct	_

```

With the exception of lines that start with "#", each item in a line is separated by `\t`, which makes it easy to extract the information we need. The function below extracts the word and universal pos (upos) from each line and export a list of sentences of token dictionaries:

```python
def conllu_dicter(text,splitter="\t"):
	output_list = [] #list for each sentence and token

	sents = text.split("\n\n") #split text into sentences

	for sent in sents:
		if sent == "":
			continue
		sent_anno = [] #sentence list for tokens
		lines = sent.split("\n")
		for line in lines:
			if len(line) == 0:
				continue
			token = {}
			if line[0] == "#": #skip lines without target annotation
				continue

			anno = line.split(splitter) #split by splitting character (in the case of our example, this will be a tab character)

			#now, we will grab all relevant information and add it to our token dictionary:
			#token["idx"] = anno[0]
			token["word"] = anno[1]#get word
			token["upos"] = anno[3] #get the universal pos tag
			#token["xpos"] = anno[4] #get the xpos tag(s) - in English this is usally Penn tags
			#token["dep"] = anno[7] #dependency relationship
			#token["head_idx"] = anno[6] #id of dependency head
			sent_anno.append(token) #append token dictionary to sentence level list
		output_list.append(sent_anno) #append sentence level list to
	return(output_list)
```

Because we are going to use the Perceptron Tagger, we will need to further convert the data in to a list of sentences of (word,upos) tuples. To do so, we will use a slightly modified version of the `tupler()` function from [Tutorial 6](pos_tagging_6.md)

```python
#We will use the simple perceptron
def tupler(lolod,posname = "pos"):
	outlist = [] #output
	for sent in lolod: #iterate through sentences
		outsent = []
		for token in sent: #iterate through tokens
			outsent.append((token["word"],token[posname])) #create tuples
		outlist.append(outsent)
	return(outlist) #return list of lists of tuples
```

## Loading the Perceptron Tagger and Related Functions
Below, we load the Perceptron Tagger and some related functions from [Tutorial 6](pos_tagging_6.md). Don't forget to put simple_perceptron.py and full_perceptron.py in your working directory!

```python
from simple_perceptron import PerceptronTagger as SimpleTron #import PerceptronTagger from simple_perceptron.py as SimpleTron
from full_perceptron import PerceptronTagger as FullTron

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

def simple_accuracy_sent(gold,test): #this takes a hand-tagged list (gold), and a machine-tagged text (test) and calculates the simple accuracy
	correct = 0 #holder for correct count
	nwords = 0 #holder for total words count

	for sent_id, sents in enumerate(gold): #iterate through sentences. Note enumerate() adds the index. So here, we define the index as "sent_id", and the item as "sents"
		for word_id, (gold_word, gold_tag) in enumerate(sents): #now, we iterate through the words in each sentence using enumerate(). the format is now [index, [word, tag]]. We define the index as 'word_id', the word as 'word' and the tag as 'tag'
			nwords += 1
			if gold_tag == test[sent_id][word_id][1]: #if the tag is correct, add one to the correct score
				correct +=1

	return(correct/nwords)
```

## Testing Perceptron tagger on new datasets

We will test the "simple" and the "full" versions of the Perceptron tagger on three languages:
- English (using the 254k-word English Web Treebank)
- Korean (using the 350k-word KAIST Treebank)
- Thai (using the 22k-word Prague Universal Dependencies Treekbank)

### English

First, we will train a model based on the simple feature set. This model achieves 90% accuracy for universal pos tags on the test set.

```python
#load English data
english_train = conllu_dicter(open("UD_English-EWT/en_ewt-ud-train.conllu").read())
english_test = conllu_dicter(open("UD_English-EWT/en_ewt-ud-test.conllu").read())

#convert from token dictionaries to token tuples:
english_train = tupler(english_train,posname = "upos")
english_test = tupler(english_test,posname = "upos")


#initialize and train tagger:
tagger_en = SimpleTron(load=False) #define tagger
tagger_en.train(english_train,save_loc = "small_feature_EWT_train_perceptron.pickle") #train tagger on train_data, save the model as "small_feature_Browntrain_perceptron.pickle"

#load pretrained model (if needed)
#tagger_en = SimpleTron(load = True, PICKLE = "small_feature_EWT_train_perceptron.pickle")

#check macro accuracy
tagged_en_test = test_tagger(english_test,tagger_en,tag_strip = True)
print(simple_accuracy_sent(tagged_en_test,english_test)) #0.9000589275191514
```

Next, we will test the "full" version of the tagger. This model achieves 93.6% accuracy on the test set data. While this is lower than with the Brown corpus, we also are using a substantially smaller set of training data.

```python
tagger2_en = FullTron(load=False)

tagger2_en.train(english_train,save_loc = "full_feature_EWT_train_perceptron.pickle")

#load pretrained model (if needed)
#tagger2_en = FullTron(load = True, PICKLE = "full_feature_EWT_train_perceptron.pickle")

tagged_en_test2 = test_tagger(english_test,tagger2_en,tag_strip = True)

simple_accuracy_sent(english_test,tagged_en_test2) #0.936436849341976
```

### Korean

First, we will train a model based on the simple feature set. This model achieves 84.7% accuracy for universal pos tags on the test set. Despite having more training data, the model is less accurate on the Korean data than the English data. This may mean that the feature set (or prediction algorithm) needs to be tuned for Korean.

```python
#load Korean data and convert to token tuples
korean_train = tupler(conllu_dicter(open("UD_Korean-Kaist/ko_kaist-ud-train.conllu").read()),posname = "upos")
korean_test = tupler(conllu_dicter(open("UD_Korean-Kaist/ko_kaist-ud-test.conllu").read()),posname = "upos")

tagger_ko = SimpleTron(load=False) #define tagger

tagger_ko.train(korean_train,save_loc = "small_feature_Kaist_train_perceptron.pickle") #train tagger on train_data, save the model as "small_feature_Browntrain_perceptron.pickle"

#load pretrained model (if needed)
tagger_ko = SimpleTron(load = True, PICKLE = "small_feature_Kaist_train_perceptron.pickle")


tagged_ko_test = test_tagger(korean_test,tagger_ko,tag_strip = True)
print(simple_accuracy_sent(tagged_ko_test,korean_test)) #0.847846012832264

```

Next, we will test the "full" version of the tagger. This model achieves 85.9% accuracy on the test set data.

```python
tagger2_ko = FullTron(load=False)

tagger2_ko.train(korean_train,save_loc = "full_feature_Kaist_train_perceptron.pickle")

#load pretrained model (if needed)
#tagger2_ko = FullTron(load = True, PICKLE = "full_feature_Kaist_train_perceptron.pickle")

tagged_ko_test2 = test_tagger(korean_test,tagger2_ko,tag_strip = True)

simple_accuracy_sent(korean_test,tagged_ko_test2) #test 1 (small set): 0.8593386448565183
```

### Thai
First, we will train a model based on the simple feature set. This model achieves 85.8% accuracy for universal pos tags on the test set, which is rather impressive given the (relatively) tiny training set.

```python
#Thai only comes with a single dataset, so we will have to divide it ourselves
thai_full = tupler(conllu_dicter(open("UD_Thai-PUD/th_pud-ud-test.conllu").read()),posname = "upos")

thai_train = thai_full[:667]
thai_test = thai_full[667:]

tagger_th = SimpleTron(load=False) #define tagger

tagger_th.train(thai_train,save_loc = "small_feature_ThaiPUD_train_perceptron.pickle") #train tagger on train_data, save the model as "small_feature_Browntrain_perceptron.pickle"

#load pretrained model (if needed)
#tagger_th = SimpleTron(load = True, PICKLE = "small_feature_ThaiPUD_train_perceptron.pickle")


tagged_th_test = test_tagger(thai_test,tagger_th,tag_strip = True)
print(simple_accuracy_sent(tagged_th_test,thai_test)) #0.8582687683676196
```

Next, we will test the "full" version of the tagger. This model achieves 88.1% accuracy on the test set data, which again is impressive given the small size of the training data.

```python
tagger2_th = FullTron(load=False)

tagger2_th.train(thai_train,save_loc = "full_feature_ThaiPUD_train_perceptron.pickle")

#load pretrained model (if needed)
#tagger2_th = FullTron(load = True, PICKLE = "full_feature_ThaiPUD_train_perceptron.pickle")

tagged_th_test2 = test_tagger(thai_test,tagger2_th,tag_strip = True)

print(simple_accuracy_sent(thai_test,tagged_th_test2)) #test 1 (small set): 0.8815121560245792

```
