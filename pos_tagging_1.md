# Introduction to POS Tagging

In this tutorial we will get starting with part of speech (POS) tagging. We will first approach the issue conceptually, and then start working on implementing our tagger.

We will be using English data, but the techniques we will be using can be applied to any language (including less-documented ones! really!).

## How a part of speech (POS) tagging works.

Although there are a few different historic approaches to POS tagging, current POS tagging involves the following steps:

- Tag all word forms with unambiguous tags (i.e., word forms that will ALWAYS get a particular tag)
- Extract features (such as word endings, the tags of previous words, etc) that can be use to disambiguate tags
- Apply an algorithm to predict the tag based on available relevant features (specific algorithms vary) for known words
- Apply an algorithm to predict the tag based on available relevant features (specific algorithms vary) for unknown words

## What we need to train a POS tagging model

In order to create a POS tagging model, we first need some pre-tagged data. In this tutorial series, we will be using a tagged version of the Brown corpus (1-million words) with [Penn POS Tags](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html) (which is a very common tagset for English). You can [download this dataset here](https://github.com/kristopherkyle/Corpus-Linguistics-Working-Group/raw/main/docs/_extracted_brown.zip). Note that the format of the tagged files is as follows: Each sentence is separated by two newline characters, each word-tag pair is separated by one newline character, and the word and tag are separated by a space.

Below, we will load this data and format it. For ease of use (and for future application), we will represent the dataset as a list of lists, where each sentence is its own list, and words are represented as dictionaries with multiple features (e.g., "word", "pos", etc.).

```python
import glob

tagged_files = glob.glob("_extracted_brown/*.txt") #get list of files
len(tagged_files) #500 files

#divide into sentences
full_data = []
for x in tagged_files:
	text = open(x).read().split("\n\n") #split into sentences
	for sent in text: #iterate through sentences
		items = [] #start sentence-level text
		for word in sent.split("\n"): #iterate through words
			if " " not in word:
				continue
			if word == "":
				continue
			if word == '': #skip extra spaces
				continue
			else:
				items.append({"word" : word.split(" ")[0], "pos" : word.split(" ")[1]}) #add dictionary representation of word and tag to list
		if len(items) == 0:
			continue
		full_data.append(items)
```

Now, we will check to make sure that our data looks as we expect it to.

```python
print(full_data[1][:10])
```
```
> [{'word': 'The', 'pos': 'DT'}, {'word': 'partners', 'pos': 'NNS'}, {'word': 'each', 'pos': 'DT'}, {'word': 'bring', 'pos': 'VBP'}, {'word': 'to', 'pos': 'TO'}, {'word': 'it', 'pos': 'PRP'}, {'word': 'unselfish', 'pos': 'JJ'}, {'word': 'love', 'pos': 'NN'}, {'word': ',', 'pos': ','}, {'word': 'and', 'pos': 'CC'}]
```

## How we build a POS model

Now that we have some tagged data loaded, lets think about how we will build our model. We won't get through all of the steps in this tutorial, but as we will see, it really isn't all that complicated (and, we can get over 90% tagging accuracy without a prediction algorithm). In future tutorials, we will work on getting state of the art accuracy from our tagging model (around 98%!)

To build a POS tagging model, we will follow these steps:

- First, we will split our tagged corpus into a training set (roughly 67% of our sentences) and a test set (33% of our sentences). This will allow us to train our models and then test them on new data (which will provide more generalizable accuracy figures).
- Second, we will make a dictionary that consists of all unambiguous word:tag pairs in our training corpus. Since state of the art taggers reach around 98% accuracy, we will treat any word:tag pair that accounts for 98% tag instances for a particular word as "unambiguous".
- Third, we will make a dictionary that consists of all words and the frequency particular tags occur with each word (this will help us predict the tag for words with tag ambiguity). At first, we will just assign the most frequent tag to each word, but later we will use a more sophisticated predictor model.
- Finally, we will determine which tag is most frequent in our corpus - for now we will assign that tag to any unknown words our model encounters. Later we will use a more sophisticated predictor model.

## Building our model

### Create training and test sets
First, we will split our data into training and test tests.

```python
import random
random.seed(10) #set seed so we get the same results each time

#number of sentences for training set:
print(len(full_data)*.67) #34912.36
train_data = random.sample(full_data,34912) #create training set with 67% of sentences

#then, we will put any sentences that are NOT in train_data in test_data (this loop will take a little while)
test_data = []
for x in full_data:
	if x not in train_data:
		test_data.append(x)

print(len(train_data)) #34912
print(len(test_data)) #17029
```

### Unambiguous word dictionary

To create our unambiguous word dictionary, we will first get the frequency of each tag for each word and store this information in a dictionary:

```python
def freq_add(item,d):
	if item not in d:
		d[item] = 1
	else:
		d[item]+= 1

#iterate through sentences, get tabulate tags for each word
def tag_freq(data_set):
	freq = {}
	for sent in data_set:
		for item in sent:
			if item["word"] not in freq:
				freq[item["word"]] = {}
			freq_add(item["pos"],freq[item["word"]])

	return(freq)

#create frequency dictionary
word_tags = tag_freq(train_data)

#check
print(word_tags["the"])
print(word_tags["The"])
print(word_tags["run"])
```
```
> {'DT': 41794, 'NNP': 4} #the
> {'DT': 4682, 'NNP': 113, 'JJ': 1} #The
> {'VBP': 18, 'VB': 65, 'NN': 34, 'VBN': 21} #run
```

As we can see, words like "the" are tagged reasonably unambiguously, while words like "run" could be assigned a variety of tags.

Now, we will iterate through each word and determine whether it is tagged in a reasonably unambiguous way. In the function below, we will allow for different operationalizations of "unambiguous" (our default with be .98) and will also control for the possibility that words that are unambiguous but infrequent in our corpus may be actually be ambiguous (we will set this default value at 5).

```python
def unambiguous_tags(freq_dict,prob_thresh = .98,occur_thresh = 5):
	unam = {} #for unambiguous word:tag pairs

	for x in freq_dict: #iterate through words in dataset
		total  = sum(freq_dict[x].values()) #get total word frequency (sum of all tag frequencies)
		if total < occur_thresh:
			continue
		for y in freq_dict[x]:
			if freq_dict[x][y]/total >= prob_thresh:
				unam[x] = y
	return(unam)

unambiguous = unambiguous_tags(word_tags)

print(unambiguous["the"]) #DT
print(unambiguous["The"]) #key error! not unambiguous (also tagged as NNP!)
print(unambiguous["run"]) #key error! not unambiguous
```

Lets check to see how much coverage we get with our unambiguous dictionary. To do so, we will create the simplest of POS tagging models. We will ONLY tag words that were unambiguous in our training data, and count all other words as "errors" (we won't assign them a tag).

```python
def simple_model_1(sent_to_tag,unam_d):
	tagged = []
	for x in sent_to_tag:
		word = x["word"]
		if word in unam_d: #if the word is unambiguous, assign the tag
			tagged.append({"word": word, "pos": unam_d[word]})
		else: #else, assign tag as "none"
			tagged.append({"word": word, "pos": "none"})

	return(tagged)

def simple_model_1_doc(doc_to_tag,unam_d):
	tagged = []
	for sent in doc_to_tag:
		tagged.append(simple_model_1(sent,unam_d))

	return(tagged)

test_simple_1_tagged = simple_model_1_doc(test_data,unambiguous) #tag test data with simple model
print(test_simple_1_tagged[0]) #check results

```

```
> [{'word': 'The', 'pos': 'none'}, {'word': 'partners', 'pos': 'NNS'}, {'word': 'each', 'pos': 'DT'}, {'word': 'bring', 'pos': 'none'}, {'word': 'to', 'pos': 'TO'}, {'word': 'it', 'pos': 'PRP'}, {'word': 'unselfish', 'pos': 'none'}, {'word': 'love', 'pos': 'none'}, {'word': ',', 'pos': ','}, {'word': 'and', 'pos': 'CC'}, {'word': 'each', 'pos': 'DT'}, {'word': 'takes', 'pos': 'VBZ'}, {'word': 'away', 'pos': 'RB'}, {'word': 'an', 'pos': 'DT'}, {'word': 'equal', 'pos': 'none'}, {'word': 'share', 'pos': 'none'}, {'word': 'of', 'pos': 'IN'}, {'word': 'pleasure', 'pos': 'none'}, {'word': 'and', 'pos': 'CC'}, {'word': 'joy', 'pos': 'NN'}, {'word': '.', 'pos': '.'}]
```

Now, we will check the accuracy of our model of course, the accuracy will not be high, but as we add the other features, the accuracy will increase substantially

```python
def accuracy_sent(gold,tested,acc_dict):
	for idx, item in enumerate(gold):
		if item["pos"] == tested[idx]["pos"]:
			acc_dict["correct"] += 1
		else:
			acc_dict["false"] += 1

def accuracy_doc(gold,tested):
	acc_dict = {"correct" : 0, "false" : 0}

	for idx, item in enumerate(gold):
		accuracy_sent(item,tested[idx],acc_dict)

	accuracy = acc_dict["correct"]/(acc_dict["correct"] + acc_dict["false"])
	acc_dict["acc"] = accuracy

	return(acc_dict)

tested_simple_1 = accuracy_doc(test_data,test_simple_1_tagged)
print(tested_simple_1)
```

```
> {'correct': 261206, 'false': 124508, 'acc': 0.6772012423712906}
```

As we can see, our extremely simple model achieved an overall tagging accuracy of 67.77% on new data. Not too shabby! Next, we will increase the accuracy by using the most probably tags for known words, and the most common tag in the corpus for unknown words.

### Adding features: Most likely tag for ambiguous known words

Now, we will create a dictionary that includes word:tag pairs for known words and their most frequently occurring tags. Then we will see how much it improved our model.

```python
import operator
def sort_tags(freq, only_top = True):
	sort_freq = {}
	for x in freq: #iterate through dictionary
		if only_top == True:
			sort_freq[x] = sorted(freq[x].items(),key=operator.itemgetter(1), reverse = True)[0][0] #get most frequent tag
		else:
			sort_freq[x] = sorted(freq[x].items(),key=operator.itemgetter(1), reverse = True)#so we can see all tags if we want

	return(sort_freq)

top_hits = sort_tags(word_tags) #get dictionary of word:top_tag pairs

print(word_tags["run"]) #all hits
print(top_hits["run"]) #top hit

```

```
> {'VBP': 18, 'VB': 65, 'NN': 34, 'VBN': 21}
> VB #run
```

Now, we can add a line to our tagger and then check the accuracy. We will still choose NOT to tag unknown words (they will get "none").

```python
def simple_model_2(sent_to_tag,unam_d,known_d):
	tagged = []
	for x in sent_to_tag:
		word = x["word"]
		if word in unam_d: #if the word is unambiguous, assign the tag
			tagged.append({"word": word, "pos": unam_d[word]})
		#this is new in model 2:
		elif word in known_d:
			tagged.append({"word": word, "pos": known_d[word]})
		else: #else, assign tag as "none"
			tagged.append({"word": word, "pos": "none"})

	return(tagged)

def simple_model_2_doc(doc_to_tag,unam_d,known_d):
	tagged = []
	for sent in doc_to_tag:
		tagged.append(simple_model_2(sent,unam_d,known_d))

	return(tagged)

test_simple_2_tagged = simple_model_2_doc(test_data,unambiguous,top_hits)
print(test_simple_2_tagged[0])
```

```
> [{'word': 'The', 'pos': 'DT'}, {'word': 'partners', 'pos': 'NNS'}, {'word': 'each', 'pos': 'DT'}, {'word': 'bring', 'pos': 'VB'}, {'word': 'to', 'pos': 'TO'}, {'word': 'it', 'pos': 'PRP'}, {'word': 'unselfish', 'pos': 'none'}, {'word': 'love', 'pos': 'NN'}, {'word': ',', 'pos': ','}, {'word': 'and', 'pos': 'CC'}, {'word': 'each', 'pos': 'DT'}, {'word': 'takes', 'pos': 'VBZ'}, {'word': 'away', 'pos': 'RB'}, {'word': 'an', 'pos': 'DT'}, {'word': 'equal', 'pos': 'JJ'}, {'word': 'share', 'pos': 'NN'}, {'word': 'of', 'pos': 'IN'}, {'word': 'pleasure', 'pos': 'NN'}, {'word': 'and', 'pos': 'CC'}, {'word': 'joy', 'pos': 'NN'}, {'word': '.', 'pos': '.'}]
```

Now, we can check the accuracy using our previous accuracy check code. As we see below, we are now getting over 91% accuracy with our simple tagger.

```python
tested_simple_2 = accuracy_doc(test_data,test_simple_2_tagged)
print(tested_simple_2)
```

```
> {'correct': 351315, 'false': 34399, 'acc': 0.9108173413461788}
```

### Adding features: Most likely tag for unknown words

Now, we will check to see what the most frequent tag in the training set is. For unknown words, we will use the most probably tag in the corpus.

```python
def item_freq(data_set,item_name):
	freq = {}
	for sent in data_set:
		for item in sent:
			freq_add(item[item_name],freq)
	return(freq)


pos_freq = item_freq(train_data,"pos") #get frequency of tags

pos_freq_sort = sorted(pos_freq.items(), key=operator.itemgetter(1), reverse = True) #sort tags

print(pos_freq_sort[:10]) #most frequent is NN
```

```
[('NN', 108064), ('IN', 91547), ('DT', 77742), ('JJ', 51659), ('NNP', 41681), (',', 39056), ('NNS', 37448), ('.', 37263), ('RB', 34656), ('PRP', 31712)]
```

Now that we know that the most frequent tag is "NN", we will update our tagger to deal with unknown words (if rather poorly).

```python
def simple_model_3(sent_to_tag,unam_d,known_d,unknown_tag):
	tagged = []
	for x in sent_to_tag:
		word = x["word"]
		if word in unam_d: #if the word is unambiguous, assign the tag
			tagged.append({"word": word, "pos": unam_d[word]})
		#this is new in model 2:
		elif word in known_d:
			tagged.append({"word": word, "pos": known_d[word]})
		else: #else, assign tag as "none"
			tagged.append({"word": word, "pos": unknown_tag})

	return(tagged)

def simple_model_3_doc(doc_to_tag,unam_d,known_d,unknown_tag):
	tagged = []
	for sent in doc_to_tag:
		tagged.append(simple_model_3(sent,unam_d,known_d,unknown_tag))

	return(tagged)
```

```python
test_simple_3_tagged = simple_model_3_doc(test_data,unambiguous,top_hits,"NN")
print(test_simple_3_tagged[0])
```

```
> [{'word': 'The', 'pos': 'DT'}, {'word': 'partners', 'pos': 'NNS'}, {'word': 'each', 'pos': 'DT'}, {'word': 'bring', 'pos': 'VB'}, {'word': 'to', 'pos': 'TO'}, {'word': 'it', 'pos': 'PRP'}, {'word': 'unselfish', 'pos': 'NN'}, {'word': 'love', 'pos': 'NN'}, {'word': ',', 'pos': ','}, {'word': 'and', 'pos': 'CC'}, {'word': 'each', 'pos': 'DT'}, {'word': 'takes', 'pos': 'VBZ'}, {'word': 'away', 'pos': 'RB'}, {'word': 'an', 'pos': 'DT'}, {'word': 'equal', 'pos': 'JJ'}, {'word': 'share', 'pos': 'NN'}, {'word': 'of', 'pos': 'IN'}, {'word': 'pleasure', 'pos': 'NN'}, {'word': 'and', 'pos': 'CC'}, {'word': 'joy', 'pos': 'NN'}, {'word': '.', 'pos': '.'}]
```

Now, we will check how much our accuracy improved. As we will see, the model didn't improve very much (though we did get 2,000 more tags correct). The final verdict: Our very simple model achieves 91.69% overall accuracy.

```python
tested_simple_3 = accuracy_doc(test_data,test_simple_3_tagged)
print(tested_simple_3)
```

```
> {'correct': 353650, 'false': 32064, 'acc': 0.9168710495341108}
```

## Next steps on our way to state of the art accuracy...

Next, we will work on improving our identification of ambiguous known words and of unknown words.
