# Introduction to POS Tagging (Part 2)
(Kristopher Kyle - Updated 2021-04-07)

In the [first part of this tutorial](pos_tagging_1.md), we created a basic POS tagger that tagged all unambiguous words, assigned the most probable tag to each ambiguous known word, and then assigned the most frequent tag in the training data to unknown words. This very basic model achieved an overall accuracy of 91.69%. In this tutorial, we will improve our tagging model by adding predictor features. We will also look at accuracy in a much more fine-grained manner. In the next tutorial (tutorial 3), we will apply machine learning algorithms to increase the accuracy of our tagger (and thereby create a tagger with state of the art accuracy).

Note, this tutorial will presume that you have the data (and functions) loaded from the first part of the tutorial.

## Adding features

So far, we have only used the word itself as a predictor for tags. This works really well for unambiguous words, decently well (though not great) for known ambiguous words, and particularly poorly for unknown words. To increase our tagging accuracy, we will need to include more predictor features. Using our experience as linguistics, we can certainly add a wide array of features that should be predictive of a word's POS tag. We will add a few below, but these should most certainly not be considered comprehensive. We will use the following information:

- The current word (as we have used before)
- The tags of the two preceding words (more accurate but less coverage)
- The tag of the preceding word (less accurate, but more coverage)
- The last two letters of a word (as a proxy for a suffix)
- The last three letters of a word (as a proxy for a suffix)
- A combination of the above features

```python
def add_features(input_sent,idx,token): #takes a sentence as input (with word and tag specified), outputs a more feature-rich version
	features = {}
	features["word"] = token["word"]
	features["pos"] = token["pos"]
	if idx == 0:
		features["prev_pos"] = "<start>" #no previous pos
		features["prev_pos_bg"] = "<start>_<start>" #no  previous pos_bg
		features["prev_word"] = "<first_word>" #no previous word

	elif idx == 1:
		features["prev_pos"] = input_sent[idx-1]["pos"] #previos pos_tag
		features["prev_pos_bg"] = "<start>_" + input_sent[idx-1]["pos"]
		features["prev_word"] = input_sent[idx-1]["word"] #no previous word

	else:
		features["prev_pos"] = input_sent[idx-1]["pos"] #
		features["prev_pos_bg"] = input_sent[idx-2]["pos"] + "_"  + input_sent[idx-1]["pos"]
		features["prev_word"] = input_sent[idx-1]["word"] #no previous word

	features["suffix_bg"] = features["word"][-2:] #get last two characters
	features["suffix_tg"] = features["word"][-3:] #get last two characters
	features["mx_bigram"] = features["prev_pos"] + "_" + features["word"]
	features["mx_trigram"] = features["prev_pos_bg"] + "_" + features["word"] #make pos bigram + current suffix
	features["mx_suffix_bg_bigram"] = features["prev_pos"] + "_" + features["suffix_bg"]
	features["mx_suffix_bg_trigram"] = features["prev_pos_bg"] + "_" + features["suffix_bg"] #make pos bigram + current suffix
	features["mx_suffix_tg_bigram"] = features["prev_pos"] + "_" + features["suffix_tg"]
	features["mx_suffix_tg_trigram"] = features["prev_pos_bg"] + "_" + features["suffix_tg"] #make pos bigram + current suffix

	return(features)
```

Now, we can test the function:

```python
test = [{'word': 'The', 'pos': 'DT'}, {'word': 'partners', 'pos': 'NNS'}, {'word': 'each', 'pos': 'DT'}, {'word': 'bring', 'pos': 'VB'}, {'word': 'to', 'pos': 'TO'}, {'word': 'it', 'pos': 'PRP'}, {'word': 'unselfish', 'pos': 'NN'}, {'word': 'love', 'pos': 'NN'}, {'word': ',', 'pos': ','}, {'word': 'and', 'pos': 'CC'}, {'word': 'each', 'pos': 'DT'}, {'word': 'takes', 'pos': 'VBZ'}, {'word': 'away', 'pos': 'RB'}, {'word': 'an', 'pos': 'DT'}, {'word': 'equal', 'pos': 'JJ'}, {'word': 'share', 'pos': 'NN'}, {'word': 'of', 'pos': 'IN'}, {'word': 'pleasure', 'pos': 'NN'}, {'word': 'and', 'pos': 'CC'}, {'word': 'joy', 'pos': 'NN'}, {'word': '.', 'pos': '.'}]

for idx, x in enumerate(test[:3]): #add features to the first three items
	print(add_features(test,idx,x))
```
```
> {'word': 'The', 'pos': 'DT', 'prev_pos': '<start>', 'prev_pos_bg': '<start>_<start>', 'prev_word': '<first_word>', 'suffix_bg': 'he', 'suffix_tg': 'The', 'mx_bigram': '<start>_The', 'mx_trigram': '<start>_<start>_The', 'mx_suffix_bg_bigram': '<start>_he', 'mx_suffix_bg_trigram': '<start>_<start>_he', 'mx_suffix_tg_bigram': '<start>_The', 'mx_suffix_tg_trigram': '<start>_<start>_The'}
{'word': 'partners', 'pos': 'NNS', 'prev_pos': 'DT', 'prev_pos_bg': '<start>_DT', 'prev_word': 'The', 'suffix_bg': 'rs', 'suffix_tg': 'ers', 'mx_bigram': 'DT_partners', 'mx_trigram': '<start>_DT_partners', 'mx_suffix_bg_bigram': 'DT_rs', 'mx_suffix_bg_trigram': '<start>_DT_rs', 'mx_suffix_tg_bigram': 'DT_ers', 'mx_suffix_tg_trigram': '<start>_DT_ers'}
{'word': 'each', 'pos': 'DT', 'prev_pos': 'NNS', 'prev_pos_bg': 'DT_NNS', 'prev_word': 'partners', 'suffix_bg': 'ch', 'suffix_tg': 'ach', 'mx_bigram': 'NNS_each', 'mx_trigram': 'DT_NNS_each', 'mx_suffix_bg_bigram': 'NNS_ch', 'mx_suffix_bg_trigram': 'DT_NNS_ch', 'mx_suffix_tg_bigram': 'NNS_ach', 'mx_suffix_tg_trigram': 'DT_NNS_ach'}
```

Next, we will compile the most frequent tags for each predictor so that we can construct a more robust back-off POS tagger.

```python
def mult_freq(training_data): #compile frequency data for each predictor
	freq = {}#output dictionary
	for sent in training_data: #iterate through training data
		for idx, token in enumerate(sent): # add features
				tok_features = add_features(sent,idx,token)
				for feature in tok_features:
					if feature not in freq:
						freq[feature] = {} #add feature dictionary to freq dictionary (e.g., freq["prev_pos"] = {})
					if tok_features[feature] not in freq[feature]:
						freq[feature][tok_features[feature]] = {} #add pos dictionary (e.g., freq["prev_pos"]["DT"] = {})
					freq_add(tok_features["pos"],freq[feature][tok_features[feature]]) #add pos instance to that dictionary
	return(freq)

def sort_tags_multi(featured_freq):
	top_freq = {}
	for feature in featured_freq:
		top_freq[feature] = sort_tags(featured_freq[feature])

	return(top_freq)

#compile stats on training data for each predictor:
featured_freq = mult_freq(train_data)

#create dictionary with top tag for each predictor:
top_featured_freq = sort_tags_multi(featured_freq)
print(top_featured_freq["prev_pos_bg"]["DT_JJR"])
print(top_featured_freq["prev_pos"]["DT"])
```

```
> "NN"
> "NN"
```

## Using features in a tagger

We will now create a slightly more complex "back-off" tagger where we will include more predictor features. Because we can only use information one predictor at a time without a more advanced algorithm (next tutorial), we will try to use the predictor with the most precise information first (e.g., tag bigram + word), and then "back-off" if our test data isn't represented and use a predictor with less information (e.g., tag + word, tag + suffix, etc.).

```python
def simple_model_4(sent_to_tag,unam_d,feature_d,unknown_tag):
	tagged = []
	for idx, x in enumerate(sent_to_tag):
		word = x["word"]
		token_d = add_features(tagged,idx,x)

		if word in unam_d: #if the word is unambiguous, assign the tag
			tagged.append({"word": word, "pos": unam_d[word]})

		#rules for known words
		elif token_d["mx_trigram"] in feature_d["mx_trigram"]: #if the pos bigram + word in the dictionary:
			tagged.append({"word": word, "pos": feature_d["mx_trigram"][token_d["mx_trigram"]]})

		elif token_d["mx_bigram"] in feature_d["mx_bigram"]: #if the pos bigram + word in the dictionary:
			tagged.append({"word": word, "pos": feature_d["mx_bigram"][token_d["mx_bigram"]]})

		elif word in feature_d["word"]:
			tagged.append({"word": word, "pos": feature_d["word"][token_d["word"]]})

		#rules for unknown words

		elif token_d["mx_suffix_tg_trigram"] in feature_d["mx_suffix_tg_trigram"]: #if the the pos bigram + last three letters are in the dictionary:
			tagged.append({"word": word, "pos": feature_d["mx_suffix_tg_trigram"][token_d["mx_suffix_tg_trigram"]]})

		elif token_d["mx_suffix_tg_bigram"] in feature_d["mx_suffix_tg_bigram"]: ##if the previous POS tag + last three letters are in the dictionary
			tagged.append({"word": word, "pos": feature_d["mx_suffix_tg_bigram"][token_d["mx_suffix_tg_bigram"]]})

		elif token_d["suffix_tg"] in feature_d["suffix_tg"]: #if the last three letters are in the dictionary:
			tagged.append({"word": word, "pos": feature_d["suffix_tg"][token_d["suffix_tg"]]})

		elif token_d["suffix_bg"] in feature_d["suffix_bg"]: ##if the last two letters are in the dictionary:
			tagged.append({"word": word, "pos": feature_d["suffix_bg"][token_d["suffix_bg"]]})

		elif token_d["prev_pos_bg"] in feature_d["prev_pos_bg"]: #if the previous pos bigram in the dictionary:
			tagged.append({"word": word, "pos": feature_d["prev_pos_bg"][token_d["prev_pos_bg"]]})

		elif token_d["prev_pos"] in feature_d["prev_pos"]: #if the previous pos in the dictionary:
			tagged.append({"word": word, "pos": feature_d["prev_pos"][token_d["prev_pos"]]})

		else: #if somehow we have a completely new word... then tag it as a "NN" - but we actually won't ever get here (because of our POS rules).
			tagged.append({"word": word, "pos": unknown_tag})

	return(tagged)

def simple_model_4_doc(doc_to_tag,unam_d,feature_d,unknown_tag):
	tagged = []
	for sent in doc_to_tag:
		tagged.append(simple_model_4(sent,unam_d,feature_d,unknown_tag))

	return(tagged)

test_simple_4_tagged = simple_model_4_doc(test_data,unambiguous,top_featured_freq,"NN")
tested_simple_4 = accuracy_doc(test_data,test_simple_4_tagged)
print(tested_simple_4) #94.4%
```

```
> {'correct': 364267, 'false': 21447, 'acc': 0.9443966254789818}
```

Our more complex model gained us a bit more accuracy! We now get approximately 94% of our tags correct.

## Next up: Getting more precise diagnostic (accuracy) information
