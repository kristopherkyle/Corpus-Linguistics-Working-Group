# Introduction to POS Tagging (Part 3)
(Kristopher Kyle - Updated 2021-04-07)

## Getting more precise accuracy figures
As we work towards achieving state of the art overall accuracy, it is helpful to know the strengths and weaknesses of our model(s). First, we will write a slightly more complex version of our accuracy analysis code to differentiate between accuracy for known words and unknown words.

Note: This code presumes that you have run the code from [Tutorial 2](pos_tagging_2.md)

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
