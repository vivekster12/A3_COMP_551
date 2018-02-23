# random classifier
import random
import pandas as pd
import numpy as np
import string
import collections
import time
import sklearn.metrics as sk
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import BernoulliNB

def random_classifier(corpus):
	true_rating = []
	pred_rating = []
	for i in range(len(corpus)):
		true_rating.append(corpus.iloc[i,1])
		pred_rating.append(random.randrange(1,5,1))

	f1_score = sk.f1_score(true_rating,pred_rating, average = 'micro')
	return (f1_score)

def majority_classifier(corpus):
	counter = collections.Counter(corpus[1])
	true_rating = []
	for i in range(len(corpus)):
		true_rating.append(corpus.iloc[i,1])

	pred_rating = np.linspace(counter.most_common(1)[0][0],counter.most_common(1)[0][0], len(true_rating))
	f1_score = sk.f1_score(true_rating,pred_rating, average = 'micro')
	return (f1_score)

# ***** NEEDED FOR BAYES ****** #
def create_bin_bag_of_words(corpus,dictionary):
	Bin_BoW = []
	for i in range(len(corpus)):
		words_in_sample = (corpus.iloc[i,0].split(" "))
		vector = np.zeros(len(dictionary))
		for word in words_in_sample:
			if (word in dictionary):
				np.put(vector,dictionary[word],1)	
		Bin_BoW.append(vector)

	return (np.array(Bin_BoW))

def create_dictionary(corpus):
	word_list = []
	translator = str.maketrans(string.punctuation.replace('\'',''),31*' ', '\'')

	for i in range(len(corpus)):
		word_list.extend(corpus.iloc[i,0].translate(translator).lower().split(" "))

	counter = collections.Counter(word_list)
	dict_size = 10001
	# stored in list then converted to dict
	dictionary= list(zip(*counter.most_common(dict_size)[1:]))[0]
	dict = {}
	for index,word in enumerate(dictionary):
		dict[word] = index

	return (dict)

def get_true_rating(corpus):
	true_rating = []
	for i in range(len(corpus)):
		true_rating.append(corpus.iloc[i,1])

	return (true_rating)

# ***** BAYES ****** #
def naive_bayes_classifier(training_Bin_BoW, dictionary,train_true_rating,testing_Bin_BoW, testing_true_rating):
	# class conditional probabilities (prob of word given class)
	U_1 = []
	U_2 = []
	# get total number of instances when rating is 1 and when rating is 0
	C1_total = np.sum(train_true_rating)
	C2_total = len(train_true_rating) - C1_total
	
	print ("calculating class conditional probs")
	for word in training_Bin_BoW.T:
		U_1_word = 0
		U_2_word = 0
		for index,review in enumerate(word):
			if (review == 1 and train_true_rating[index] == 1):
				U_1_word +=1
			if (review == 1 and train_true_rating[index] == 0):
				U_2_word +=1

		U_1.append((U_1_word + 1)/float(C1_total + 2))
		U_2.append((U_2_word + 1)/float(C2_total + 2))
	
	print ("finished calculating class conditional probs")
	# each array stores the probability that a review is class 1 or class 0
	print ("making predictions")
	P_1 = []
	P_0 = []
	pred_arr = []
	for review in testing_Bin_BoW:
		prob_c1 = 1
		prob_c0 = 1
		for index,word in enumerate(review):
			if (word == 1):
				prob_c1 *= U_1[index]
				prob_c0 *= U_2[index]
		if(prob_c1 >= prob_c0):
			pred_arr.append(1)
		else:
			pred_arr.append(0)

	print ("finished making predictions, calc f_score")
	f1_score = sk.f1_score(testing_true_rating,pred_arr, average = 'micro')
	return (f1_score)

def test_scores(predicted, actual):
    print('F1 Score:', sk.f1_score(actual, predicted, average='micro'))
    print('Confusion Matrix:\n', confusion_matrix(actual, predicted))

if __name__ == "__main__":
	start = time.clock()

	IMDB_corpus_train = pd.read_csv(r'/Users/vivek/git/A3_COMP_551/IMDB_Datasets/IMDB-train.txt',encoding='utf-8',header = None,sep='\t')
	IMDB_dictionary = create_dictionary(IMDB_corpus_train)
	IMDB_Bin_BoW_train = create_bin_bag_of_words(IMDB_corpus_train,IMDB_dictionary)
	print ("created Bag of Words for training ")
	IMDB_corpus_valid = pd.read_csv(r'/Users/vivek/git/A3_COMP_551/IMDB_Datasets/IMDB-valid.txt',encoding='utf-8',header = None,sep='\t')
	IMDB_Bin_BoW_valid = create_bin_bag_of_words(IMDB_corpus_valid, IMDB_dictionary)
	print ("created Bag of Words for validation")

	# random_f_score = random_classifier(IMDB_corpus_valid)
	# majority_f_score = majority_classifier(IMDB_corpus_valid)
	# naive_bayes_f_score = naive_bayes_classifier(IMDB_Bin_BoW_train,IMDB_dictionary,get_true_rating(IMDB_corpus_train),IMDB_Bin_BoW_valid, get_true_rating(IMDB_corpus_valid))

	# print ("random:	", random_f_score)
	# print ("majority:	", majority_f_score)
	# print ("bayes:	", naive_bayes_f_score)

	clf = BernoulliNB()
	clf.fit(IMDB_Bin_BoW_train,get_true_rating(IMDB_corpus_train))
	pred_arr = clf.predict(IMDB_Bin_BoW_valid)
	test_scores(pred_arr,get_true_rating(IMDB_corpus_valid))

	print (time.clock() - start)



