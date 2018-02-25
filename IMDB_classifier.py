# IMDB classifiers
import random
import pandas as pd
import numpy as np
import string
import collections
import time
import sklearn.metrics as sk
from sklearn.naive_bayes import BernoulliNB
import matplotlib.pyplot as plt


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

def do_Naive_Bayes_Bernoulli(train_BoW, testing_BoW, train_true_rating, testing_true_rating):
	smoothing = [1.0]
	f1_list = []
	for i in smoothing:
		print ("i:	",i)
		clf = BernoulliNB(alpha = i)
		clf.fit(train_BoW,train_true_rating)
		pred_arr = clf.predict(testing_BoW)
		f1_score = sk.f1_score(testing_true_rating, pred_arr, average='micro')
		f1_list.append(f1_score)
		# print (f1_score)
	
	# plt.plot(smoothing,f1_list)
	# plt.xlabel('Smoothing Coefficient - alpha')
	# plt.axvline(x=smoothing[np.argmax(f1_list)],linestyle='dashed', color = 'black')
	# plt.axhline(y=np.amax(f1_list),linestyle='dashed', color = 'black')
	# plt.ylabel('F-Measure')
	# plt.show()
	print (smoothing[np.argmax(f1_list)])
	print (np.amax(f1_list))

	


if __name__ == "__main__":
	start = time.clock()

	IMDB_corpus_train = pd.read_csv(r'/Users/vivek/git/A3_COMP_551/IMDB_Datasets/IMDB-train.txt',encoding='utf-8',header = None,sep='\t')
	IMDB_dictionary = create_dictionary(IMDB_corpus_train)
	IMDB_Bin_BoW_train = create_bin_bag_of_words(IMDB_corpus_train,IMDB_dictionary)
	print ("created Bag of Words for training ")
	IMDB_corpus_valid = pd.read_csv(r'/Users/vivek/git/A3_COMP_551/IMDB_Datasets/IMDB-valid.txt',encoding='utf-8',header = None,sep='\t')
	IMDB_Bin_BoW_valid = create_bin_bag_of_words(IMDB_corpus_valid, IMDB_dictionary)
	print ("created Bag of Words for validation")
	IMDB_corpus_test = pd.read_csv(r'/Users/vivek/git/A3_COMP_551/IMDB_Datasets/IMDB-test.txt',encoding='utf-8',header = None,sep='\t')
	IMDB_Bin_BoW_test = create_bin_bag_of_words(IMDB_corpus_test, IMDB_dictionary)

	do_Naive_Bayes_Bernoulli(IMDB_Bin_BoW_train, IMDB_Bin_BoW_test, get_true_rating(IMDB_corpus_train), get_true_rating(IMDB_corpus_test))
	
	print (time.clock() - start)