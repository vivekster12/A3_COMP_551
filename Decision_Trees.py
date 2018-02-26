# Decision Trees
import random
import pandas as pd
import numpy as np
import string
import collections
import time
import sklearn.metrics as sk
from sklearn.naive_bayes import BernoulliNB
from sklearn import tree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

def create_freq_bag_of_words(corpus,dictionary):
	translator = str.maketrans(string.punctuation.replace('\'',''),31*' ', '\'')
	freq_bag_of_words = []
	for i in range(len(corpus)):
		count = 0
		review = corpus.iloc[i,0].translate(translator).lower().split(" ")
		counter = collections.Counter(review)
		vector = np.zeros(len(dictionary))
		for index,word in enumerate(dictionary):
			count += counter[word]
			np.put(vector,index,counter[word])		
		
		if(count == 0):
			count = 1

		freq_bag_of_words.append(np.true_divide(vector,count))
	
	freq_bag_of_words = np.array(freq_bag_of_words)
	return (freq_bag_of_words)

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

# do Decision Trees
def do_DT(train_BoW, testing_BoW, train_true_rating, testing_true_rating, param_depth,param_leafs):
	f1_list = []
	
	for max_depth in param_depth:
		for max_leaf in param_leafs:
			clf = tree.DecisionTreeClassifier(max_depth = max_depth, max_leaf_nodes = max_leaf)
			print ("calculating DT")
			clf.fit(train_BoW, train_true_rating)
			pred_arr = clf.predict(testing_BoW)
			print (time.clock() - start)
			f1_score = sk.f1_score(testing_true_rating, pred_arr, average='micro')
			f1_list.append(f1_score)

	print ("Best f1_score")
	best_f1 = np.amax(f1_list)
	print (best_f1)
	print ("Best depth")
	best_depth = int((np.argmax(f1_list)) /len(param_leafs))
	print (best_depth)
	print ("Best num leaf")
	best_num_leaf = int((np.argmax(f1_list))%len(param_leafs))
	print (best_num_leaf)
	
	return (best_f1, best_depth, best_num_leaf)


if __name__ == "__main__":
	start = time.clock()
	
	# YELP
	# read/ set up
	# Yelp_corpus_train = pd.read_csv(r'/Users/vivek/git/A3_COMP_551/Yelp_Datasets/yelp-train.txt',encoding='utf-8',header = None,sep='\t')
	# Yelp_dictionary = create_dictionary(Yelp_corpus_train)
	# Yelp_corpus_valid = pd.read_csv(r'/Users/vivek/git/A3_COMP_551/Yelp_Datasets/yelp-valid.txt',encoding='utf-8',header = None,sep='\t')
	# Yelp_corpus_test = pd.read_csv(r'/Users/vivek/git/A3_COMP_551/Yelp_Datasets/yelp-test.txt',encoding='utf-8',header = None,sep='\t')
	
	# ****** Binary Bag of Words ******* #
	# Yelp_Bin_BoW_train = create_bin_bag_of_words(Yelp_corpus_train,Yelp_dictionary)
	# print ("created Bin Bag of Words for training ")
	# Yelp_Bin_BoW_valid = create_bin_bag_of_words(Yelp_corpus_valid, Yelp_dictionary)
	# print ("created Bin Bag of Words for validation")
	# Yelp_Bin_BoW_test = create_bin_bag_of_words(Yelp_corpus_test, Yelp_dictionary)
	# print ("created Bin Bag of Words for test")

	# Do decision trees
	# # param_depth = [None, 2, 10, 100]
	# # param_leafs = np.arange(2,21)
	# valid_f1_score, best_param_depth, best_param_leaf = do_DT(Yelp_Bin_BoW_train, Yelp_Bin_BoW_valid, get_true_rating(Yelp_corpus_train), get_true_rating(Yelp_corpus_valid), param_depth, param_leafs)
	# # run best param on test
	# # best_param_depth = [100]
	# # best_param_leaf = [18]
	# test_f1_score,scrap1,scrap2 = do_DT(Yelp_Bin_BoW_train, Yelp_Bin_BoW_test, get_true_rating(Yelp_corpus_train), get_true_rating(Yelp_corpus_test), best_param_depth, best_param_leaf)


	# ****** Frequency Bag of Words ******* #
	# Yelp_Freq_BoW_train = create_freq_bag_of_words(Yelp_corpus_train,Yelp_dictionary)
	# print ("created Freq Bag of Words for training ")
	# Yelp_Freq_BoW_valid = create_freq_bag_of_words(Yelp_corpus_valid, Yelp_dictionary)
	# print ("created Freq Bag of Words for validation")
	# Yelp_Freq_BoW_test = create_freq_bag_of_words(Yelp_corpus_test, Yelp_dictionary)
	# print ("created Freq Bag of Words for test")

	# Do decision trees
	# param_depth = [None, 2, 10, 100]
	# param_leafs = np.arange(2,21)
	# valid_f1_score, best_param_depth, best_param_leaf = do_DT(Yelp_Freq_BoW_train, Yelp_Freq_BoW_valid, get_true_rating(Yelp_corpus_train), get_true_rating(Yelp_corpus_valid), param_depth, param_leafs)
	# best_param_depth = [None]
	# best_param_leaf = [18]
	# test_f1_score,scrap1,scrap2 = do_DT(Yelp_Freq_BoW_train, Yelp_Freq_BoW_test, get_true_rating(Yelp_corpus_train), get_true_rating(Yelp_corpus_test), best_param_depth, best_param_leaf)

	# IMDB
	# read/ set up
	IMDB_corpus_train = pd.read_csv(r'/Users/vivek/git/A3_COMP_551/IMDB_Datasets/IMDB-train.txt',encoding='utf-8',header = None,sep='\t')
	IMDB_dictionary = create_dictionary(IMDB_corpus_train)
	IMDB_corpus_valid = pd.read_csv(r'/Users/vivek/git/A3_COMP_551/IMDB_Datasets/IMDB-valid.txt',encoding='utf-8',header = None,sep='\t')
	IMDB_corpus_test = pd.read_csv(r'/Users/vivek/git/A3_COMP_551/IMDB_Datasets/IMDB-test.txt',encoding='utf-8',header = None,sep='\t')
	
	# ****** Binary Bag of Words ******* #
	# IMDB_Bin_BoW_train = create_bin_bag_of_words(IMDB_corpus_train,IMDB_dictionary)
	# print ("created Bag of Words for training ")
	# IMDB_Bin_BoW_valid = create_bin_bag_of_words(IMDB_corpus_valid, IMDB_dictionary)
	# print ("created Bag of Words for validation")
	# IMDB_Bin_BoW_test = create_bin_bag_of_words(IMDB_corpus_test, IMDB_dictionary)
	# print ("created Bag of Words for test")

	# Do decision trees
	# param_depth = [None, 2, 10, 100]
	# param_leafs = np.arange(2,21)
	# valid_f1_score, best_param_depth, best_param_leaf = do_DT(IMDB_Bin_BoW_train, IMDB_Bin_BoW_valid, get_true_rating(IMDB_corpus_train), get_true_rating(IMDB_corpus_valid),param_depth, param_leafs)
	# best_param_depth = [None]
	# best_param_leaf = [17]
	# test_f1_score,scrap1,scrap2 = do_DT(IMDB_Bin_BoW_train, IMDB_Bin_BoW_test, get_true_rating(IMDB_corpus_train), get_true_rating(IMDB_corpus_test),best_param_depth,best_param_leaf)

	# # ****** Frequency Bag of Words ******* #
	IMDB_Freq_BoW_train = create_freq_bag_of_words(IMDB_corpus_train,IMDB_dictionary)
	print ("created Freq Bag of Words for training ")
	# IMDB_Freq_BoW_valid = create_freq_bag_of_words(IMDB_corpus_valid, IMDB_dictionary)
	# print ("created Freq Bag of Words for validation")
	IMDB_Freq_BoW_test = create_freq_bag_of_words(IMDB_corpus_test, IMDB_dictionary)
	print ("created Freq Bag of Words for test")

	# Do decision trees
	# param_depth = [None, 2, 10, 100]
	# param_leafs = np.arange(2,21)
	# valid_f1_score, best_param_depth, best_param_leaf = do_DT(IMDB_Freq_BoW_train, IMDB_Freq_BoW_valid, get_true_rating(IMDB_corpus_train),get_true_rating(IMDB_corpus_valid), param_depth, param_leafs)
	
	print (time.clock() - start)




