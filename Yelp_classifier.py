# yelp classifers
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
	# smoothing = np.linspace(0,1,100)
	smoothing = [0.010101010101010102]
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
	# # plt.axvline(x=smoothing[np.argmax(f1_list) - 10],linestyle='dashed')
	# # plt.axvline(x=smoothing[np.argmax(f1_list) + 10],linestyle='dashed')
	# plt.ylabel('F-Measure')
	# # plt.annotate('local max', xy=(np.argmax(f1_list), np.amax(f1_list)),arrowprops=dict(facecolor='black', shrink=0.05))
	# plt.show()
	print (np.amax(f1_list))
	# print (smoothing[np.argmax(f1_list)])
	
def do_Decision_Trees(train_BoW, testing_BoW, train_true_rating, testing_true_rating):
	# parameters:
	# max_depth
	# min_samples_split
	# min_samples_leaf
	# min_weight_fraction_leaf
	# max_features
	# max_leaf_nodes
	# min_impurity_decrease
	# min_impurity_split
	f1 = []
	max_depth = np.linspace(1,1,1)
	for depth in max_depth:
		min_samples_split = np.linspace(2,1,1)
		for num_sample_split in min_samples_split:
			min_samples_leaf = np.linspace(1,1,1)
			for num_sample_leaf in min_samples_leaf:
				min_weight_fraction_leaf = np.linspace(0,1,1)
				for weight_fraction_leaf in min_weight_fraction_leaf:
					max_features = np.linspace(1,1,1)
					for num_features in max_features:
						max_leaf_nodes = np.linspace(1,1,1)
						for num_leaf_nodes in max_leaf_nodes:
							min_impurity_decrease = np.linspace(0,1,1)
							for impurity_decrease in min_impurity_decrease:
								clf = tree.DecisionTreeClassifier(max_depth = depth, min_samples_split = num_sample_split, min_samples_leaf = num_sample_leaf,min_weight_fraction_leaf = weight_fraction_leaf, max_features = num_features, max_leaf_nodes = num_leaf_nodes, min_impurity_decrease = impurity_decrease)
								print ("calculating")
								# print (time.clock() - start)
								clf = tree.DecisionTreeClassifier()
								clf.fit(train_BoW, train_true_rating)
								pred_arr = clf.predict(testing_BoW)
								f1_score = sk.f1_score(testing_true_rating, pred_arr, average='micro')
								print (f1_score)
								f1.append(f1_score)
		
	return (np.array(f1))

if __name__ == "__main__":
	start = time.clock()

	Yelp_corpus_train = pd.read_csv(r'/Users/vivek/git/A3_COMP_551/Yelp_Datasets/yelp-train.txt',encoding='utf-8',header = None,sep='\t')
	Yelp_dictionary = create_dictionary(Yelp_corpus_train)
	Yelp_Bin_BoW_train = create_bin_bag_of_words(Yelp_corpus_train,Yelp_dictionary)
	print ("created Bag of Words for training ")
	Yelp_corpus_valid = pd.read_csv(r'/Users/vivek/git/A3_COMP_551/Yelp_Datasets/yelp-valid.txt',encoding='utf-8',header = None,sep='\t')
	Yelp_Bin_BoW_valid = create_bin_bag_of_words(Yelp_corpus_valid, Yelp_dictionary)
	print ("created Bag of Words for validation")
	Yelp_corpus_test = pd.read_csv(r'/Users/vivek/git/A3_COMP_551/Yelp_Datasets/yelp-test.txt',encoding='utf-8',header = None,sep='\t')
	Yelp_Bin_BoW_test = create_bin_bag_of_words(Yelp_corpus_test, Yelp_dictionary)

	do_Naive_Bayes_Bernoulli(Yelp_Bin_BoW_train, Yelp_Bin_BoW_test, get_true_rating(Yelp_corpus_train), get_true_rating(Yelp_corpus_test))
	# print (np.amax(np.array(list)))
	# list = do_Decision_Trees(Yelp_Bin_BoW_train, Yelp_Bin_BoW_valid, get_true_rating(Yelp_corpus_train), get_true_rating(Yelp_corpus_valid))


	# smoothing = np.linspace(15,35,100)
	# print (smoothing[55])
	print (time.clock() - start)

