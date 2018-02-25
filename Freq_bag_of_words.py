# frequncy bag-of-words
import pandas as pd
import numpy as np
import string
import collections
import time
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
import sklearn.metrics as sk

# returns dictionary of 10,000 most common words
# returns frequncy bag of words
def create_dictionary(corpus):
	# pre-processing for creating dictionary
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

	print ("dictionary created")
	# create bag of words
	return(dict)

def create_freq_bag_of_words(corpus,dictionary):
	print ("creating bag of words")
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

def get_true_rating(corpus):
	true_rating = []
	for i in range(len(corpus)):
		true_rating.append(corpus.iloc[i,1])

	return (true_rating)

def do_Guassian_Bayes_Bernoulli(train_BoW, testing_BoW, train_true_rating, testing_true_rating):
	print ("doing Bayes")
	clf = GaussianNB()
	clf.fit(train_BoW,train_true_rating)
	pred_arr = clf.predict(testing_BoW)
	f1_score = sk.f1_score(testing_true_rating, pred_arr, average='micro')
	print (f1_score)
	

if __name__ == "__main__":
	start = time.clock()

	# yelp_corpus_train = pd.read_csv(r'/Users/vivek/git/A3_COMP_551/Yelp_Datasets/yelp-train.txt',encoding='utf-8',header = None,sep='\t')
	# yelp_dictionary= create_dictionary(yelp_corpus_train)
	# train_freq_bag_of_words = create_freq_bag_of_words(yelp_corpus_train,yelp_dictionary)
	# yelp_corpus_valid = pd.read_csv(r'/Users/vivek/git/A3_COMP_551/Yelp_Datasets/yelp-valid.txt',encoding='utf-8',header = None,sep='\t')
	# valid_freq_bag_of_words = create_freq_bag_of_words(yelp_corpus_valid, yelp_dictionary)
	# yelp_corpus_test = pd.read_csv(r'/Users/vivek/git/A3_COMP_551/Yelp_Datasets/yelp-test.txt',encoding='utf-8',header = None,sep='\t')
	# test_freq_bag_of_words = create_freq_bag_of_words(yelp_corpus_test, yelp_dictionary)

	# do_Guassian_Bayes_Bernoulli(train_freq_bag_of_words, test_freq_bag_of_words,get_true_rating(yelp_corpus_train),get_true_rating(yelp_corpus_test))
	
	IMDB_corpus_train = pd.read_csv(r'/Users/vivek/git/A3_COMP_551/IMDB_Datasets/IMDB-train.txt',encoding='utf-8',header = None,sep='\t')
	IMDB_dictionary = create_dictionary(IMDB_corpus_train)
	train_freq_bag_of_words = create_freq_bag_of_words(IMDB_corpus_train,IMDB_dictionary)
	IMDB_corpus_valid = pd.read_csv(r'/Users/vivek/git/A3_COMP_551/IMDB_Datasets/IMDB-valid.txt',encoding='utf-8',header = None,sep='\t')
	valid_freq_bag_of_words = create_freq_bag_of_words(IMDB_corpus_valid, IMDB_dictionary)
	IMDB_corpus_test = pd.read_csv(r'/Users/vivek/git/A3_COMP_551/IMDB_Datasets/IMDB-test.txt',encoding='utf-8',header = None,sep='\t')
	test_freq_bag_of_words = create_freq_bag_of_words(IMDB_corpus_test, IMDB_dictionary)

	print ("valid")
	do_Guassian_Bayes_Bernoulli(train_freq_bag_of_words, valid_freq_bag_of_words,get_true_rating(IMDB_corpus_train),get_true_rating(IMDB_corpus_valid))
	print ("test")
	do_Guassian_Bayes_Bernoulli(train_freq_bag_of_words, test_freq_bag_of_words,get_true_rating(IMDB_corpus_train),get_true_rating(IMDB_corpus_test))
	
	# pd.DataFrame(data = np.array(IMDB_freq_bag_of_words)).to_csv(r'/Users/vivek/git/A3_COMP_551/IMDB_Datasets/IMDB-freq_BoW.csv',index = False, header = labels)
	
	print (time.clock() - start)