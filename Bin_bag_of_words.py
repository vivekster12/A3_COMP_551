# binary bag-of-words
import pandas as pd
import numpy as np
import string
import collections
import time
import sklearn.metrics as sk

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
		print (index, word)
		dict[word] = index

	return (dictionary,dict)

def create_bin_bag_of_words(corpus,dictionary):
	term_doc_mat = []
	for i in range(len(corpus)):
		words_in_sample = (corpus.iloc[i,0].split(" "))
		vector = np.zeros(len(dictionary))
		for word in words_in_sample:
			if (word in dictionary):
				np.put(vector,dictionary[word],1)	
		term_doc_mat.append(vector)

	# print (np.array(term_doc_mat).shape)
	return (np.array(term_doc_mat))

def get_true_rating(corpus):
	true_rating = []
	for i in range(len(corpus)):
		true_rating.append(corpus.iloc[i,1])

	return (true_rating)

if __name__ == "__main__":
	start = time.clock()

	# yelp_corpus = pd.read_csv(r'/Users/vivek/git/A3_COMP_551/Yelp_Datasets/yelp-train.txt',encoding='utf-8',header = None,sep='\t')
	# labels, yelp_dictionary = create_dictionary(yelp_corpus)
	# yelp_term_doc_mat = create_bin_bag_of_words(yelp_corpus,yelp_dictionary)
	# pd.DataFrame(data = np.array(yelp_term_doc_mat)).to_csv(r'/Users/vivek/git/A3_COMP_551/Yelp_Datasets/yelp-Bin_BoW.csv', index = False, header = False)

	# IMDB_corpus = pd.read_csv(r'/Users/vivek/git/A3_COMP_551/IMDB_Datasets/IMDB-train.txt',encoding='utf-8',header = None,sep='\t')
	# labels, IMDB_dictionary = create_dictionary(IMDB_corpus)
	# IMDB_Bin_BoW = create_bin_bag_of_words(IMDB_corpus,IMDB_dictionary)
	# print (yelp_dictionary)
	# pd.DataFrame(data = np.array(IMDB_Bin_BoW)).to_csv(r'/Users/vivek/git/A3_COMP_551/IMDB_Datasets/IMDB-Bin_BoW.csv', index = False, header = labels)
	
	
	print (time.clock() - start)
