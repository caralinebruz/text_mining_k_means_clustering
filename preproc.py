#!/usr/bin/python3

import logging
import string

import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from nltk.util import ngrams
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

import spacy
from spacy import displacy
# en-core-web-sm-3.4.0
# python -m spacy download en_core_web_sm




logging.basicConfig(
	format='%(asctime)s %(levelname)-8s %(message)s',
	level=logging.INFO,
	datefmt='%Y-%m-%d %H:%M:%S')

logger = logging


infile_path = "data/"
outfile_path = "data/out/"
files = [
	"C1/article01.txt",
	"C1/article02.txt",
	"C1/article03.txt",
	"C1/article04.txt",
	"C1/article05.txt",
	"C1/article06.txt",
	"C1/article07.txt",
	"C1/article08.txt"
]


class Preprocessor():
	def __init__(self, filename):
		self.filename = filename
		self.lines = []
		self.stop_words = set(stopwords.words('english'))
		self.lemmatizer = WordNetLemmatizer()
		self.document = []
		self.document_text = ""
		self.ngrams = []
		self.ngrams_frequency = {}


	def read_file(self):
		# read the file into an array of lines
		logger.info("opening file %s ...", self.filename)

		with open(self.filename) as f:
			self.lines = f.readlines()

			logger.info("line 0: %s", self.lines[0])


	def filter_stopwords_lemmatize(self):
		logger.info("removing stopwords ...")
		new_lines = []

		for line in self.lines:
			if line is not None:

				# split and tokenize
				old_sentence = word_tokenize(line)

				for word in old_sentence:

					# remove punctuation
					# https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string
					exclude = set(string.punctuation)
					word = ''.join(ch for ch in word if ch not in exclude)

					# if its not empty 
					if word is not None and len(word) > 0:

						# remove stopwords
						if word not in self.stop_words:

							# lemmatize 
							# best guess here to treat anything ending in 's'
							#	as a noun, anything else gets verb treatment
							new_word = word
							if word.endswith('s'):
								new_word = self.lemmatizer.lemmatize(word)
							else:
								new_word = self.lemmatizer.lemmatize(word, "v")

							## not entirely sure if i should be lowercasing everything
							# new_word = new_word.lower()

							# and add it to the text document
							self.document.append(new_word)
							logger.info("%s => %s" % (word,new_word))	

		self.document_text = ' '.join(self.document)


	def apply_ner(self):
		logger.info("applying NER ...")
		NER = spacy.load('en_core_web_sm')

		mytext = NER(self.document_text)

		logger.info("Found the following entities:")
		for ent in mytext.ents:
			# print(ent.text, ent.start_char, ent.end_char, ent.label_)
			logger.info("\t %s : %s" % (ent.text, ent.label_))
			this_ent = ent.text

			# if there is one or more spaces in the ENT
			if " " in this_ent:
				# then convert them to underscores in the document text
				new_ent = this_ent.replace(" ","_")
				# print(new_ent)

				# then also replace the original text document
				self.document_text = self.document_text.replace(this_ent, new_ent)

		# also update the tokenized array
		self.document = word_tokenize(self.document_text)


	# https://www.geeksforgeeks.org/python-bigrams-frequency-in-string/
	def _find_bi_grams(self, text):

		bigrams = zip(text, text[1:])
		for gram in bigrams:

			bigram_string = ' '.join(gram)
			self.ngrams.append(bigram_string)


	def _find_tri_grams(self, text):
		# this doesnt seem to be producing as meaningful result as the bigram :/

		trigrams = zip(text, text[1:], text[2:])
		for gram in trigrams:

			trigram_string = ' '.join(gram)
			self.ngrams.append(trigram_string)


	def sliding_window_merge(self):
		logger.info("using a sliding window to merge remaining phrases ...")

		# ****************************************************
		# BI-GRAMS VS TRI-GRAMS ::
		# 
		# 	I won't use trigrams bc frequencies arent as good
		#		but logic for it is here in this block
		#
		#
		# self.ngrams = []
		#
		# self._find_tri_grams(self.document)
		#
		# for ngram in self.ngrams:
		# 	frequency = self.document_text.count(ngram)
		#
		# 	self.ngrams_frequency['ngram'] = frequency
		# 	print("%s : %s "% (ngram, frequency))
		#
		# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

		logger.info("using bi-grams for this, there are more matches...")
		# ngram_dist = nltk.FreqDist(nltk.bigrams(self.document))
		# print(ngram_dist.most_common())

		# i will pick everything with freq > 1 for the merge

		self._find_bi_grams(self.document)
		# print(self.ngrams)

		# dedupe the ngrams
		self.ngrams = list(dict.fromkeys(self.ngrams))

		for ngram in self.ngrams:
			frequency = self.document_text.count(ngram)
		
			self.ngrams_frequency['ngram'] = frequency
			# print("%s : %s "% (ngram, frequency))

			# if frequency > 1, merge
			if frequency > 1:
				new_ngram = ngram.replace(" ","_")

				# then also replace the original text document
				self.document_text = self.document_text.replace(ngram, new_ngram)	
				







# preprocess the raw data

def do_preprocessing():

	for file in files[0:1]:

		# get the fullpath together
		filename = "./" + infile_path + file
		logger.info("Starting with file "  + filename);

		# now, instantiate a preprocess object
		P = Preprocessor(filename)

		# read the file
		P.read_file()

		# 2 - remove stopwords, lemmatize, and tokenize
		# https://www.geeksforgeeks.org/python-lemmatization-with-nltk/
		P.filter_stopwords_lemmatize()


		# 3 - apply NER 
		# https://www.analyticsvidhya.com/blog/2021/06/nlp-application-named-entity-recognition-ner-in-python-with-spacy/#:~:text=Named%20Entity%20Recognition%20is%20the,%2C%20money%2C%20time%2C%20etc.
		P.apply_ner()

		

		# 4 - use sliding window approach to merge remaining phrases
		P.sliding_window_merge()

		print(P.document_text)



		# 5 - at the end, write to out_file for each document








if __name__ == '__main__':
	logger.info("starting ...");
	do_preprocessing()
















