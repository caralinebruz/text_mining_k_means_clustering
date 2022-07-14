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

import itertools



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

my_process_objects = []




class Preprocessor():

	# easy way to increment an index of the class instance
	# https://stackoverflow.com/questions/1045344/how-do-you-create-an-incremental-id-in-a-python-class
	index = itertools.count()

	def __init__(self, file):
		self.filename = "./" + infile_path + file
		self.out_filename = "./" + outfile_path + file
		self.lines = []
		self.stop_words = set(stopwords.words('english'))
		self.lemmatizer = WordNetLemmatizer()
		self.document = []
		self.document_text = ""
		self.keywords_concepts = []
		self.ngrams = []
		self.ngrams_frequency = {}
		self.index = next(Preprocessor.index)

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
							# logger.info("%s => %s" % (word,new_word))	

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

				# save the ENT for later matrix
				self.keywords_concepts.append(new_ent)

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
		
			self.ngrams_frequency[ngram] = frequency
			#print("%s : %s "% (ngram, frequency))

			# if frequency > 1, merge
			if frequency > 1:
				new_ngram = ngram.replace(" ","_")

				# save the NGRAM for later matrix
				self.keywords_concepts.append(new_ngram)

				# then also replace the original text document
				self.document_text = self.document_text.replace(ngram, new_ngram)	

				print("\t\t %s : %s "% (ngram, frequency))


	def cleanup(self):

		for i in range(len(self.keywords_concepts)):
			self.keywords_concepts[i] = self.keywords_concepts[i].replace("_"," ").lower()
			# print(self.keywords_concepts[i])

		self.ngrams_frequency = {k.replace("_"," ").lower() : v for k, v in self.ngrams_frequency.items()}
		# print(self.ngrams_frequency.items())

		self.document_text = self.document_text.replace("_"," ").lower()
		# print(self.document_text)

				
	def write_output(self):

		logger.info("Writing output file "  + self.out_filename);

		# WRITE THIS FILE WITHOUT ANY UNDERSCORES

		with open(self.out_filename, "w") as outfile:
			for word in self.document_text:

				outfile.write(word.lower())


def write_keywords_concepts_file(P):

	logger.info("Appending to concepts file ...")
	concepts_file = "./" + outfile_path + "concepts.txt"
	
	with open(concepts_file, "a") as f:

		lines = P.keywords_concepts
		for line in lines:
			# print(line)
			f.write(line)
			f.write("\n")



# preprocess the raw data
def do_preprocessing():

	for file in files[0:3]:
	#for file in files:

		P = Preprocessor(file)

		# and add that object to the processed objects list
		my_process_objects.append(P)

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

		# clean up my findings:
		# 	removes underscores from document text, lowercases
		#	removes underscores from frequency keys, lowercases
		P.cleanup()

		# 5 - at the end, write to out_file for each document for safety
		P.write_output()

		# also write the keywords concepts file
		write_keywords_concepts_file(P)

	return my_process_objects



# # collect keywords and terms across all the files
# def generate_term_document_matrix():
class DocuTermMatrix():

	def __init__(self):
		self.keywords_concepts = []
		self.matrix = []

	def consolidate_keywords_concepts(self):
		# read the file into an array of lines
		logger.info("Collecting all of the keywords concepts ...")
		for file_object in my_process_objects:

			for keyword in file_object.keywords_concepts:
				if keyword not in self.keywords_concepts:

					self.keywords_concepts.append(keyword.lower())


	def initialize_matrix(self):
		logger.info("initializing the zero matrix ...")

		# fill with 0s for the correct size matrix
		num_rows = len(my_process_objects)
		num_cols = len(self.keywords_concepts)

		# https://intellipaat.com/community/63426/how-to-create-a-zero-matrix-without-using-numpy
		self.matrix = [([0]*num_cols) for i in range(num_rows)]


	def fill_matrix(self):

		logger.info("Creating the document term matrix ...")
		i = 0
		for i in range(len(my_process_objects)):

			# print(i)
			# print(my_process_objects[i].index)
			# print(my_process_objects[i].filename)

			file_object = my_process_objects[i]

			# convert all the keys to lowercase for now
			the_files_ngrams =  {k.lower(): v for k, v in file_object.ngrams_frequency.items()}
			# print(file_object.ngrams_frequency)


			# iterate over the keywords_concepts list
			for j in range(len(self.keywords_concepts)):

				# if a keyword_concept is in the document_text of the document
				# count the number of times the substring appears
				if self.keywords_concepts[j] in file_object.document_text:

					# print("\n TRUE !")
					# https://stackoverflow.com/questions/8899905/count-number-of-occurrences-of-a-substring-in-a-string
					frequency = file_object.document_text.count(self.keywords_concepts[j])
					self.matrix[i][j] = frequency

				# else:
				# 	print("%s not in document text" % self.keywords_concepts[j])
				# 	print(file_object.document_text)




def generate_document_term_matrix():

	M = DocuTermMatrix()

	# firt, consolidate and dedupe all keywords across the files
	M.consolidate_keywords_concepts()

	# second, create the matrix
	#	rows will be filenames (file_object.index)
	#	columns will be keywords (keywords_concepts)
	#	populate cell value based on the FREQUENCY of that keyword in the file


	print(M.keywords_concepts)

	# second, create the matrix
	M.initialize_matrix()
	M.fill_matrix()

	# print(M.matrix)
	for k in range(len(M.matrix)):
		print(M.matrix[k])





if __name__ == '__main__':

	logger.info("starting ...");
	# global file_object_index
	# file_object_index = 0

	# does preprocessing on the files
	# returns a list of preprocessed file objects for each file
	logger.info("First: Do preprocessing on the files")
	processed_objects = do_preprocessing()

	# then give it the matrix class here
	logger.info("Next: Generating Document Term Matrix")
	generate_document_term_matrix()

























































