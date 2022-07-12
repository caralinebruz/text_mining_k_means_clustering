#!/usr/bin/python3

import logging
import string

import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

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


	def read_file(self):
		# read the file into an array of lines
		logger.info("opening file %s ...", self.filename)

		with open(self.filename) as f:
			self.lines = f.readlines()

			logger.info("line 0: %s", self.lines[0])


	def filter_stopwords_lemmatize(self):
		logger.info("removing stopwords ...")
		new_lines = []
		# lemmatizer = WordNetLemmatizer()
		# document = []

		for line in self.lines:
			# remove empty lines
			if line is not None:
				new_sentence = []

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

							# add it to new_sentence
							# new_sentence.append(new_word)

							# and add it to the text document
							self.document.append(new_word)

							logger.info("%s => %s" % (word,new_word))	

				# # if its not just a blank line
				# if not len(new_sentence) < 1:
				# 	# add it the sentence to the newlines
				# 	new_lines.append(new_sentence)
		self.document_text = ' '.join(self.document)
		# self.lines = new_lines

	def apply_ner(self):
		logger.info("applying NER ...")

		NER = spacy.load('en_core_web_sm')

		
		# print(document_text)

		mytext = NER(self.document_text)

		# print(mytext.ents)

		for ent in mytext.ents:
			# print(ent.text, ent.start_char, ent.end_char, ent.label_)
			logger.info("%s : %s" % (ent.text, ent.label_))

			# if there is one or more spaces in the ENT
				# then convert them to underscores in the document text






# preprocess the raw data

def do_preprocessing():

	for file in files[0:1]:
		# initialize a new array with each file for now
		lines = []

		# get the fullpath together
		filename = "./" + infile_path + file
		logger.info("Starting with file "  + filename);


		# now, instantiate a preprocess object
		P = Preprocessor(filename)

		# read the file
		P.read_file()

		# remove stopwords, lemmatize, and tokenize
		# https://www.geeksforgeeks.org/python-lemmatization-with-nltk/
		P.filter_stopwords_lemmatize()


		logger.info("\n")
		# print(P.document)

		# apply NER 
		# https://www.analyticsvidhya.com/blog/2021/06/nlp-application-named-entity-recognition-ner-in-python-with-spacy/#:~:text=Named%20Entity%20Recognition%20is%20the,%2C%20money%2C%20time%2C%20etc.
		P.apply_ner()





		# at the end, write to out_file for each document








if __name__ == '__main__':
	logger.info("starting ...");
	do_preprocessing()
















