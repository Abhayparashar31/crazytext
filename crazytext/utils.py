from operator import index
import re
import os
import sys
import json

import pandas as pd
import numpy as np

import nltk
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import PorterStemmer

import unicodedata
from textblob import TextBlob




path = os.path.dirname(os.path.abspath(__file__))
abbreviations_path = os.path.join(path, 'data','abbreviations_wordlist.json')



################## STOPWORDS ##########################

stopwords = set(
    """
a about above across after afterwards again against all almost alone along
already also although always am among amongst amount an and another any anyhow
anyone anything anyway anywhere are around as at

back be became because become becomes becoming been before beforehand behind
being below beside besides between beyond both bottom but by

call can cannot ca could

did do does doing done down due during

each eight either eleven else elsewhere empty enough even ever every
everyone everything everywhere except

few fifteen fifty first five for former formerly forty four from front full
further

get give go

had has have he hence her here hereafter hereby herein hereupon hers herself
him himself his how however hundred

i if in indeed into is it its itself

keep

last latter latterly least less

just

made make many may me meanwhile might mine more moreover most mostly move much
must my myself

name namely neither nevertheless next nine noone 
nothing now nowhere

of off often on once one only onto or other others otherwise our ours ourselves
out over own o oo

part per perhaps please put

quite

rather re really regarding

same say see seem seemed seeming seems serious several she should show side
since six sixty so some somehow someone something sometime sometimes somewhere
still such

take ten than that the their them themselves then thence there thereafter
thereby therefore therein thereupon these they third this those though three
through throughout thru thus to together too top toward towards twelve twenty
two

under until up unless upon us used using

various very very via was we well were what whatever when whence whenever where
whereafter whereas whereby wherein whereupon wherever whether which while
whither who whoever whole whom whose why will with within without would

yet you your yours yourself yourselves
""".split()
)




#########################################################################################################
#######################################---- COUNTER CLASS -------- ######################################
#########################################################################################################

class Counter(object):
	""" Count The Occurence of Different Parts of Text.
		Eg:Count Number of  Emails, Stopwords, HTML Tags, Emojies, etc.

		parameters
		-----------------
		self.text :: Main Text

		usage
		-----------------
		doc = Counter(text="YOUR TEXT")

	"""
	def __init__(self, text=None):
		super(Counter, self).__init__()
		self.text = text
	
	def __repr__(self):
		return f'Counter(text="{self.text}")'

	def count_words(self):
		"""It take a stringh and return number of words in it.

			Process
			----------------------
			
			Eg->  x = 'Python Programming Language'

			STEP 1: Convert Into String
			x = str(x)

			STEP 2: Split The Data
			x = x.split()                 ## ['Python','Programming','Language']
			
			STEP 3: Get The Length
			length = len(x)               ## 3


			Return
			----------------------
			int :: NUMBER OF WORDS

		"""

		x = self.text.lower()
		length = len(str(x).split())      ### split the data and count characters
		return length
		
	def count_characters(self):
		"""Return Number of Characters in a string.

			Process
			------------------

			x = 'Python Programming Language'

			STEP 1: Split the data
			x = x.split()                  ## ['Python','Programming','Language']

			STEP 2: Joint The List of Strings Into One String With No Space
			x = ''.join(x)                ## PythonProgrammingLanguage

			STEP 3: Return The Length of the string
			len(x)                        ## 25


			Return
			-----------------
			int :: NUMBER OF CHARACTERS

		"""


		x = self.text.lower()
		x_splitted = x.split()
		x = ''.join(x_splitted)
		return len(x)
	
	def count_avg_wordlength(self):
		"""Return Average Word Lenght

			Process
			-----------------------

			x = 'Python Programming Language'

			char_count = 25
			word_count = 3

			average_word_count = 25/3  = 8.33

			Return
			-------------------------
			float :: AVERAGE WORD LENGTH

		"""
		avg_count = self.count_characters()/self.count_words()
		return avg_count

	def count_stopwords(self):
		""" Return Length of Stopwords Inside The Text

			Process
			--------------------

			STEP 1: Splitting The Text
			STEP 2: Checking Whether Each Splitted Word Is Stopword or not.
			STEP 3: If It is a stopword then adding it to the list
			STEP 4: Returning The Length of List


			Return
			---------------------
			int :: NUMBER OF STOPWORDS
		"""

		x = self.text.lower()
		lst = len([t for t in x.split() if t in stopwords])
		return lst
	
	def count_hashtags(self):
		""" Return The Number of Hastag's Inside Your Text
		
			Process
			--------------
			Return The Number of Text That Starts With an `#` using 
			Python String Function `startswith`.

			Return
			---------------
			int :: NUMBER OF HASHTAGS
			
		"""
		x = self.text
		lst = len([t for t in x.split() if t.startswith('#')])
		return lst

	def count_mentions(self):
		"""Return Number of Mentions Inside Your Text
		
			Process
			------------
			Return The Count of All The Text That Starts With an '@' using 
			Python String Function `startswith`.

			Return
			------------
			int :: NUMBER OF MENTIONS

		"""
		x = self.text
		lst = len([t for t in x.split() if t.startswith('@')])
		return lst
		
	def count_digits(self):
		""" Return Number of Digits Inside Your Text
		
			Process
			---------------
			Compare Each Part of Text With Regrex `[0-9]` and Return The Count.

			
			Return
			-----------
			int :: NUMBER OF DIGITS 
		"""
		x = self.text
		digits = re.findall(r'[0-9]+', x)
		return len(digits)
	
	def count_uppercase_words(self):
		""" Return The Count of All UPEER CASE words.

			Process
			-------------------
			Check Whether Each Word Inside the Text is UPPER CASE or Not Using 
			Python String `isupper()` function.

			Return
			-------------------
			int :: NUMBER OF UPPERCASE WORDS
		
		"""
		x = self.text
		return len([word for word in x.split() if word.isupper()])

	def count_whitespace(self):
		""" Return The Number of White Space Inside Your Text.

			Process
			-------------------
			Check For Each Character Inside The Data To Be A White Space.

			Return
			-------------------
			int :: NUMBER OF WHITE SPACES
		
		"""
		count = 0
		x = self.text
		for char in x:
			if char==" ":
				count+=1
		return count

	def count_vowels(self):
		""" Return The Number of Vowels Inside Your Text.

			Process
			-------------------
			Check For Each Character To Be A Vowel.

			Return
			-------------------
			int :: NUMBER OF Vowels
			
		"""
		count = 0
		x = self.text
		for char in x:
			if char in ['a','e','i','o','u']:
				count+=1
		return count

	def count_consonants(self):
		""" Return The Number of Consonanat Inside Your Text.

			Process
			-------------------
			Check For Each Character To Be A Consonant.

			Return
			-------------------
			int :: NUMBER OF CONSONANTS
				
		"""
		count = 0
		x = self.text
		for char in x:
			if char not in ['a','e','i','o','u']:
				count+=1
		return count

	def count_emails(self):
		""" Return The Count of All The Emails.

			Process
			-------------------
			Count All The Emails Inside The Text and Return The Count.

			Return
			-------------------
			int :: NUMBER OF EMAILS
		
		"""
		x = self.text
		email = re.findall(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+\b)', x)
		return len(email)

	def count_urls(self):
		"""Count The Number of URLs Inside The Text

			Return
			-----------------
			int :: NUMBER OF URLs
		
		"""
		x = self.text
		urls = re.findall(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', x)
		count = len(urls)

		return count

	def count_special_chars(self):
		"""Count The Number of Special Character Inside The Text

			Example of Special Characters : &,^,@,(,),[,], %, etc.

			Return
			-----------------
			int :: SPECIAL CHARS COUNT
		"""
		x = self.text
		x = re.sub(r'[^\w ]+', "", x)

		return len(x)

	def count_phone_numbers(self):
		"""Count The Occurence of Phone Numbers Inside The Text

		Return
		----------------------
		int :: PhoneNumbers Count 
		"""
		x = self.text
		phone = re.compile(r"[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]")
		return len(re.findall(phone,x))

	def count_html_tags(self):
		"""Counts Number of HTML Tags From The Text

			Return
			---------------
			int :: HTML Tags Count
		
		"""
		x = self.text
		clean = re.compile('<.*?>')
		return len(re.findall(clean,x))


	################  DESCRIBE DATA ##################
	def info(self):
		print(f"Length of String: {self.count_characters()}")
		print(f"Number of URLs: {self.count_urls()}")
		print(f'Number of Emails: {self.count_emails()}')
		print(f'Number of Words: {self.count_words()}')
		print(f'Average Word Count: {self.count_avg_wordlength()}')
		print(f'Number of Stopwords: {self.count_stopwords()}')
		print(f'Total Hashtags: {self.count_hashtags()}')
		print(f'Total Mentions: {self.count_mentions()}')
		print(f'Total Length of Numeric Data: {self.count_digits()}')
		print(f'Special Characters: {self.count_special_chars()}')
		print(f'White Spaces: {self.count_whitespace()}')
		print(f'Number of Vowels: {self.count_vowels()}')
		print(f'Number of Consonants: {self.count_consonants()}')
		print(f'Total Uppercase Words {self.count_uppercase_words()}')
		print(f'Number of Phone Number Inside Text: {self.count_phone_numbers()}')
		print(f'Number of HTML Tags {self.count_html_tags()}')

		print(f'Observed Sentiment: {Extractor(text=self.text).get_sentiment()}')

#########################################################################################################
#######################################---- EXTRACTOR CLASS -------- ####################################
#########################################################################################################

class Extractor(Counter):
	""" Extractor Class For Extracting Different ELements of Text.
		Eg: HTML tags, Emails, Digits, Emojis, Stopwords etc.

		parameters
		----------------
		self.text :: Main Text

		usage
		----------------
		doc = Extractor(text="YOUR TEXT")

	"""
	def __init__(self, text=None):
		super(Extractor, self).__init__()
		self.text = text
	
	def __str__(self):
		return f"{self.text}"
	
	def __repr__(self):
		return f'Extractor(text="{self.text}")'

	def get_stopwords(self):
		"""Return Stopwords From The Text Inside a List

			Process
			-----------------

			STEP 1: Splitting The Text
			STEP 2: Checking Whether Each Splitted Word Is Stopword or not.
			STEP 3: If It is a stopword then adding it to the list
			STEP 4: Returning The List

			
			Return
			-----------
			List[STOPWORDS]

		"""
		x = self.text.lower()

		lst = [text for text in x.split() if text in stopwords]
		return lst

	def get_sentence_tokens(self):
		"""Generate Sentence Tokens And Return Inside A List

		Return
		------------------
		LIST[SENTENCE TOKENS]

		"""
		return sent_tokenize(self.text)
	
	def get_word_tokens(self):
		"""Generate Work Tokens And Return Inside A List


		Return
		--------------------
		LIST[WORD TOKENS]

		"""
		return word_tokenize(self.text)

	def get_hashtags(self):
		""" Return All The Hastag's From Your Text Inside A List
		
			Process
			--------------
			Return The Number of Text That Starts With an `#` using 
			Python String Function `startswith`.

			Return
			-----------
			List[HASHTAGS]
		
		"""
		x = self.text
		hashtags = [word for word in x.split() if word.startswith('#')]
		return hashtags
	
	def get_mentions(self):
		"""Return All Mentions From Your Text Inside A List
			
			Process
			------------
			Return All The Text That Starts With an '@' using 
			Python String Function `startswith` Inside a List.

			
			Return
			-----------
			List[MENTIONS]

		"""
		x = self.text
		mentions = [word for word in x.split() if word.startswith('@')]
		return mentions

	def get_digits(self):
		"""Retunrn List of All Digits Appear Inside The Text
		
			Process
			---------------
			Compare Each Part of Text With Regrex `[0-9]` and Return It if Matches.

			
			Return
			-----------
			LIST[DIGITS]
		
		"""
		x = self.text
		digits = re.findall(r'[0-9]+', x)
		return digits
	
	def get_uppercase_words(self):
		""" Return All UPEER CASE Words Inside A List.

			Process
			-------------------
			Check Whether Each Word Inside the Text is UPPER CASE or Not Using 
			Python String `isupper()` function.

			Return
			-------------------
			LIST[UPPERCASE WORDS]
		
		"""
		x = self.text
		return [word for word in x.split() if word.isupper()]
	
	def get_emails(self):
		""" Return All The Emails Inside A List.

			Process
			-------------------
			Looks For Emails Inside The Text and Return Them inside A List.

			Return
			-------------------
			LIST[EMAILS]
		
		"""
		x = self.text
		emails = re.findall(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+\b)', x)
		return emails

	def get_urls(self):
		"""Return All The URLs From The Text

			Return
			--------------------
			LIST[URLS]
			
		"""
		x = self.text
		urls = re.findall(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', x)
		return urls

	def get_repeated_words(self,n):
		''' Return All The Repeated Words With Their Fequency
		
			```
			get_repeated_words(self,n)
			
			Paramaters
			----------------
			n = number of repeated words to be extract
			```
		
		'''
		x = self.text
		from collections import Counter
		x = x.split()
		
		frequent = Counter(x).most_common(n)
		return frequent

	def get_correct_spelling(self):
		"""Correct The Misspelled Words

		Return
		--------------
		Text With No Misspelled Word
		
		"""
		x = self.text
		x = TextBlob(x).correct()
		return x

	def get_phone_numbers(self):
		"""Return All The Phone Numbers Inside A List
		"""
		x = self.text
		phone = re.compile(r"[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]")
		return re.findall(phone,x)

	def get_html_tags(self):
		"""Extracts HTML Tags From The Text

			Return
			---------------
			LIST[HTML TAGS]
		
		"""
		x = self.text
		clean = re.compile('<.*?>')
		return re.findall(clean,x)


	def get_sentiment(self):
		"""Return The Sentiment of The Text

		Return
		--------------
		Tuple With Score and Sentiment.

		"""
		x = self.text
		score = TextBlob(x).sentiment.polarity
		if score==0: 
			sentiment = "Neutral"
		elif score<0:
			sentiment = "Negative"
		elif score>0: 
			sentiment = "Positive"
		return score, sentiment


#########################################################################################################
####################################------- CLEANER CLASS -----------####################################
#########################################################################################################
class Cleaner(Extractor):
	""" Clean Text By Removing Certain Parts Of The Text.
		Eg: Removing Emails, Stopwords, HTML Tags, Emojies, etc.

		parameters
		-----------------
		self.text :: Main Text

		usage
		-----------------
		doc = Cleaner(text="YOUR TEXT")

	"""
	def __init__(self, text=None):
		super(Cleaner, self).__init__()
		self.text = text
	
	def __str__(self):
		return f"{self.text}"
	
	def __repr__(self):
		return f'Cleaner(text="{self.text}")'
	
	def remove_stopwords(self):
		""" Remove All Stopwords From The Text

			Process
			-------------------

			STEP 1: Splitting The Text
			STEP 2: Checking Whether Each Splitted Word Is Stopword or not.
			STEP 3: If It is a stopword then remove it.
			STEP 4: Return Cleaned Text.


			Return
			--------------------
			Text With No Stopwords
		
		"""
		x = self.text
		return ' '.join([word for word in x.split() if word not in stopwords])	

	def remove_digits(self):
		""" Remove All The Digits From The Text
				
			Process
			---------------
			Compare Each Part of Text With Regrex `[0-9]` and Remove It if Matches.

			
			Return
			-----------
			Text Without Digits ( Numeric Data )
		
		
		"""
		x = self.text
		return re.sub('[0-9]',"",x)

	def remomve_abber(self):
		""" Replace Abbreviations Words With Their Full Form.

			Example of Abberviations : 
			TTL : Talk To You Later, KG: Kilo Gram etc.
				
			Process
			---------------
			Compare Each Word With The Abberviation List and If Matches Replace it.

			
			Return
			-----------
			Text Without Abberviations
			
		"""

		abbreviations = json.load(open(abbreviations_path))
		x = self.text

		if type(x) is str:
			for key in abbreviations:
				value = abbreviations[key]
				x = x.replace(key, value)
			return x
		else:
			return x

	def remove_emails(self):
		""" Remove All The Emails From The Text and Replace Them With EMAIL.

			Process
			-------------------
			Looks For Emails Inside The Text and Replace Them With EMAIL

			Return
			-------------------
			CLEANED TEXT (xyz@gmail.com -> EMAIL)
		
		"""
		x = self.text
		return re.sub(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+)',"EMAIL", x)

	def remove_emails_c(self):
		""" Remove All The Emails From The Text Completly.

			Process
			-------------------
			Looks For Emails Inside The Text and Remove Them Completly.

			Return
			-------------------
			CLEANED TEXT
		
		"""
		x = self.text
		return re.sub(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+)',"", x)

	def remove_urls(self):
		"""Replace All The URLs From The Text With URL

		Return
		--------------
		CLEANED TEXT (https://xyz.com -> URL)
		
		"""
		x = self.text
		return re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', 'URL ' , x)

	def remove_urls_c(self):
		"""Remove All The URLs From The Text.

			Return
			--------------
			CLEANED TEXT
			
		"""
		x = self.text
		return re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '' , x)

	def remove_special_chars_c(self):
		"""Remove All The Special Characters
		
		
			Example of Special Characters : &,^,@,(,),[,], %, etc.


			Return
			------------------
			CLEANED TEXT 
		"""
		x = self.text
		x = re.sub(r'[^\w ]+', "", x)
		x = ' '.join(x.split())
		return x

	def remove_special_chars(self):
		"""Remove All The Special Characters
		
		
			Example of Special Characters : &,^,@,(,),[,], %, etc.


			Return
			------------------
			CLEANED TEXT 
		"""
		x = self.text
		x = re.sub(r'[^\w ]+', "SpecialCharacter", x)
		x = ' '.join(x.split())
		return x


	def remove_html_tags_c(self):
		"""Remove HTML Tags From The Text

		Return
		---------------
		CLEANED TEXT
		
		"""
		x = self.text
		clean = re.compile('<.*?>')
		return re.sub(clean, '', x)

	def remove_html_tags(self):
		"""Remove HTML Tags From The Text

		Return
		---------------
		CLEANED TEXT
		
		"""
		x = self.text
		clean = re.compile('<.*?>')
		return re.sub(clean, 'HtmlTag', x)


	def remove_accented_chars(self):
		'''Remove Accented Character From The Text
		Example: Ó ó Ñ ñ Œ, œÆ, æ, etc.
		'''
		x = self.text
		x = unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8', 'ignore')
		return x

	def remove_emojis_c(self):
		emoj = re.compile("["
			u"\U0001F600-\U0001F64F"  # emoticons
			u"\U0001F300-\U0001F5FF"  # symbols & pictographs
			u"\U0001F680-\U0001F6FF"  # transport & map symbols
			u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
			u"\U00002500-\U00002BEF"  # chinese char
			u"\U00002702-\U000027B0"
			u"\U00002702-\U000027B0"
			u"\U000024C2-\U0001F251"
			u"\U0001f926-\U0001f937"
			u"\U00010000-\U0010ffff"
			u"\u2640-\u2642" 
			u"\u2600-\u2B55"
			u"\u200d"
			u"\u23cf"
			u"\u23e9"
			u"\u231a"
			u"\ufe0f"  # dingbats
			u"\u3030"
						"]+", re.UNICODE)
		x = self.text
		return re.sub(emoj, '', x)

	def remove_emojis(self):
		emoj = re.compile("["
			u"\U0001F600-\U0001F64F"  # emoticons
			u"\U0001F300-\U0001F5FF"  # symbols & pictographs
			u"\U0001F680-\U0001F6FF"  # transport & map symbols
			u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
			u"\U00002500-\U00002BEF"  # chinese char
			u"\U00002702-\U000027B0"
			u"\U00002702-\U000027B0"
			u"\U000024C2-\U0001F251"
			u"\U0001f926-\U0001f937"
			u"\U00010000-\U0010ffff"
			u"\u2640-\u2642" 
			u"\u2600-\u2B55"
			u"\u200d"
			u"\u23cf"
			u"\u23e9"
			u"\u231a"
			u"\ufe0f"  # dingbats
			u"\u3030"
						"]+", re.UNICODE)
		x = self.text
		return re.sub(emoj, 'EMOJI', x)

	def remove_duplicate_words(self):
		"""Remove All The Duplicate Words From The Text
		
		Return
		---------------
		Text With No Duplicates

		"""
		x = self.text
		x = " ".join(set(x.split()))
		return x

	def remove_phone_numbers(self):
		"""Replace All The Phone Numbers With PhoneNumber 
		"""
		x = self.text
		phone = re.compile(r"[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]")
		return re.sub(phone,'PhoneNumber',x)
	
	def remove_phone_numbers_c(self):
		"""Replace All The Phone Numbers With PhoneNumber 
		"""
		x = self.text
		phone = re.compile(r"[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]")
		return re.sub(phone,'',x)
		
	def stemming(self):
		"""Stem The Word To Its Base
		"""
		ps = PorterStemmer()
		text = " ".join([ps.stem(word) for word in word_tokenize(self.text)])
		return text
	
	def lemmatization(self):
		"""Lemmatize The Word To Its Base
		"""

		lemmatizer = WordNetLemmatizer()
		text = " ".join([lemmatizer.lemmatize(word) for word in word_tokenize(self.text)])
		return text

	def quick_clean(self,remove_complete,make_base):
		if remove_complete==True:
			self.text = self.remove_html_tags_c()
			self.text = self.remove_emails_c()
			self.text = self.remove_urls_c()
			self.text = self.remove_phone_numbers_c()
			self.text = self.remove_accented_chars()
			self.text = self.remove_emojis_c()
			self.text = self.remove_special_chars_c()
			self.text = self.remove_stopwords()
			self.text = self.remove_digits()
		else:
			self.text = self.remove_html_tags()
			self.text = self.remove_emails()
			self.text = self.remove_urls()
			self.text = self.remove_phone_numbers()
			self.text = self.remove_accented_chars()
			self.text = self.remove_emojis()
			self.text = self.remove_special_chars()
			self.text = self.remove_stopwords()
			self.text = self.remove_digits()
		
		if make_base=='stemming':
			self.text = self.stemming()
		elif make_base=='lemmatization':
			self.text = self.lemmatization()
		else:
			pass

		self.text = self.text.replace("  "," ")
		self.text = " ".join(self.text.split())
		self.text = self.text.lower()
		
		return self.text


#########################################################################################################
##############################------------ Dataframe Class -------------- ###############################
#########################################################################################################
class Dataframe(Cleaner):
	""" Perform Different Analysis and Tasks On Pandas DataFrame
		
		Eg: Cleaning A DataFrame Column, Performing Vectorization etc.

		parameters
		-----------------
		self.df :: DATAFRAME
		self.col :: DATAFRAME COLUMN

		usage
		-----------------
		frame = Dataframe(df=DATAFRAME,col=Column)

	"""
	def __init__(self, df=None,col=None):
		super(Dataframe, self).__init__()
		self.df = df
		self.col = col
	
	def __repr__(self):
		return f'Dataframe(df="{self.df}", col = "{self.col}")'
	
	def get_df_words_frequency_count(self):
		"""Return Word Frequency Count Inside Dataframe Column
		"""
		text = ' '.join(self.df[self.col])
		text = text.split()
		freq = pd.Series(text).value_counts()
		return freq

	def to_cv(self, max_features):
		"""Convert Text Into Vectors
		"""
		from sklearn.feature_extraction.text import CountVectorizer
		corpus = self.df[self.col]
		cv = CountVectorizer(max_features=max_features)
		x = cv.fit_transform(corpus).toarray()

		return x

	def to_tfidf(self,max_features):
		"""Convert Text Into Vectors
		
		"""
		from sklearn.feature_extraction.text import TfidfVectorizer
		corpus = self.df[self.col]
		tfidf = TfidfVectorizer(max_features=max_features)
		x = tfidf.fit_transform(corpus).toarray()

		return x
	
	def clean(self,remove_complete,make_base):
		"""Clean The Text By Removing HTML Tags, Emails, Digits, Stopwords, etc.

			```
			clean(self,remove_complete,make_base)

			Parameters
			------------------
			remove_complete = Bool(True,False)
			make_base = (stemming,lemmatization,False)
			```
		"""
		corpus = []
		for row in self.df[self.col]:
			doc = Cleaner(text=row)
			text = doc.quick_clean(remove_complete,make_base)
			corpus.append(text)

		return corpus

