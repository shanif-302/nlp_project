import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

text = "I am Learning Machine Learning!!!"

text = text.lower()
text = text.translate(str.maketrans('', '', string.punctuation))

words = word_tokenize(text)

stop_words = set(stopwords.words('english'))
words = [word for word in words if word not in stop_words]

ps = PorterStemmer()
words = [ps.stem(word) for word in words]

print(words)
