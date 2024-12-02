import unicodedata

from nltk.stem.snowball import EnglishStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from functools import partial


fff = partial(unicodedata.normalize, "NFC")
a = fff("dfdsf")

aaa = 5
