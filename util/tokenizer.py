"""
@author : Hyunwoong
@when : 2019-10-29
@homepage : https://github.com/gusdnd852
"""
import nltk
from nltk.tokenize import word_tokenize

class Tokenizer:
    def __init__(self):
        self.spacy_de = 'german'
        self.spacy_en = 'english'

    def tokenize_de(self, text):
        """
        Tokenizes German txt from a string into a list of strings
        """
        return word_tokenize(text, language=self.spacy_de)

    def tokenize_en(self, text):
        """
        Tokenizes English text from a string into a list of strings
        """
        return word_tokenize(text, language=self.spacy_en)