""""

PROYECTO 3  Aplicaciones de IA

Procesamiento de Lenguaje Natural (NLP)


Nombre: Armando Rodriguez


email: armandospxp@gmail.com


"""

###Importacion de librerias

# Librerias de string para python

import string
import re

# random
import random

# Libreria nltk

import nltk
from ntlk.corpus import movie_reviews, stop_words
from nltk.stem import WordNetLemmatizer

# Vectorizacion y split

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Entrenamiento y metricas

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score,
                             classification_report, ConfusionMatrixDisplay)

def iniciacion_nltk():
    """Metodo para inicializar la libreria nltk"""
    nltk.download('movie_reviews')
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    print(f"Nº de documentos: {len(movie_reviews.fileids())}")
    print(f"Etiquetas disponibles: {movie_reviews.categories()}")
    print("Ejemplo de review breve:\n", movie_reviews.raw(movie_reviews.fileids()[0])[:400], "...")

def division_texto_etiquetas():
    """Metodo para dividir las etiquetas y el texto"""
