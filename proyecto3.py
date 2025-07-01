""""

PROYECTO 3  Aplicaciones de IA

Procesamiento de Lenguaje Natural (NLP)


Nombre: Armando Rodriguez


email: armandospxp@gmail.com


"""

# Importacion de librerias

# Librerias de string para python

import string
import re

# tipos de datos y tuplas
from typing import List, Tuple

# random
import random

# Libreria nltk

import nltk
from nltk.corpus import movie_reviews, stopwords
from nltk.stem import WordNetLemmatizer

# Vectorizacion y split

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Entrenamiento y metricas

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score,
                             classification_report, ConfusionMatrixDisplay)

# Visualización
import matplotlib.pyplot as plt


def descarga_nltk_corpora() -> None:
    """Descarga los recursos NLTK necesarios (solo la primera vez)."""
    for pkg in ("movie_reviews", "punkt", "stopwords", "wordnet", "punkt_tab"):
        nltk.download(pkg, quiet=True)


def iniciador_nlp_tools() -> Tuple[WordNetLemmatizer, set[str]]:
    """Inicializa lematizador y conjunto de stop-words."""
    return WordNetLemmatizer(), set(stopwords.words("english"))


def carga_dataset() -> Tuple[List[str], List[str]]:
    """Devuelve listas de textos y etiquetas ('pos' / 'neg')."""
    fids = movie_reviews.fileids()
    texts = [" ".join(movie_reviews.words(fid)) for fid in fids]
    labels = [movie_reviews.categories(fid)[0] for fid in fids]
    return texts, labels


# ───────────────────────────── PREPROCESADO ──────────────────────────────
def clean_text(text: str, lemmatizer: WordNetLemmatizer, stop_set: set[str]) -> str:
    """Lower-case, quita HTML, puntuación, stop-words y lematiza."""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok not in stop_set and len(tok) > 2
    ]
    return " ".join(tokens)


def preprocesado_texto(
    texts: List[str], lemmatizer: WordNetLemmatizer, stop_set: set[str]
) -> List[str]:
    """Aplica limpieza a cada string del corpus."""
    return [clean_text(t, lemmatizer, stop_set) for t in texts]


def vectorizado_texto(
    texts: List[str], max_features: int = 7_500, ngram_range: Tuple[int, int] = (1, 2)
):
    """Crea TF-IDF y devuelve matriz X + el vectorizador entrenado."""
    vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X = vec.fit_transform(texts)
    return X, vec


def entrenamiento_clasificador(X_train, y_train, max_iter: int = 3_000):
    clf = LogisticRegression(max_iter=max_iter, n_jobs=-1)
    clf.fit(X_train, y_train)
    return clf


def print_metrics(y_true, y_pred) -> None:
    print("\n=== Métricas en test ===")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.3f}")
    print(f"Precision: {precision_score(y_true, y_pred, pos_label='pos'):.3f}")
    print(f"Recall   : {recall_score(y_true, y_pred, pos_label='pos'):.3f}")
    print(f"F1-score : {f1_score(y_true, y_pred, pos_label='pos'):.3f}\n")
    print(classification_report(y_true, y_pred, digits=3))


def plot_confusion_matrix(y_true, y_pred) -> None:
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, cmap="Blues", xticks_rotation=45
    )
    plt.title("Matriz de confusión – Reviews de películas")
    plt.tight_layout()
    plt.show()


def predecir_sentencia(
    sentence: str,
    vectorizer: TfidfVectorizer,
    clf: LogisticRegression,
    lemmatizer: WordNetLemmatizer,
    stop_set: set[str],
) -> str:
    """Devuelve 'pos' / 'neg' para una frase nueva."""
    clean = clean_text(sentence, lemmatizer, stop_set)
    return clf.predict(vectorizer.transform([clean]))[0]


if __name__ == "__main__":
    """Corrida del script"""
    descarga_nltk_corpora()
    lemmatizer, stop_set = iniciador_nlp_tools()

    texts, labels = carga_dataset()
    print(f"Nº de documentos: {len(texts)}")
    print(f"Etiquetas disponibles: {set(labels)}")
    print("Ejemplo de review breve:\n", texts[0][:400], "...")

    # preprocesado_texto
    print("\n=== Preprocesado de texto ===")
    texts_clean = preprocesado_texto(texts, lemmatizer, stop_set)

    # vectorizado_texto
    X, vectorizer = vectorizado_texto(texts_clean)

    # split de datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.25, stratify=labels, random_state=42
    )

    # entrenamiento_clasificador
    print("\n=== Entrenamiento del clasificador ===")
    clf = entrenamiento_clasificador(X_train, y_train)

    # evaluación del modelo
    print("\n=== Evaluación del modelo ===")
    y_pred = clf.predict(X_test)
    print_metrics(y_test, y_pred)
    plot_confusion_matrix(y_test, y_pred)

    # predicción de una frase de prueba
    print("\n=== Predicción de una frase de prueba ===")
    sample = "The movie was beautifully shot and emotionally powerful."
    pred = predecir_sentencia(sample, vectorizer, clf, lemmatizer, stop_set)
    print(f"\nFrase de prueba: «{sample}»\n→ Sentimiento predicho: '{pred}'")
