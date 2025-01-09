# Импорт необходимых библиотек
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Iterator
import gensim
from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import logging

# Настройка логирования (опционально)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Класс для представления текста с меткой
class Text:
    def __init__(self, label: str, text: str):
        self.label = label
        self.text = text

# Функция для чтения текстов из файла
def read_texts(fn: str) -> Iterator[Text]:
    with open(fn, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            parts = line.strip().split("\t", 1)  # Разбиваем только по первой табуляции
            if len(parts) != 2:
                logging.warning(f"Строка {line_number}: некорректный формат")
                continue
            try:
                yield Text(*parts)
            except Exception as e:
                logging.error(f"Строка {line_number}: ошибка при создании объекта Text - {e}")

# Предобработка текста: токенизация, удаление стоп-слов, лемматизация с использованием spaCy
def preprocess_spacy(text: str, stop_words: set, nlp) -> list:
    """
    Предобработка текста с использованием spaCy:
    - Приведение к нижнему регистру
    - Токенизация
    - Удаление пунктуации и стоп-слов
    - Лемматизация
    """
    doc = nlp(text.lower())
    tokens = [
        token.lemma_
        for token in doc
        if token.is_alpha and token.lemma_ not in stop_words
    ]
    return tokens

def main():
    # Шаг 1: Чтение данных
    texts = list(read_texts('./data/news.txt'))
    
    # Инициализация списков для меток и текстов
    X = []
    y = []
    
    for item in texts:
        label, text = item.label, item.text
        X.append(text)
        y.append(label)
    
    print(f"Общее количество текстов: {len(texts)}")
    
    # Шаг 2: Разделение данных на обучающую и тестовую выборки
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True
    )
    
    print(f"Обучающая выборка: {len(X_train_texts)}")
    print(f"Тестовая выборка: {len(X_test_texts)}")
    
    # Шаг 3: Предобработка текстов
    nltk.download('stopwords')
    nltk.download('punkt')
    
    stop_words = set(stopwords.words('russian'))
    nlp = spacy.load("ru_core_news_sm")
    
    print("Предобработка текстов...")
    X_train_tokens = [preprocess_spacy(text, stop_words, nlp) for text in X_train_texts]
    X_test_tokens = [preprocess_spacy(text, stop_words, nlp) for text in X_test_texts]
    
    # Шаг 4: Обучение модели Word2Vec
    print("Обучение модели Word2Vec...")
    vector_size = 100
    window = 5
    min_count = 2
    workers = 4
    
    model = Word2Vec(
        sentences=X_train_tokens,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=10  # Количество эпох обучения
    )
    
    # Сохранение модели (опционально)
    # model.save("word2vec.model")
    
    # Шаг 5: Представление документов через усреднение векторов слов
    def document_vector(doc, model):
        # Исключение слов, отсутствующих в модели
        doc = [word for word in doc if word in model.wv.key_to_index]
        if len(doc) == 0:
            return np.zeros(model.vector_size)
        return np.mean(model.wv[doc], axis=0)
    
    print("Создание векторов документов (усреднение)...")
    X_train_vec = np.array([document_vector(doc, model) for doc in X_train_tokens])
    X_test_vec = np.array([document_vector(doc, model) for doc in X_test_tokens])
    
    # Шаг 6: Классификация текстов с использованием SVM
    print("Обучение модели SVM (усреднение)...")
    svm = SVC(kernel='linear', random_state=42)
    svm.fit(X_train_vec, y_train)
    
    print("Оценка модели SVM (усреднение)...")
    y_pred = svm.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Точность модели SVM: {accuracy:.2f}")
    print("Отчет о классификации:")
    print(classification_report(y_test, y_pred))
    
    # Шаг 7: Альтернативный способ представления документов (взвешенное усреднение с TF-IDF)
    print("Создание взвешенных векторов документов с использованием TF-IDF...")
    
    # Объединение токенов обратно в строки для TfidfVectorizer
    X_train_processed = [' '.join(doc) for doc in X_train_tokens]
    X_test_processed = [' '.join(doc) for doc in X_test_tokens]
    
    # Создание TF-IDF векторизатора
    tfidf = TfidfVectorizer()
    tfidf.fit(X_train_processed)
    
    # Получение словаря TF-IDF
    tfidf_vocab = tfidf.vocabulary_
    idf = tfidf.idf_
    
    # Создание словаря IDF для быстрого доступа
    idf_dict = dict(zip(tfidf.get_feature_names_out(), idf))
    
    def weighted_document_vector(doc, model, idf_dict):
        vectors = []
        weights = []
        for word in doc:
            if word in model.wv.key_to_index and word in idf_dict:
                vectors.append(model.wv[word])
                weights.append(idf_dict[word])
        if not vectors:
            return np.zeros(model.vector_size)
        vectors = np.array(vectors)
        weights = np.array(weights)
        weighted_avg = np.average(vectors, axis=0, weights=weights)
        return weighted_avg
    
    print("Создание взвешенных векторов документов...")
    X_train_weighted = np.array([weighted_document_vector(doc, model, idf_dict) for doc in X_train_tokens])
    X_test_weighted = np.array([weighted_document_vector(doc, model, idf_dict) for doc in X_test_tokens])
    
    # Шаг 8: Классификация текстов с использованием SVM на взвешенных векторах
    print("Обучение модели SVM (взвешенное усреднение)...")
    svm_weighted = SVC(kernel='linear', random_state=42)
    svm_weighted.fit(X_train_weighted, y_train)
    
    print("Оценка модели SVM (взвешенное усреднение)...")
    y_pred_weighted = svm_weighted.predict(X_test_weighted)
    accuracy_weighted = accuracy_score(y_test, y_pred_weighted)
    print(f"Точность модели SVM с TF-IDF взвешиванием: {accuracy_weighted:.2f}")
    print("Отчет о классификации (взвешенное усреднение):")
    print(classification_report(y_test, y_pred_weighted))
    
if __name__ == "__main__":
    main()
