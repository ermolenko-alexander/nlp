import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import click
from tqdm import tqdm
from datasets import load_dataset

@click.command()
@click.option('--index_file', default='faiss_index.bin', help='Файл для сохранения FAISS индекса.')
@click.option('--mapping_file', default='id_mapping.pkl', help='Файл для сохранения соответствия ID и документов.')
@click.option('--model_name', default='all-MiniLM-L6-v2', help='Название модели sentence-transformers.')
def index_documents(index_file, mapping_file, model_name):
    """
    Индексирует документы и сохраняет FAISS индекс и соответствие ID-документам.
    """
    # Инициализация модели
    print(f"Загрузка модели '{model_name}'...")
    model = SentenceTransformer(model_name)

    # Загрузка данных из wikitext
    print("Загрузка данных из wikitext...")
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    documents = dataset['train']['text']  # Получаем текстовые данные
    filenames = [f"doc_{i}" for i in range(len(documents))]  # Генерируем имена файлов

    print(f"Количество документов: {len(documents)}")

    # Генерация векторов
    print("Генерация векторных представлений документов...")
    embeddings = model.encode(documents, show_progress_bar=True, convert_to_numpy=True)

    # Создание FAISS индекса
    dimension = embeddings.shape[1]
    print(f"Создание FAISS индекса с размерностью {dimension}...")
    index = faiss.IndexFlatL2(dimension)  # Используем простой индекс без обучения
    index.add(embeddings)
    print(f"Количество векторов в индексе: {index.ntotal}")

    # Сохранение FAISS индекса
    faiss.write_index(index, index_file)
    print(f"FAISS индекс сохранён в '{index_file}'.")

    # Создание и сохранение соответствия ID-документам
    id_mapping = {i: filename for i, filename in enumerate(filenames)}
    with open(mapping_file, 'wb') as f:
        pickle.dump(id_mapping, f)
    print(f"Соответствие ID-документам сохранено в '{mapping_file}'.")

if __name__ == '__main__':
    index_documents()
