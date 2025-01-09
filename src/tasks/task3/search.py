import faiss
import pickle
from sentence_transformers import SentenceTransformer
import click
from tqdm import tqdm

@click.command()
@click.option('--index_file', default='faiss_index.bin', help='Файл FAISS индекса.')
@click.option('--mapping_file', default='id_mapping.pkl', help='Файл соответствия ID-документам.')
@click.option('--model_name', default='all-MiniLM-L6-v2', help='Название модели sentence-transformers.')
@click.option('--top_k', default=5, help='Количество возвращаемых результатов.')
def search_query(index_file, mapping_file, model_name, top_k):
    """
    Выполняет поиск по заданному запросу и возвращает наиболее релевантные документы.
    """
    # Загрузка FAISS индекса
    print(f"Загрузка FAISS индекса из '{index_file}'...")
    index = faiss.read_index(index_file)

    # Загрузка соответствия ID-документам
    print(f"Загрузка соответствия ID-документам из '{mapping_file}'...")
    with open(mapping_file, 'rb') as f:
        id_mapping = pickle.load(f)

    # Инициализация модели
    print(f"Загрузка модели '{model_name}'...")
    model = SentenceTransformer(model_name)

    # Приём поискового запроса
    query = click.prompt("Введите поисковый запрос", type=str)

    # Генерация эмбеддинга для запроса
    print("Генерация эмбеддинга для запроса...")
    query_embedding = model.encode([query], convert_to_numpy=True)

    # Поиск в FAISS индексе
    print(f"Поиск {top_k} наиболее похожих документов...")
    distances, indices = index.search(query_embedding, top_k)

    # Вывод результатов
    print("\nРезультаты поиска:")
    for rank, (distance, idx) in enumerate(zip(distances[0], indices[0]), start=1):
        filename = id_mapping.get(idx, "Unknown")
        print(f"{rank}. {filename} (Расстояние: {distance:.4f})")

if __name__ == '__main__':
    search_query()
