import logging
import csv
from yargy import Parser, rule, or_, and_
from yargy.pipelines import morph_pipeline
from yargy.predicates import gram, eq, custom, in_
from dataclasses import dataclass
from typing import Optional, List, Set
import concurrent.futures

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Entry:
    rubric: str
    headline: str
    names: Set[str]
    birth_dates: Set[str]
    birth_places: Set[str]

def extract_names(content: str) -> Set[str]:
    """
    Извлекает имена из текста.

    Поддерживает различные форматы имен, включая:
    - Имя и фамилия (например, Иванов Пётр)
    - Имя, отчество и фамилия (например, Иванов Пётр Сергеевич)
    - Только имя или только фамилия
    - Имена с инициалами (например, П. С. Иванов)

    Параметры:
    content (str): Текст, из которого необходимо извлечь имена.

    Возвращает:
    Set[str]: Множество уникальных имен, найденных в тексте.
    """
    # Определяем правила для имени и фамилии
    FULL_NAME = rule(
        gram('Name'),    # Имя
        gram('Patr'),    # Отчество (опционально)
        gram('Surn')     # Фамилия
    )

    INITIAL_NAME = rule(
        gram('Name'),    # Имя
        eq('.'),         # Точка после имени
        gram('Surn')     # Фамилия
    )

    # Комбинируем правила
    NAME_RULE = or_(
        FULL_NAME,
        INITIAL_NAME,
        rule(gram('Name')),  # Только имя
        rule(gram('Surn'))   # Только фамилия
    )

    name_parser = Parser(NAME_RULE)
    names = set()

    for match in name_parser.findall(content):
        name = " ".join(token.value for token in match.tokens)
        names.add(name)

    logger.info(f"Извлеченные имена: {names}")
    return names

def extract_locations(content: str) -> Set[str]:
    """
    Извлекает места рождения из текста.

    Поддерживает различные форматы выражений, связанных с местом рождения, включая:
    - "родился в Москве"
    - "родом из Казани"
    - "проживает в Санкт-Петербурге"
    - "родной город: Владивосток"
    - "место рождения – Новосибирск"
    - "родился и вырос в Екатеринбурге"

    Параметры:
    content (str): Текст, из которого необходимо извлечь места.

    Возвращает:
    Set[str]: Множество уникальных мест, найденных в тексте.
    """
    LOCATION_RULE = or_(
        # Простые конструкции
        rule(eq('родился'), eq('в'), gram('Geox')),
        rule(eq('родилась'), eq('в'), gram('Geox')),
        rule(eq('родом'), eq('из'), gram('Geox')),
        rule(eq('проживает'), eq('в'), gram('Geox')),
        
        # Сложные конструкции
        rule(eq('дом'), eq('является'), eq('местом'), eq('рождения'), gram('Geox')),
        rule(eq('родной'), eq('город'), eq(':'), gram('Geox')),
        rule(eq('место'), eq('рождения'), eq('-'), gram('Geox')),
        rule(eq('родился'), eq('и'), eq('вырос'), eq('в'), gram('Geox'))
    )

    location_parser = Parser(LOCATION_RULE)
    locations = set()

    for match in location_parser.findall(content):
        # Извлекаем только географическую часть
        tokens = match.tokens
        location = []
        for token in tokens:
            if token.tag == 'Geox':
                location.append(token.value)
        if location:
            locations.add(" ".join(location))

    logger.info(f"Извлеченные места: {locations}")
    return locations

def extract_dates(content: str) -> Set[str]:
    """
    Извлекает даты рождения из текста.

    Поддерживает различные форматы дат, включая:
    - "родился 12 мая 1990 года"
    - "родилась 05.06.1985"
    - "дата рождения: 1992-07-23"
    - "родился 23 июля"
    - "родился в 1990 году"
    - "родился 12/05/1990"

    Параметры:
    content (str): Текст, из которого необходимо извлечь даты.

    Возвращает:
    Set[str]: Множество уникальных дат, найденных в тексте.
    """
    def is_valid_year(value: str) -> bool:
        """Проверяет, является ли значение допустимым годом."""
        return value.isdigit() and 1000 <= int(value) <= 9999

    YEAR = custom(is_valid_year)

    # Определяем месяцы
    MONTHS = morph_pipeline([
        'января', 'февраля', 'марта', 'апреля', 'мая', 'июня',
        'июля', 'августа', 'сентября', 'октября', 'ноября', 'декабря'
    ])

    # Определяем префиксы, связанные с рождением
    BIRTH_PREFIX = morph_pipeline([
        'родился', 'родилась', 'дата рождения', 'дата'
    ])

    # Определяем различные форматы дат
    DATE_RULE = or_(
        # Формат: родился в 12 мая 1990 года
        rule(
            BIRTH_PREFIX,
            gram('NUM'),          # День
            MONTHS,
            YEAR.optional(),
            eq('года').optional()
        ),
        # Формат: родился 12 мая 1990 года
        rule(
            gram('NUM'),
            MONTHS,
            YEAR.optional(),
            eq('года').optional()
        ),
        # Формат: дата рождения: 1990-05-12
        rule(
            eq(':'),
            YEAR,
            eq('-'),
            gram('NUM'),
            eq('-'),
            gram('NUM')
        ),
        # Формат: родился 12/05/1990
        rule(
            gram('NUM'),
            eq('/'),
            gram('NUM'),
            eq('/'),
            YEAR
        ),
        # Формат: родился 12.05.1990
        rule(
            gram('NUM'),
            eq('.'),
            gram('NUM'),
            eq('.'),
            YEAR
        )
    )

    date_parser = Parser(DATE_RULE)
    dates = set()

    for match in date_parser.findall(content):
        tokens = match.tokens
        date_parts = []
        for token in tokens:
            if token.tag in ['NUM'] or token.value.lower() in [
                'января', 'февраля', 'марта', 'апреля', 'мая', 'июня',
                'июля', 'августа', 'сентября', 'октября', 'ноября', 'декабря'
            ]:
                date_parts.append(token.value)
            elif token.value in ['-', '/','.']:
                date_parts.append(token.value)
        date = " ".join(date_parts)
        dates.add(date)

    logger.info(f"Извлеченные даты: {dates}")
    return dates

def process_line(line: str) -> Optional[List[Entry]]:
    """Обрабатывает строку и извлекает данные."""
    parts = line.strip().split('\t')
    if len(parts) < 3:
        logger.warning("Недостаточно данных в строке, пропускаем.")
        return None

    rubric, headline, text = parts
    content = f"{headline} {text}"

    names = extract_names(content)
    if not names:
        logger.warning("Имена не найдены, пропускаем строку.")
        return None

    locations = extract_locations(content)
    dates = extract_dates(content)

    if not locations and not dates:
        logger.warning("Не найдены места или даты рождения, пропускаем строку.")
        return None

    entries = []
    for name in names:
        entry = Entry(
            rubric=rubric,
            headline=headline,
            names={name},
            birth_dates=dates,
            birth_places=locations
        )
        entries.append(entry)
        logger.info(f"Создана запись: {entry}")

    return entries

def process_news_file(file_path: str) -> List[Entry]:
    """Обрабатывает файл новостей и извлекает данные."""
    entries = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = list(executor.map(process_line, f))
                for result in results:
                    if result:
                        entries.extend(result)

    except FileNotFoundError:
        logger.error(f"Ошибка: Файл '{file_path}' не найден.")
    except Exception as e:
        logger.error(f"Произошла ошибка: {e}")

    logger.info(f"Обработано записей: {len(entries)}")
    return entries

def save_to_csv(entries: List[Entry], output_file: str):
    """Сохраняет извлеченные данные в CSV файл."""
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['rubric', 'headline', 'name', 'birth_date', 'birth_place']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for entry in entries:
            writer.writerow({
                'rubric': entry.rubric,
                'headline': entry.headline,
                'name': ", ".join(entry.names),
                'birth_date': ", ".join(entry.birth_dates) if entry.birth_dates else 'Не найдено',
                'birth_place': ", ".join(entry.birth_places) if entry.birth_places else 'Не найдено'
            })
    logger.info(f"Данные сохранены в файл: {output_file}")

if __name__ == "__main__":
    input_file_path = './data/news.txt'
    output_file_path = './data/extracted_data.csv'
    entries = process_news_file(input_file_path)
    save_to_csv(entries, output_file_path)
