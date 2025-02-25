import csv
import sys
import requests
import asyncio
from googletrans import Translator

# URL API Ollama (deepseek)
OLLAMA_API_URL = "http://localhost:11434/api/chat"

# Инициализация переводчика
translator = Translator()

def parse_google_form_csv(file_path):
    """
    Парсит CSV файл, полученный из Google Формы.
    """
    with open(file_path, mode='r', encoding='utf-8-sig') as file:
        reader = csv.reader(file)
        questions = next(reader)
        answers = {}
        for row in reader:
            if row:
                identifier = row[0]
                answer_data = row[1:]
                answers[identifier] = answer_data
    return questions, answers

async def translate_word(text, src='ru', dest='en'):
    """
    Переводит текст с языка src на язык dest асинхронно.
    По умолчанию: с русского (ru) на английский (en).
    """
    translation = await translator.translate(text, src=src, dest=dest)
    return translation.text

async def create_messages(answer, question):
    """
    Создает список сообщений для отправки в API deepseek.
    """
    translated_question = await translate_word(question, src='ru', dest='en')
    translated_answer = await translate_word(answer, src='ru', dest='en')
    prompt = (f"Is the answer '{translated_answer}' to the question '{translated_question}'? "
              "Rate the answer on a scale of 1 to 10 and explain your choice.")
    messages = [{'role': 'user', 'content': prompt}]
    return messages

def chat_with_ollama(model, messages):
    """
    Отправляет запрос к API Ollama (deepseek) и возвращает ответ ассистента.
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        if ('message' in result and
            'role' in result['message'] and
            result['message']['role'] == 'assistant'):
            return result['message']['content']
        else:
            return "Не удалось получить ответ от ассистента."
    except requests.exceptions.RequestException as e:
        return f"Error communicating with Ollama: {e}"

async def main():
    # Если путь к CSV файлу передан через аргументы командной строки, используем его
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = 'new_form.csv'

    # Читаем CSV
    questions, answers_dict = parse_google_form_csv(file_path)
    print("Заголовки (questions):")
    print(questions)
    print("=" * 50)

    question_texts = questions[1:]
    model = "deepseek-r1:1.5b"

    # Проходим по каждому набору ответов
    for identifier, answer_list in answers_dict.items():
        print(f"Оценка ответа для записи: {identifier}")
        
        # Для каждого вопроса и соответствующего ответа
        for idx, answer in enumerate(answer_list):
            question = question_texts[idx]
            messages = await create_messages(answer, question)
            evaluation = chat_with_ollama(model, messages)
            print(f"Вопрос: {question}")
            print(f"Ответ: {answer}")
            print("Оценка от deepseek:")
            print(evaluation)
            print("-" * 40)

# Запускаем асинхронный код
if __name__ == "__main__":
    asyncio.run(main())