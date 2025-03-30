#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Скрипт для получения транскрипта YouTube видео, его анализа и сохранения
"""

import os
import sys
import json
import requests
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, _errors

def get_video_id(url):
    """Извлечение ID видео из URL YouTube"""
    # Проверка формата URL
    if "youtu.be" in url:
        # Формат https://youtu.be/VIDEO_ID
        video_id = url.split("/")[-1].split("?")[0]
    elif "youtube.com/watch" in url:
        # Формат https://www.youtube.com/watch?v=VIDEO_ID
        import urllib.parse
        query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        video_id = query.get("v", [None])[0]
    else:
        raise ValueError("Неподдерживаемый формат URL")
    
    return video_id

def get_youtube_transcript(video_id, languages=['ru', 'en']):
    """Получение транскрипта с YouTube, если доступен"""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
        
        # Объединяем все части транскрипта в один текст
        full_transcript = " ".join([item['text'] for item in transcript_list])
        
        return full_transcript
    except _errors.TranscriptsDisabled:
        print("Транскрипты отключены для этого видео.")
        return None
    except _errors.NoTranscriptAvailable:
        print("Транскрипты недоступны для этого видео.")
        return None
    except Exception as e:
        print(f"Ошибка при получении транскрипта: {e}")
        return None

def analyze_with_openai(transcript, api_key):
    """Анализ транскрипта с помощью OpenAI API"""
    prompt = f"""
    Проанализируй следующий транскрипт видео.
    
    Транскрипт:
    {transcript}
    
    Создай следующее:
    1. Краткое содержание (3-5 предложений)
    2. Основные тезисы и ключевые мысли (8-10 пунктов в виде маркированного списка). Выдели самые важные идеи, концепции и примеры из транскрипта. Рассмотри различные аспекты обсуждаемой темы.
    """
    
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "Ты - аналитик видеоконтента. Твоя задача - выделять ключевые идеи и создавать подробный анализ содержания."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 2000
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions", 
            headers=headers, 
            json=payload
        )
        result = response.json()
        
        # Получаем ответ от API
        if "choices" in result and len(result["choices"]) > 0:
            analysis = result["choices"][0]["message"]["content"]
            return analysis
        else:
            print(f"Ошибка при анализе: {result}")
            return None
    except Exception as e:
        print(f"Ошибка при анализе транскрипта: {e}")
        return None

def save_results(video_id, transcript, analysis, output_dir="results"):
    """Сохранение результатов анализа"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Сохраняем в формате Markdown
    md_file = f"{output_dir}/{video_id}_transcript.md"
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(f"# Анализ видео YouTube (ID: {video_id})\n\n")
        f.write(analysis)
        f.write("\n\n## Транскрипт\n\n")
        f.write(transcript)
        
    print(f"Транскрипт и анализ сохранены в Markdown файл: {md_file}")
    
    # Сохраняем JSON
    json_file = f"{output_dir}/{video_id}_analysis.json"
    results = {
        "video_id": video_id,
        "transcript": transcript,
        "analysis": analysis
    }
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    print(f"Результаты также сохранены в JSON: {json_file}")
    
    return md_file, json_file

def main():
    # Загружаем переменные окружения из .env файла
    load_dotenv()
    
    if len(sys.argv) < 2:
        print("Использование: python get_transcript.py <youtube_url>")
        sys.exit(1)
    
    url = sys.argv[1]
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("Ошибка: API ключ OpenAI не найден в переменной окружения OPENAI_API_KEY")
        sys.exit(1)
    
    try:
        # Получаем ID видео
        video_id = get_video_id(url)
        print(f"ID видео: {video_id}")
        
        # Получаем транскрипт
        transcript = get_youtube_transcript(video_id)
        
        if not transcript:
            print("Не удалось получить транскрипт. Завершение работы.")
            sys.exit(1)
        
        print(f"Транскрипт получен. Длина: {len(transcript)} символов")
        
        # Анализируем транскрипт
        print("Анализ транскрипта с помощью OpenAI...")
        analysis = analyze_with_openai(transcript, api_key)
        
        if not analysis:
            print("Не удалось выполнить анализ транскрипта.")
            sys.exit(1)
        
        # Сохраняем результаты
        transcript_file, json_file = save_results(video_id, transcript, analysis)
        
        print("Готово!")
        print(f"Транскрипт и анализ: {transcript_file}")
        print(f"JSON-файл с результатами: {json_file}")
        
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 