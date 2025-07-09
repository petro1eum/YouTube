#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Улучшенный скрипт для получения транскрипта YouTube видео с использованием pytubefix
"""

import os
import sys
import json
import requests
from dotenv import load_dotenv
from pytubefix import YouTube

def get_video_id(url):
    """Извлечение ID видео из URL YouTube"""
    if "youtu.be" in url:
        video_id = url.split("/")[-1].split("?")[0]
    elif "youtube.com/watch" in url:
        import urllib.parse
        query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        video_id = query.get("v", [None])[0]
    else:
        raise ValueError("Неподдерживаемый формат URL")
    
    return video_id

def get_transcript_with_pytubefix(url, use_oauth=False):
    """Получение транскрипта с помощью pytubefix"""
    try:
        print("Попытка получения транскрипта с помощью pytubefix...")
        
        # Создаем объект YouTube с OAuth если нужно
        if use_oauth:
            print("Используем OAuth аутентификацию...")
            yt = YouTube(url, use_oauth=True, allow_oauth_cache=True)
        else:
            yt = YouTube(url)
        
        print(f"Название видео: {yt.title}")
        print(f"Длительность: {yt.length} секунд")
        
        # Получаем доступные субтитры
        captions = yt.captions
        print(f"Доступные субтитры: {list(captions.keys())}")
        
        if not captions:
            print("Субтитры недоступны для этого видео")
            return None, yt.title
        
        # Пробуем получить русские субтитры, затем английские
        caption = None
        for lang in ['a.ru', 'ru', 'a.en', 'en']:
            if lang in captions:
                caption = captions[lang]
                print(f"Используем субтитры на языке: {lang}")
                break
        
        if not caption:
            # Берем первые доступные субтитры
            caption_key = list(captions.keys())[0]
            caption = captions[caption_key]
            print(f"Используем субтитры: {caption_key}")
        
        # Получаем текст субтитров
        transcript_text = caption.generate_srt_captions()
        
        # Очищаем от временных меток SRT
        lines = transcript_text.split('\n')
        clean_text = []
        for line in lines:
            line = line.strip()
            # Пропускаем номера субтитров и временные метки
            if line and not line.isdigit() and '-->' not in line:
                clean_text.append(line)
        
        final_transcript = ' '.join(clean_text)
        
        return final_transcript, yt.title
        
    except Exception as e:
        print(f"Ошибка при получении транскрипта с pytubefix: {e}")
        return None, None

def analyze_with_openai(transcript, api_key, title="YouTube видео"):
    """Анализ транскрипта с помощью OpenAI API"""
    prompt = f"""
    Проанализируй следующий транскрипт видео "{title}".
    
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
        
        if "choices" in result and len(result["choices"]) > 0:
            analysis = result["choices"][0]["message"]["content"]
            return analysis
        else:
            print(f"Ошибка при анализе: {result}")
            return None
    except Exception as e:
        print(f"Ошибка при анализе транскрипта: {e}")
        return None

def save_results(video_id, video_title, transcript, analysis, output_dir="results"):
    """Сохранение результатов анализа"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Сохраняем в формате Markdown
    md_file = f"{output_dir}/{video_id}_transcript.md"
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(f"# Анализ видео YouTube: {video_title}\n")
        f.write(f"**ID видео:** {video_id}\n\n")
        f.write(analysis)
        f.write("\n\n## Транскрипт\n\n")
        f.write(transcript)
        
    print(f"Транскрипт и анализ сохранены в Markdown файл: {md_file}")
    
    # Сохраняем JSON
    json_file = f"{output_dir}/{video_id}_analysis.json"
    results = {
        "video_id": video_id,
        "video_title": video_title,
        "transcript": transcript,
        "analysis": analysis
    }
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    print(f"Результаты также сохранены в JSON: {json_file}")
    
    return md_file, json_file

def main():
    load_dotenv()
    
    if len(sys.argv) < 2:
        print("Использование:")
        print("  python youtube_improved.py <youtube_url>")
        print("  python youtube_improved.py <youtube_url> --oauth")
        print("\nПримеры:")
        print("  python youtube_improved.py https://youtu.be/VIDEO_ID")
        print("  python youtube_improved.py https://youtu.be/VIDEO_ID --oauth")
        sys.exit(1)
    
    url = sys.argv[1]
    use_oauth = "--oauth" in sys.argv
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("Ошибка: API ключ OpenAI не найден в переменной окружения OPENAI_API_KEY")
        sys.exit(1)
    
    try:
        # Получаем ID видео
        video_id = get_video_id(url)
        print(f"ID видео: {video_id}")
        
        # Получаем транскрипт
        transcript, video_title = get_transcript_with_pytubefix(url, use_oauth)
        
        if not transcript:
            print("Не удалось получить транскрипт. Завершение работы.")
            print("\nВозможные решения:")
            print("1. Попробуйте с флагом --oauth для аутентификации")
            print("2. Убедитесь, что у видео есть субтитры")
            print("3. Попробуйте другое видео")
            sys.exit(1)
        
        print(f"Транскрипт получен. Длина: {len(transcript)} символов")
        
        # Анализируем транскрипт
        print("Анализ транскрипта с помощью OpenAI...")
        analysis = analyze_with_openai(transcript, api_key, video_title)
        
        if not analysis:
            print("Не удалось выполнить анализ транскрипта.")
            sys.exit(1)
        
        # Сохраняем результаты
        transcript_file, json_file = save_results(video_id, video_title, transcript, analysis)
        
        print("Готово!")
        print(f"Транскрипт и анализ: {transcript_file}")
        print(f"JSON-файл с результатами: {json_file}")
        
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 