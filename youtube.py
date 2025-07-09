# -*- coding: utf-8 -*-
"""
Скрипт для анализа YouTube видео и транскриптов с использованием LLM
"""

# Стандартные импорты
import os
import argparse
import base64
from pytubefix import YouTube  # Используем pytubefix вместо pytube
import whisper
import json
import requests
import cv2
import numpy as np
import tempfile
import time
import pytesseract  # Для OCR-распознавания текста
from youtube_transcript_api import YouTubeTranscriptApi, _errors
from dotenv import load_dotenv
import re
from PIL import Image, ImageEnhance, ImageFilter
from openai import OpenAI

def get_video_id(url):
    """Извлечение ID видео из URL YouTube"""
    try:
        # Проверка формата URL
        if "youtu.be" in url:
            # Формат https://youtu.be/VIDEO_ID
            video_id = url.split("/")[-1].split("?")[0]
        elif "youtube.com/watch" in url:
            # Формат https://www.youtube.com/watch?v=VIDEO_ID
            import urllib.parse
            query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
            video_id = query.get("v", [None])[0]
        elif "youtube.com/shorts" in url:
            # Формат https://www.youtube.com/shorts/VIDEO_ID
            video_id = url.split("/shorts/")[1].split("?")[0]
        else:
            # Пробуем использовать pytubefix, если формат URL не распознан
            yt = YouTube(url)
            video_id = yt.video_id
            
        if not video_id:
            raise ValueError("Не удалось извлечь ID видео из URL")
            
        return video_id
    except Exception as e:
        print(f"Ошибка при извлечении ID видео: {e}")
        raise

def download_audio(url, output_dir="temp"):
    """Загрузка аудио из YouTube видео в самом низком качестве"""
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Сначала пробуем без po_token
        try:
            yt = YouTube(url)
            print(f"Загрузка аудио из видео: {yt.title}")
        except Exception as e:
            print(f"Попытка загрузки без po_token не удалась: {e}")
            print("Пробуем с use_po_token=True...")
            yt = YouTube(url, use_po_token=True)
            print(f"Загрузка аудио из видео: {yt.title}")
        
        # Выбираем аудио поток с самым низким качеством для экономии трафика и времени
        audio_stream = yt.streams.filter(only_audio=True).order_by('abr').first()
        
        print(f"Выбран аудио поток с битрейтом: {audio_stream.abr if hasattr(audio_stream, 'abr') else 'Unknown'}")
        
        # Загружаем аудио
        audio_file = audio_stream.download(output_path=output_dir)
        
        # Переименовываем файл для удобства
        base, ext = os.path.splitext(audio_file)
        new_file = f"{output_dir}/{yt.video_id}.mp3"
        os.rename(audio_file, new_file)
        
        print(f"Аудио успешно загружено: {new_file}")
        return new_file, yt.title
        
    except Exception as e:
        print(f"Ошибка при загрузке аудио: {e}")
        return None, None

def download_video(url, output_dir="temp", quality="highest"):
    """Загрузка видео из YouTube в высоком качестве для анализа кадров"""
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Сначала пробуем без po_token
        try:
            yt = YouTube(url)
            print(f"Загрузка видео для анализа кадров: {yt.title}")
        except Exception as e:
            print(f"Попытка загрузки без po_token не удалась: {e}")
            print("Пробуем с use_po_token=True...")
            yt = YouTube(url, use_po_token=True)
            print(f"Загрузка видео для анализа кадров: {yt.title}")
        
        # Выбираем видео поток с наилучшим качеством для лучшего распознавания кода
        if quality == "lowest":
            video_stream = yt.streams.filter(progressive=True).order_by('resolution').first()
        else:
            video_stream = yt.streams.filter(progressive=True).get_highest_resolution()
        
        print(f"Выбран видео поток с разрешением: {video_stream.resolution}")
        
        # Загружаем видео
        video_file = video_stream.download(output_path=output_dir)
        
        # Переименовываем файл для удобства
        base, ext = os.path.splitext(video_file)
        new_file = f"{output_dir}/{yt.video_id}{ext}"
        if video_file != new_file:
            os.rename(video_file, new_file)
        
        print(f"Видео успешно загружено: {new_file}")
        return new_file
        
    except Exception as e:
        print(f"Ошибка при загрузке видео: {e}")
        return None

def get_youtube_transcript(video_id):
    """Получение транскрипта с YouTube, если доступен"""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['ru', 'en'])
        
        # Возвращаем как список сегментов с временными метками
        return transcript_list
    except _errors.TranscriptsDisabled:
        print("Транскрипты отключены для этого видео.")
        return None
    except _errors.NoTranscriptFound:
        print("Транскрипты недоступны для этого видео.")
        return None
    except Exception as e:
        print(f"Ошибка при получении транскрипта: {e}")
        return None

def create_transcript_with_whisper(audio_file, model_size="tiny"):
    """Создание транскрипта с помощью Whisper, использует tiny модель по умолчанию для быстрой транскрипции"""
    try:
        print(f"Загрузка модели Whisper ({model_size})...")
        model = whisper.load_model(model_size)
        
        print("Транскрибирование аудио...")
        result = model.transcribe(audio_file)
        
        # Возвращаем сегменты с временными метками в формате, аналогичном YouTube API
        segments = []
        for segment in result["segments"]:
            segments.append({
                "text": segment["text"],
                "start": segment["start"],
                "duration": segment["end"] - segment["start"]
            })
        
        return segments
    except Exception as e:
        print(f"Ошибка при создании транскрипта с Whisper: {e}")
        return None

def enhance_image_for_ocr(image_path, output_path=None):
    """Улучшает изображение для более точного OCR-распознавания кода"""
    try:
        # Если путь для вывода не указан, используем временный файл
        if not output_path:
            output_path = f"{os.path.splitext(image_path)[0]}_enhanced.png"
        
        # Открываем изображение с помощью PIL
        img = Image.open(image_path)
        
        # Преобразуем в оттенки серого
        img = img.convert('L')
        
        # Увеличиваем контраст
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)
        
        # Увеличиваем резкость
        img = img.filter(ImageFilter.SHARPEN)
        
        # Увеличиваем размер изображения для лучшего распознавания
        width, height = img.size
        img = img.resize((width*2, height*2), Image.BICUBIC)
        
        # Бинаризация изображения
        threshold = 150
        img = img.point(lambda x: 0 if x < threshold else 255, '1')
        
        # Сохраняем обработанное изображение
        img.save(output_path)
        
        return output_path
    except Exception as e:
        print(f"Ошибка при улучшении изображения для OCR: {e}")
        return image_path

def extract_code_with_gpt(image_path, api_key, transcript_context=None):
    """Извлекает текст кода из изображения с помощью GPT-4o-mini"""
    try:
        # Кодируем изображение в base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Создаем клиент OpenAI
        client = OpenAI(api_key=api_key)
        
        # Формируем промпт на основе контекста из транскрипта
        prompt = "Если на изображении есть код, извлеки его и определи язык программирования. Ответь в формате JSON: {\"code\": \"извлеченный код\", \"language\": \"язык программирования\"}. Если код отсутствует, верни {\"code\": null, \"language\": null}."
        
        if transcript_context:
            # Добавляем контекст из транскрипта для улучшения распознавания
            context_text = " ".join([segment['text'] for segment in transcript_context])
            prompt += f" В видео в этот момент обсуждается: {context_text}"
        
        # Отправляем запрос к GPT-4o-mini
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }],
            max_tokens=2000
        )
        
        # Получаем результат
        result_text = response.choices[0].message.content.strip()
        
        # Пытаемся извлечь JSON из ответа
        try:
            # Ищем JSON в ответе (может быть окружен другим текстом)
            import re
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result_json = json.loads(json_match.group(0))
                code = result_json.get("code")
                language = result_json.get("language")
                
                # Если код пустой, возвращаем None
                if not code:
                    return None, None
                
                # Если язык не определен, используем резервную функцию
                if not language:
                    language = determine_code_language(code)
                
                return code, language
        except json.JSONDecodeError:
            # Если не удалось извлечь JSON, обрабатываем ответ как текст
            if "Изображение не содержит кода" in result_text or not result_text:
                return None, None
            
            # Пытаемся определить язык с помощью нашей функции
            language = determine_code_language(result_text)
            return result_text, language
            
        return None, None
    except Exception as e:
        print(f"Ошибка при извлечении кода с помощью GPT-4o-mini: {e}")
        return None, None

def extract_code_frames(video_path, output_dir, transcript_segments=None, interval=1, api_key=None):
    """Извлекает кадры, которые вероятно содержат код из видео, и анализирует их с GPT-4o-mini
    
    Args:
        video_path (str): Путь к видео файлу
        output_dir (str): Директория для сохранения извлеченных кадров
        transcript_segments (list): Список сегментов транскрипта с временными метками
        interval (int): Интервал между проверяемыми кадрами в секундах
        api_key (str): API ключ для OpenAI
    
    Returns:
        list: Список путей к сохраненным изображениям с кодом, временными метками и текстом кода
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Открываем видео файл
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть видео {video_path}")
        return []
    
    # Получаем информацию о видео
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(fps * interval)  # Проверять каждые interval секунд
    
    print(f"Анализ видео: всего {total_frames} кадров, FPS: {fps}")
    
    saved_images = []
    frame_count = 0
    last_code_time = -5  # Чтобы не сохранять похожие кадры слишком часто
    
    # Функция для определения потенциальных кадров с кодом
    def is_potential_code_frame(frame, transcript_segments=None, timestamp=None):
        """Быстрая проверка - содержит ли кадр потенциально код (без использования OCR)"""
        # Если есть контекст транскрипта, проверяем на ключевые слова
        if transcript_segments and timestamp is not None:
            relevant_segments = []
            for segment in transcript_segments:
                segment_start = segment['start']
                segment_end = segment_start + segment['duration']
                
                if segment_start - 5 <= timestamp <= segment_end + 5:
                    relevant_segments.append(segment)
            
            # Проверяем наличие ключевых слов, связанных с кодом
            code_keywords = ['код', 'функция', 'метод', 'класс', 'переменная', 'импорт', 'code', 'function', 'method', 'class', 'variable', 'import']
            for segment in relevant_segments:
                for keyword in code_keywords:
                    if keyword.lower() in segment['text'].lower():
                        return True
        
        # Базовая проверка на наличие прямоугольных областей
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Проверяем на наличие прямоугольных областей, которые могут быть блоками кода
        code_like_rects = 0
        frame_area = frame.shape[0] * frame.shape[1]
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            area = w * h
            area_ratio = float(area) / frame_area
            
            if w > frame.shape[1] * 0.2 and 0.05 < area_ratio < 0.8 and aspect_ratio > 2:
                code_like_rects += 1
        
        return code_like_rects > 3
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Проверяем только каждый frame_interval кадр для экономии времени
        if frame_count % frame_interval == 0:
            current_time = frame_count / fps
            
            # Проверяем, прошло ли достаточно времени с последнего сохраненного кода
            if current_time - last_code_time >= 3:  # Минимум 3 секунды между сохранениями
                # Получаем контекст из транскрипта для текущего времени, если доступен
                transcript_context = None
                if transcript_segments:
                    transcript_context = [
                        segment for segment in transcript_segments 
                        if segment['start'] - 5 <= current_time <= segment['start'] + segment['duration'] + 5
                    ]
                
                # Проверяем, содержит ли кадр потенциально код
                if is_potential_code_frame(frame, transcript_segments, current_time):
                    # Сохраняем кадр
                    timestamp = int(current_time)
                    image_path = f"{output_dir}/code_frame_{timestamp}.jpg"
                    
                    # Сохраняем с высоким качеством
                    cv2.imwrite(image_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                    
                    # Распознаем код из изображения с помощью GPT-4o-mini
                    if api_key:
                        code_text, code_lang = extract_code_with_gpt(image_path, api_key, transcript_context)
                        
                        # Если код успешно извлечен, добавляем в список результатов
                        if code_text:
                            saved_images.append((image_path, timestamp, code_text, code_lang))
                            last_code_time = current_time
                            print(f"Обнаружен код на {timestamp} секунде видео")
        
        frame_count += 1
        if frame_count % 1000 == 0:
            print(f"Обработано {frame_count}/{total_frames} кадров")
    
    cap.release()
    print(f"Извлечено {len(saved_images)} кадров с кодом")
    return saved_images

def get_image_base64(image_path):
    """Преобразует изображение в строку base64 для встраивания в markdown"""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            ext = os.path.splitext(image_path)[1].lstrip('.')
            return f"data:image/{ext};base64,{encoded_string}"
    except Exception as e:
        print(f"Ошибка при кодировании изображения в base64: {e}")
        return None

def analyze_with_llm(transcript_text, video_title, api_key, api_url, code_frames=None, prompt_template=None):
    """Анализ транскрипта с помощью LLM API (использует более доступную модель GPT-4o-mini)"""
    if not prompt_template:
        prompt_template = """
        Проанализируй следующий транскрипт видео "{title}".
        
        Транскрипт:
        {transcript}
        
        {code_context}
        
        Создай следующее:
        1. Краткое содержание (3-5 предложений)
        2. Основные тезисы и ключевые мысли (8-10 пунктов в виде маркированного списка)
        """
    
    # Формируем контекст кода, если имеются кадры с кодом
    code_context = ""
    if code_frames and len(code_frames) > 0:
        code_context = "В видео также были обнаружены следующие фрагменты кода:\n\n"
        for i, (_, timestamp, code_text, code_lang) in enumerate(code_frames):
            if code_text:
                minutes = timestamp // 60
                seconds = timestamp % 60
                code_context += f"Код на {minutes}:{seconds:02d}:\n```{code_lang}\n{code_text}\n```\n\n"
    
    # Объединяем транскрипт в текст, если передан список сегментов
    if isinstance(transcript_text, list):
        transcript_text = " ".join([segment["text"] for segment in transcript_text])
    
    prompt = prompt_template.format(
        title=video_title, 
        transcript=transcript_text,
        code_context=code_context
    )
    
    try:
        # Адаптируйте этот код в соответствии с API вашей LLM
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": "gpt-4o-mini",  # Используем более экономичную модель GPT-4o-mini
            "messages": [
                {"role": "system", "content": "Ты - аналитик видеоконтента. Твоя задача - выделять ключевые идеи и создавать подробный анализ содержания, включая понимание и объяснение кода, если он присутствует в видео."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 2000
        }
        
        response = requests.post(api_url, headers=headers, json=payload)
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

def save_results(video_id, video_title, transcript, analysis, code_frames=None, output_dir="results"):
    """Сохранение результатов анализа"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Извлекаем секции анализа
    summary_section = ""
    key_points_section = ""
    
    # Простой парсер для выделения секций
    if analysis:
        lines = analysis.split('\n')
        current_section = ""
        for line in lines:
            if "Краткое содержание" in line or "Резюме" in line:
                current_section = "summary"
                summary_section += line + "\n"
            elif "Основные тезисы" in line or "Ключевые моменты" in line or "Ключевые идеи" in line:
                current_section = "key_points"
                key_points_section += line + "\n"
            elif current_section == "summary":
                summary_section += line + "\n"
            elif current_section == "key_points":
                key_points_section += line + "\n"
    
    # Преобразуем транскрипт в текст, если он представлен как список сегментов
    transcript_text = transcript
    if isinstance(transcript, list):
        transcript_text = " ".join([segment["text"] for segment in transcript])
    
    # Сохраняем в формате Markdown
    md_file = f"{output_dir}/{video_id}_transcript.md"
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(f"# Анализ видео YouTube (ID: {video_id})\n\n")
        
        # Добавляем секцию резюме
        if summary_section:
            f.write(summary_section)
        else:
            f.write(analysis)
        
        # Добавляем секцию ключевых идей, если она есть
        if key_points_section:
            f.write(key_points_section)
        
        # Добавляем найденные кадры с кодом, если они есть
        if code_frames and len(code_frames) > 0:
            f.write("\n\n## Примеры кода из видео\n\n")
            for i, (image_path, timestamp, code_text, code_lang) in enumerate(code_frames):
                minutes = timestamp // 60
                seconds = timestamp % 60
                f.write(f"### Код на {minutes}:{seconds:02d}\n\n")
                
                # Добавляем изображение через base64, чтобы оно всегда отображалось
                image_base64 = get_image_base64(image_path)
                if image_base64:
                    f.write(f"![Пример кода {i+1}]({image_base64})\n\n")
                
                # Добавляем извлеченный код, если он доступен
                if code_text:
                    f.write(f"```{code_lang}\n{code_text}\n```\n\n")
        
        f.write("\n\n## Транскрипт\n\n")
        f.write(transcript_text)
    
    print(f"Транскрипт и анализ сохранены в Markdown файл: {md_file}")
        
    # Формируем имя файла на основе ID видео для JSON
    json_file = f"{output_dir}/{video_id}_analysis.json"
    
    # Подготавливаем данные для JSON
    code_frames_data = []
    if code_frames:
        for image_path, timestamp, code_text, code_lang in code_frames:
            code_frames_data.append({
                "image_path": image_path,
                "timestamp": timestamp,
                "code_text": code_text,
                "code_language": code_lang
            })
    
    results = {
        "video_id": video_id,
        "video_title": video_title,
        "transcript": transcript_text,
        "analysis": analysis,
        "code_frames": code_frames_data
    }
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    print(f"Результаты также сохранены в JSON: {json_file}")
    return md_file, json_file

def determine_code_language(code_text):
    """Определяет язык программирования на основе ключевых слов в коде"""
    # Словарь ключевых слов для основных языков программирования
    languages = {
        'python': ['def', 'import', 'class', 'for', 'while', 'if', 'elif', 'else', 'try', 'except', 'with', 'as', 'return', 'print', 'self', 'None', 'True', 'False', '==', ':', '!='],
        'javascript': ['function', 'const', 'let', 'var', 'return', 'if', 'else', 'for', 'while', 'new', 'this', 'class', 'extends', 'import', 'export', '=>', '===', '!==', 'undefined', 'null'],
        'java': ['public', 'private', 'protected', 'class', 'interface', 'extends', 'implements', 'void', 'static', 'final', 'new', 'return', 'this', 'super', 'instanceof'],
        'c#': ['public', 'private', 'protected', 'class', 'interface', 'using', 'namespace', 'void', 'static', 'string', 'int', 'bool', 'var', 'new', 'return', 'this', 'base'],
        'html': ['<!DOCTYPE', '<html', '<head', '<body', '<div', '<span', '<p>', '<a', '<img', '<script', '<style', '<link', '<meta', 'href=', 'src='],
        'css': ['{', '}', ':', ';', 'margin', 'padding', 'color', 'background', 'font-', 'width', 'height', 'display', 'position', '@media', '@keyframes'],
        'bash': ['#!/bin', 'echo', 'export', 'if', 'then', 'else', 'fi', 'for', 'do', 'done', 'while', 'case', 'esac', '$', '&&']
    }
    
    # Подсчитываем количество вхождений ключевых слов для каждого языка
    scores = {lang: 0 for lang in languages}
    
    for lang, keywords in languages.items():
        for keyword in keywords:
            if keyword in code_text:
                scores[lang] += 1
    
    # Возвращаем язык с наибольшим количеством вхождений
    if max(scores.values()) > 0:
        return max(scores.items(), key=lambda x: x[1])[0]
    
    # Если не удалось определить язык, возвращаем 'text'
    return 'text'

def main():
    # Загружаем переменные окружения из .env файла
    load_dotenv()
    
    parser = argparse.ArgumentParser(description='Анализ YouTube видео и транскриптов с помощью LLM')
    parser.add_argument('url', help='URL видео на YouTube')
    parser.add_argument('--api-key', help='API ключ для LLM (если не указан, берется из переменной окружения OPENAI_API_KEY)')
    parser.add_argument('--api-url', default="https://api.openai.com/v1/chat/completions", 
                        help='URL API для LLM')
    parser.add_argument('--whisper-model', default="tiny", 
                        choices=["tiny", "base", "small", "medium", "large"],
                        help='Размер модели Whisper для транскрибирования (tiny - самая быстрая, large - самая точная)')
    parser.add_argument('--output', default="results", help='Директория для выходных файлов')
    parser.add_argument('--temp', default="temp", help='Директория для временных файлов')
    parser.add_argument('--extract-code', action='store_true', help='Извлекать кадры с кодом из видео')
    parser.add_argument('--video-quality', default="highest", choices=["lowest", "highest"],
                        help='Качество скачиваемого видео для извлечения кода')
    
    args = parser.parse_args()
    
    # Используем API ключ из аргументов или из переменной окружения
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Ошибка: API ключ не указан. Укажите его через аргумент --api-key или в .env файле.")
        return
    
    # Получаем ID видео
    video_id = get_video_id(args.url)
    
    # Пытаемся получить транскрипт с YouTube
    transcript_segments = get_youtube_transcript(video_id)
    
    video_title = None
    audio_file = None
    code_frames = []
    
    # Если транскрипт недоступен, создаем его с помощью Whisper
    if not transcript_segments:
        print("Транскрипт недоступен на YouTube. Создаем транскрипт с помощью Whisper...")
        audio_file, video_title = download_audio(args.url, args.temp)
        
        if audio_file:
            transcript_segments = create_transcript_with_whisper(audio_file, args.whisper_model)
        else:
            print("Не удалось загрузить аудио. Прерываем выполнение.")
            return
    else:
        # Если транскрипт доступен, получаем название видео
        try:
            yt = YouTube(args.url)
            video_title = yt.title
        except Exception as e:
            print(f"Попытка получения названия без po_token не удалась: {e}")
            yt = YouTube(args.url, use_po_token=True)
            video_title = yt.title
    
    if not transcript_segments:
        print("Не удалось получить или создать транскрипт. Прерываем выполнение.")
        return
    
    # Создаем единый текст транскрипта для анализа
    transcript_text = " ".join([segment["text"] for segment in transcript_segments])
    
    # Если включена опция извлечения кода, скачиваем видео и извлекаем кадры с кодом
    if args.extract_code:
        print("Извлечение кадров с кодом из видео...")
        video_file = download_video(args.url, args.temp, quality=args.video_quality)
        if video_file:
            # Создаем папку для кадров с кодом в директории вывода
            code_frames_dir = os.path.join(args.output, "code_frames", video_id)
            if not os.path.exists(code_frames_dir):
                os.makedirs(code_frames_dir)
            
            # Извлекаем кадры с кодом, передавая контекст транскрипта и API ключ для GPT-4o-mini
            code_frames = extract_code_frames(video_file, code_frames_dir, transcript_segments, api_key=api_key)
    
    # Анализируем транскрипт с помощью LLM, передавая информацию о коде
    analysis = analyze_with_llm(transcript_text, video_title, api_key, args.api_url, code_frames)
    
    if analysis:
        # Сохраняем результаты
        md_file, json_file = save_results(video_id, video_title, transcript_segments, analysis, code_frames, args.output)
        print(f"Анализ завершен успешно. Результаты сохранены в {md_file} и {json_file}")
    else:
        print("Не удалось выполнить анализ транскрипта.")

if __name__ == "__main__":
    main()