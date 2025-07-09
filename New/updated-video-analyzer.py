#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Скрипт для анализа локальных видеофайлов с транскрипцией и адаптивным извлечением скриншотов
"""

import os
import sys
import json
import requests
import whisper
import cv2
import numpy as np
import base64
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
import re

# Импортируем адаптивный экстрактор
from adaptive_screenshot_extractor import AdaptiveScreenshotExtractor
# Импортируем умный экстрактор на основе транскрипта  
from smart_transcript_extractor import SmartTranscriptExtractor
# Импортируем систему кэширования
from cache_manager import CacheManager

# Импортируем хронологический процессор
try:
    from chronological_transcript_processor import ChronologicalTranscriptProcessor
    CHRONOLOGICAL_AVAILABLE = True
except ImportError:
    CHRONOLOGICAL_AVAILABLE = False
    print("⚠️  Хронологический процессор недоступен. Переименуйте chronological-transcript-processor.py в chronological_transcript_processor.py")

def extract_audio_from_video(video_path, output_dir="temp", cache_manager=None):
    """Извлекает аудио из видеофайла для последующей транскрипции"""
    
    # Проверяем кэш, если менеджер кэша предоставлен
    if cache_manager:
        cached_audio = cache_manager.get_cached_audio(video_path)
        if cached_audio:
            return cached_audio
    
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Определяем имя выходного аудиофайла
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        audio_path = os.path.join(output_dir, f"{video_name}_audio.wav")
        
        print(f"Извлечение аудио из {video_path}...")
        
        # Используем cv2 для извлечения аудио (простой способ)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Не удалось открыть видеофайл: {video_path}")
        
        # Для извлечения аудио используем ffmpeg через os.system
        import subprocess
        
        # Команда ffmpeg для извлечения аудио
        cmd = [
            "ffmpeg", "-y",  # -y для перезаписи файла
            "-i", video_path,
            "-vn",  # не включать видео
            "-acodec", "pcm_s16le",  # кодек для wav
            "-ar", "16000",  # частота дискретизации 16kHz (оптимально для Whisper)
            "-ac", "1",  # моно
            audio_path
        ]
        
        print("Выполняется извлечение аудио...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Ошибка ffmpeg: {result.stderr}")
            # Пробуем альтернативный метод через moviepy
            try:
                from moviepy.editor import VideoFileClip
                print("Пробуем альтернативный метод через moviepy...")
                video_clip = VideoFileClip(video_path)
                audio_clip = video_clip.audio
                audio_clip.write_audiofile(audio_path, verbose=False, logger=None)
                audio_clip.close()
                video_clip.close()
            except ImportError:
                print("Для извлечения аудио нужно установить ffmpeg или moviepy")
                print("Установка: pip install moviepy")
                return None
        
        if os.path.exists(audio_path):
            print(f"Аудио успешно извлечено: {audio_path}")
            
            # Сохраняем в кэш, если менеджер кэша предоставлен
            if cache_manager:
                cached_path = cache_manager.save_audio_cache(video_path, audio_path)
                return cached_path
            
            return audio_path
        else:
            print("Не удалось извлечь аудио")
            return None
            
    except Exception as e:
        print(f"Ошибка при извлечении аудио: {e}")
        return None

def create_transcript_with_whisper(audio_file, model_size="base", video_path=None, cache_manager=None):
    """Создание транскрипта с помощью Whisper"""
    
    # Проверяем кэш, если менеджер кэша предоставлен
    if cache_manager and video_path:
        cached_transcript = cache_manager.get_cached_transcript(video_path)
        if cached_transcript:
            return cached_transcript
    
    try:
        print(f"Загрузка модели Whisper ({model_size})...")
        model = whisper.load_model(model_size)
        
        print("Транскрибирование аудио...")
        result = model.transcribe(audio_file, language="ru")
        
        # Возвращаем сегменты с временными метками
        segments = []
        for segment in result["segments"]:
            segments.append({
                "text": segment["text"],
                "start": segment["start"],
                "duration": segment["end"] - segment["start"]
            })
        
        full_text = result["text"]
        
        # Сохраняем в кэш, если менеджер кэша предоставлен
        if cache_manager and video_path:
            cache_manager.save_transcript_cache(video_path, segments, full_text)
        
        return segments, full_text
    except Exception as e:
        print(f"Ошибка при создании транскрипта с Whisper: {e}")
        return None, None

def extract_screenshots_traditional(video_path, output_dir, interval=30, api_key=None):
    """Традиционное извлечение скриншотов через заданные интервалы"""
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
    duration = total_frames / fps
    frame_interval = int(fps * interval)  # Кадры через каждые interval секунд
    
    print(f"Анализ видео: всего {total_frames} кадров, FPS: {fps}, длительность: {duration:.1f} сек")
    print(f"Будет извлечено ~{int(duration // interval)} скриншотов")
    
    saved_screenshots = []
    frame_count = 0
    screenshot_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Сохраняем кадр каждые interval секунд
        if frame_count % frame_interval == 0:
            current_time = frame_count / fps
            timestamp = int(current_time)
            
            # Имя файла скриншота
            screenshot_path = f"{output_dir}/screenshot_{timestamp:05d}s.jpg"
            
            # Сохраняем кадр с высоким качеством
            cv2.imwrite(screenshot_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            description = None
            if api_key:
                # Анализируем скриншот с помощью GPT-4V
                description = analyze_screenshot_with_gpt4v(screenshot_path, api_key)
            
            saved_screenshots.append((screenshot_path, timestamp, description, "periodic"))
            screenshot_count += 1
            
            minutes = timestamp // 60
            seconds = timestamp % 60
            print(f"Скриншот {screenshot_count}: {minutes}:{seconds:02d} - {screenshot_path}")
        
        frame_count += 1
    
    cap.release()
    print(f"Извлечено {len(saved_screenshots)} скриншотов")
    return saved_screenshots

def extract_screen_content(image_path, api_key):
    """Извлекает текстовый контент, код, таблицы с экрана"""
    try:
        # Кодируем изображение в base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Создаем клиент OpenAI
        client = OpenAI(api_key=api_key)
        
        # Промпт для извлечения контента
        prompt = """Извлеки весь текстовый контент с этого скриншота:

1. **Код** - если есть программный код, скопируй его точно
2. **Текст** - весь читаемый текст на экране
3. **Таблицы** - данные в табличном формате
4. **Диаграммы** - описание схем и диаграмм
5. **UI элементы** - названия кнопок, меню, полей

Верни в JSON формате:
{
    "content_type": "code/text/table/diagram/ui",
    "extracted_text": "весь извлеченный текст",
    "code_snippets": ["фрагменты кода если есть"],
    "table_data": "данные таблиц если есть",
    "ui_elements": ["элементы интерфейса"],
    "description": "краткое описание содержимого"
}

ВАЖНО: Извлекай текст ТОЧНО как написано, включая код и данные."""
        
        # Отправляем запрос к GPT-4V
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
            response_format={"type": "json_object"},
            max_tokens=1500
        )
        
        # Парсим результат
        result = json.loads(response.choices[0].message.content)
        return result
        
    except Exception as e:
        print(f"Ошибка при извлечении контента экрана: {e}")
        return None

def analyze_screenshot_with_gpt4v(image_path, api_key):
    """Анализирует скриншот с помощью GPT-4V"""
    try:
        # Кодируем изображение в base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Создаем клиент OpenAI
        client = OpenAI(api_key=api_key)
        
        # Промпт для анализа скриншота
        prompt = """Опиши содержимое этого скриншота из видео встречи. Включи:
1. Что видно на экране (интерфейс, документы, презентация, код и т.д.)
2. Основные элементы и их содержание
3. Если есть текст, кратко передай его суть
4. Общая тема/контекст происходящего

Ответ должен быть кратким но информативным, на русском языке."""
        
        # Отправляем запрос к GPT-4V
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
            max_tokens=500
        )
        
        # Получаем результат
        description = response.choices[0].message.content.strip()
        return description
        
    except Exception as e:
        print(f"Ошибка при анализе скриншота с GPT-4V: {e}")
        return None

def analyze_content_with_gpt(transcript_text, api_key, video_title="Локальное видео", screenshots_info=None, video_path=None, cache_manager=None):
    """Анализ транскрипта и скриншотов с помощью GPT"""
    
    # Проверяем кэш, если менеджер кэша предоставлен
    if cache_manager and video_path:
        cached_analysis = cache_manager.get_cached_analysis(video_path, "basic")
        if cached_analysis:
            return cached_analysis
    
    # Добавляем информацию о скриншотах в промпт
    screenshots_summary = ""
    if screenshots_info and len(screenshots_info) > 0:
        screenshots_summary = "\n\nИнформация о ключевых моментах видео (на основе скриншотов):\n"
        for i, (_, timestamp, description, reason) in enumerate(screenshots_info[:10]):  # Берем первые 10
            if description:
                minutes = int(timestamp // 60)
                seconds = int(timestamp % 60)
                screenshots_summary += f"\n{minutes}:{seconds:02d} - {description[:200]}..."
    
    prompt = f"""
    Проанализируй следующий транскрипт видео "{video_title}".
    
    Транскрипт:
    {transcript_text}
    {screenshots_summary}
    
    Создай следующее:
    1. Краткое содержание (3-5 предложений)
    2. Основные тезисы и ключевые мысли (8-10 пунктов в виде маркированного списка)
    3. Визуальные материалы (если были показаны презентации, код, демо - опиши что именно)
    4. Выводы и рекомендации (если применимо)
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
            
            # Сохраняем в кэш, если менеджер кэша предоставлен
            if cache_manager and video_path:
                cache_manager.save_analysis_cache(video_path, analysis, "basic")
            
            return analysis
        else:
            print(f"Ошибка при анализе: {result}")
            return None
    except Exception as e:
        print(f"Ошибка при анализе транскрипта: {e}")
        return None

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

def save_chronological_results(video_name, chronological_data, output_dir="results"):
    """Сохранение ИНТЕГРИРОВАННОГО хронологического отчета со скриншотами в диалоге"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    speakers = chronological_data['speakers']
    timeline = chronological_data['timeline']
    structured_content = chronological_data['structured_content']
    report = chronological_data['report']
    
    # Создаем ЕДИНУЮ временную линию с объединенными событиями
    unified_timeline = []
    
    # Объединяем все события и сортируем по времени
    for event in timeline:
        unified_timeline.append(event)
    
    # Сортируем все события по времени
    unified_timeline.sort(key=lambda x: x.timestamp)
    
    # Сохраняем в формате Markdown
    md_file = f"{output_dir}/{video_name}_INTEGRATED_chronological.md"
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(f"# 🎬 ИНТЕГРИРОВАННЫЙ анализ встречи: {video_name}\n\n")
        
        # Добавляем краткую информацию об участниках
        f.write("## 👥 Участники\n\n")
        for speaker_id, speaker in speakers.items():
            name = speaker.name or f"Участник {speaker_id[-1]}"
            role = speaker.role or "участник"
            f.write(f"- **{name}** ({role})")
            if speaker.characteristics:
                f.write(f" - {', '.join(speaker.characteristics[:2])}")
            f.write("\n")
        
        # Добавляем основной анализ
        f.write(f"\n## 📋 Краткий анализ\n\n{report}\n\n")
        
        # ГЛАВНАЯ ЧАСТЬ - ИНТЕГРИРОВАННАЯ ВРЕМЕННАЯ ЛИНИЯ
        f.write("## 🎯 ХРОНОЛОГИЧЕСКИЙ ХОД ВСТРЕЧИ (с интегрированными скриншотами)\n\n")
        f.write("*Все события расположены в хронологическом порядке. Скриншоты показывают, что происходило на экране в момент разговора.*\n\n")
        f.write("---\n\n")
        
        current_topic = "Начало встречи"
        last_speaker = None
        screenshot_counter = 1
        
        for event in unified_timeline:
            minutes = int(event.timestamp // 60)
            seconds = int(event.timestamp % 60)
            timestamp_str = f"{minutes}:{seconds:02d}"
            
            # Обработка смены темы
            if event.type == 'topic_change':
                current_topic = event.content.get('topic', 'Новая тема')
                f.write(f"\n\n### 📌 [{timestamp_str}] ТЕМА: {current_topic.upper()}\n\n")
                continue
            
            # Обработка транскрипта
            elif event.type == 'transcript':
                speaker_id = event.content.get('speaker_id', 'unknown')
                speaker_name = event.content.get('speaker_name')
                
                if not speaker_name:
                    speaker_obj = speakers.get(speaker_id)
                    speaker_name = speaker_obj.name if speaker_obj else f"Участник {speaker_id[-1] if speaker_id != 'unknown' else '1'}"
                
                # Используем исправленный текст если есть
                text = event.content.get('corrected_text', event.content.get('text', ''))
                
                # Показываем общий контекст группы, если это первая реплика в группе
                overall_context = event.content.get('overall_context')
                if overall_context:
                    f.write(f"\n💡 **Контекст момента:** *{overall_context}*\n\n")
                
                # Показываем смену говорящего более заметно
                if speaker_name != last_speaker:
                    f.write(f"\n**[{timestamp_str}] {speaker_name}:** {text}")
                    last_speaker = speaker_name
                else:
                    f.write(f" {text}")
                
                # Добавляем контекстные пояснения
                context_explanation = event.content.get('context_explanation')
                if context_explanation:
                    f.write(f" *[{context_explanation}]*")
                
                # Добавляем визуальные ссылки
                visual_reference = event.content.get('visual_reference')
                if visual_reference:
                    f.write(f" 👁️ *{visual_reference}*")
                
                f.write("\n")
            
            # Обработка скриншотов - ВСТРАИВАЕМ ПРЯМО В ДИАЛОГ
            elif event.type == 'screenshot':
                reason = event.content.get('reason', 'Важный момент')
                description = event.content.get('description', '')
                image_path = event.content.get('path', '')
                
                f.write(f"\n\n📸 **[{timestamp_str}] СКРИНШОТ #{screenshot_counter}: {reason}**\n\n")
                
                # Добавляем изображение ПРЯМО В ПОТОК
                if image_path and os.path.exists(image_path):
                    image_base64 = get_image_base64(image_path)
                    if image_base64:
                        f.write(f"![Скриншот {screenshot_counter}]({image_base64})\n\n")
                
                # Добавляем описание того, что на экране
                if description:
                    f.write(f"*На экране: {description}*\n\n")
                else:
                    f.write(f"*Визуальное изменение на экране в момент разговора*\n\n")
                
                screenshot_counter += 1
                f.write("---\n")
        
        # Добавляем финальную статистику
        f.write(f"\n\n## 📊 Статистика встречи\n\n")
        total_duration = max([e.timestamp for e in unified_timeline]) if unified_timeline else 0
        total_screenshots = len([e for e in unified_timeline if e.type == 'screenshot'])
        total_topics = len([e for e in unified_timeline if e.type == 'topic_change'])
        
        f.write(f"- **Длительность:** {int(total_duration // 60)}:{int(total_duration % 60):02d}\n")
        f.write(f"- **Участников:** {len(speakers)}\n")
        f.write(f"- **Скриншотов:** {total_screenshots}\n")
        f.write(f"- **Смен темы:** {total_topics}\n")
        
        # Статистика по участникам
        f.write(f"\n### Активность участников:\n")
        for speaker_id, speaker in speakers.items():
            name = speaker.name or f"Участник {speaker_id[-1]}"
            segments_count = len(speaker.voice_segments) if speaker.voice_segments else 0
            f.write(f"- **{name}:** {segments_count} реплик\n")
    
    print(f"🎯 ИНТЕГРИРОВАННЫЙ хронологический анализ сохранен: {md_file}")
    
    # Также сохраняем JSON с подробными данными
    json_file = f"{output_dir}/{video_name}_integrated_data.json"
    json_data = {
        "video_name": video_name,
        "total_duration": max([e.timestamp for e in unified_timeline]) if unified_timeline else 0,
        "speakers": {
            speaker_id: {
                "name": speaker.name,
                "role": speaker.role,
                "characteristics": speaker.characteristics,
                "segments_count": len(speaker.voice_segments) if speaker.voice_segments else 0
            }
            for speaker_id, speaker in speakers.items()
        },
        "timeline_events_count": {
            "total": len(unified_timeline),
            "transcript": len([e for e in unified_timeline if e.type == 'transcript']),
            "screenshots": len([e for e in unified_timeline if e.type == 'screenshot']),
            "topic_changes": len([e for e in unified_timeline if e.type == 'topic_change'])
        },
        "report": report
    }
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)
    
    return md_file, json_file

def save_results(video_name, transcript_segments, full_transcript, analysis, screenshots=None, output_dir="results"):
    """Сохранение результатов анализа"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Сохраняем в формате Markdown
    md_file = f"{output_dir}/{video_name}_analysis.md"
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(f"# Анализ видео: {video_name}\n\n")
        
        # Добавляем анализ
        if analysis:
            f.write(analysis)
        
        # Добавляем скриншоты, если они есть
        if screenshots and len(screenshots) > 0:
            f.write("\n\n## Ключевые моменты видео\n\n")
            
            # Группируем скриншоты по типу
            ai_screenshots = [s for s in screenshots if s[3] != "periodic"]
            periodic_screenshots = [s for s in screenshots if s[3] == "periodic"]
            
            if ai_screenshots:
                f.write("### 🤖 Автоматически определенные важные моменты\n\n")
                for i, (image_path, timestamp, description, reason) in enumerate(ai_screenshots):
                    minutes = timestamp // 60
                    seconds = timestamp % 60
                    f.write(f"#### {minutes}:{seconds:02d} - {reason}\n\n")
                    
                    # Добавляем изображение через base64
                    image_base64 = get_image_base64(image_path)
                    if image_base64:
                        f.write(f"![Скриншот]({image_base64})\n\n")
                    
                    # Добавляем описание, если есть
                    if description:
                        f.write(f"{description}\n\n")
                    
                    f.write("---\n\n")
            
            # Добавляем периодические скриншоты в конце, если нужно
            if periodic_screenshots and len(ai_screenshots) < 5:
                f.write("### 📸 Дополнительные скриншоты\n\n")
                for i, (image_path, timestamp, description, _) in enumerate(periodic_screenshots[:5]):
                    minutes = timestamp // 60
                    seconds = timestamp % 60
                    f.write(f"#### {minutes}:{seconds:02d}\n\n")
                    
                    image_base64 = get_image_base64(image_path)
                    if image_base64:
                        f.write(f"![Скриншот]({image_base64})\n\n")
                    
                    if description:
                        f.write(f"{description}\n\n")
                    
                    f.write("---\n\n")
        
        f.write("\n\n## Полный транскрипт\n\n")
        f.write(full_transcript)
        
        # Добавляем транскрипт с временными метками
        if transcript_segments:
            f.write("\n\n## Транскрипт с временными метками\n\n")
            for segment in transcript_segments:
                start_time = segment['start']
                minutes = int(start_time // 60)
                seconds = int(start_time % 60)
                f.write(f"**{minutes}:{seconds:02d}** - {segment['text']}\n\n")
    
    print(f"Анализ сохранен в Markdown файл: {md_file}")
    
    # Сохраняем JSON
    json_file = f"{output_dir}/{video_name}_analysis.json"
    
    # Подготавливаем данные для JSON
    screenshots_data = []
    if screenshots:
        for image_path, timestamp, description, reason in screenshots:
            screenshots_data.append({
                "image_path": image_path,
                "timestamp": timestamp,
                "description": description,
                "reason": reason
            })
    
    results = {
        "video_name": video_name,
        "full_transcript": full_transcript,
        "transcript_segments": transcript_segments,
        "analysis": analysis,
        "screenshots": screenshots_data
    }
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    print(f"Результаты также сохранены в JSON: {json_file}")
    
    return md_file, json_file

def main():
    load_dotenv()
    
    if len(sys.argv) < 2:
        print("Использование:")
        print("  python local_video_analyzer.py <путь_к_видео> [опции]")
        print("\nОпции:")
        print("  --whisper-model MODEL      Модель Whisper (tiny, base, small, medium, large)")
        print("  --screenshot-interval N    Интервал между скриншотами в секундах (по умолчанию: 30)")
        print("  --no-screenshots          Не извлекать скриншоты")
        print("  --smart-screenshots       Использовать ИИ для умного извлечения скриншотов")
        print("  --screenshot-mode MODE    Режим скриншотов: 'periodic', 'smart', 'transcript', 'both' (по умолчанию: transcript)")
        print("  --chronological           Создать хронологический отчет с участниками и коррекцией")
        print("  --output DIR              Директория для результатов (по умолчанию: results)")
        print("  --clear-cache             Очистить кэш для этого видео")
        print("  --force-refresh           Принудительно обновить все данные (игнорировать кэш)")
        print("  --cache-status            Показать только статус кэша и выйти")
        print("\nПример:")
        print("  python local_video_analyzer.py video.mp4 --whisper-model base --chronological")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    # Парсим аргументы
    whisper_model = "base"
    screenshot_interval = 30
    extract_screenshots_flag = True
    screenshot_mode = "transcript"  # По умолчанию используем анализ транскрипта
    chronological_mode = False
    output_dir = "results"
    clear_cache = False
    force_refresh = False
    cache_status_only = False
    
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--whisper-model" and i + 1 < len(sys.argv):
            whisper_model = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--screenshot-interval" and i + 1 < len(sys.argv):
            screenshot_interval = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--no-screenshots":
            extract_screenshots_flag = False
            i += 1
        elif sys.argv[i] == "--smart-screenshots":
            screenshot_mode = "smart"
            i += 1
        elif sys.argv[i] == "--screenshot-mode" and i + 1 < len(sys.argv):
            screenshot_mode = sys.argv[i + 1]
            if screenshot_mode not in ["periodic", "smart", "transcript", "both"]:
                print(f"Неверный режим скриншотов: {screenshot_mode}")
                sys.exit(1)
            i += 2
        elif sys.argv[i] == "--output" and i + 1 < len(sys.argv):
            output_dir = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--chronological":
            chronological_mode = True
            i += 1
        elif sys.argv[i] == "--clear-cache":
            clear_cache = True
            i += 1
        elif sys.argv[i] == "--force-refresh":
            force_refresh = True
            i += 1
        elif sys.argv[i] == "--cache-status":
            cache_status_only = True
            i += 1
        else:
            i += 1
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("Предупреждение: API ключ OpenAI не найден. Анализ скриншотов и содержания будет пропущен.")
        print("Для полного анализа добавьте OPENAI_API_KEY в .env файл")
        if screenshot_mode in ["smart", "transcript"]:
            screenshot_mode = "periodic"  # Переключаемся на периодический режим без API
    
    if not os.path.exists(video_path):
        print(f"Ошибка: Видеофайл {video_path} не найден")
        sys.exit(1)
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"Анализ видео: {video_name}")
    print(f"Режим скриншотов: {screenshot_mode}")
    
    try:
        # Создаем менеджер кэша
        cache_manager = CacheManager()
        
        # Обрабатываем специальные команды кэша
        if clear_cache:
            print("🧹 Очищаем кэш для видео...")
            cache_manager.clear_cache(video_path)
            print("✅ Кэш очищен")
            return
        
        if cache_status_only:
            cache_manager.print_cache_status(video_path)
            return
        
        # Показываем статус кэша
        cache_manager.print_cache_status(video_path)
        
        # Если принудительное обновление, временно "отключаем" кэш
        if force_refresh:
            print("🔄 Принудительное обновление - игнорируем кэш")
            cache_manager = None
        
        # Извлекаем аудио (с проверкой кэша)
        audio_path = extract_audio_from_video(video_path, cache_manager=cache_manager)
        if not audio_path:
            print("Не удалось извлечь аудио. Завершение работы.")
            sys.exit(1)
        
        # Создаем транскрипт с помощью Whisper (с проверкой кэша)
        transcript_segments, full_transcript = create_transcript_with_whisper(
            audio_path, whisper_model, video_path=video_path, cache_manager=cache_manager
        )
        if not full_transcript:
            print("Не удалось создать транскрипт. Завершение работы.")
            sys.exit(1)
        
        print(f"Транскрипт создан. Длина: {len(full_transcript)} символов")
        
        # Извлекаем скриншоты, если нужно
        screenshots = []
        if extract_screenshots_flag:
            screenshots_dir = os.path.join(output_dir, f"{video_name}_screenshots")
            
            # Проверяем кэш скриншотов
            cached_screenshots = cache_manager.get_cached_screenshots(video_path, screenshot_mode)
            if cached_screenshots:
                print(f"📸 Используем кэшированные скриншоты режима '{screenshot_mode}'")
                screenshots = [(shot['path'], shot['timestamp'], shot['description'], shot['reason']) 
                              for shot in cached_screenshots]
            else:
                if screenshot_mode == "transcript":
                    print("\n🧠 Используем умный анализ транскрипта для определения ключевых моментов...")
                    extractor = SmartTranscriptExtractor(api_key)
                    screenshots = extractor.extract_screenshots(
                        video_path, screenshots_dir, transcript_segments
                    )
                
                elif screenshot_mode == "smart":
                    print("\n🤖 Используем ИИ для интеллектуального извлечения скриншотов...")
                    extractor = AdaptiveScreenshotExtractor(api_key)
                    screenshots = extractor.extract_screenshots(
                        video_path, screenshots_dir, transcript_segments
                    )
                
                elif screenshot_mode == "periodic":
                    print(f"\n📸 Извлекаем скриншоты каждые {screenshot_interval} секунд...")
                    screenshots = extract_screenshots_traditional(
                        video_path, screenshots_dir, screenshot_interval, api_key
                    )
                
                elif screenshot_mode == "both":
                    print("\n🔬 Используем комбинированный подход: анализ транскрипта + периодические...")
                    # Сначала анализируем транскрипт
                    transcript_extractor = SmartTranscriptExtractor(api_key)
                    transcript_screenshots = transcript_extractor.extract_screenshots(
                        video_path, screenshots_dir, transcript_segments
                    )
                    
                    # Затем добавляем периодические (реже)
                    periodic_screenshots = extract_screenshots_traditional(
                        video_path, screenshots_dir, screenshot_interval * 2, api_key  # Реже для комбинации
                    )
                    
                    # Объединяем, избегая дубликатов
                    screenshots = transcript_screenshots
                    for p_shot in periodic_screenshots:
                        # Проверяем, нет ли близкого умного скриншота
                        is_duplicate = any(
                            abs(s[1] - p_shot[1]) < 10  # В пределах 10 секунд
                            for s in transcript_screenshots
                        )
                        if not is_duplicate:
                            screenshots.append(p_shot)
                    
                    # Сортируем по времени
                    screenshots.sort(key=lambda x: x[1])
                
                # Сохраняем скриншоты в кэш после извлечения
                if screenshots:
                    cache_manager.save_screenshots_cache(video_path, screenshots, screenshot_mode)
        
        # Анализируем содержание
        analysis = None
        if api_key:
            print("\n📊 Анализ содержания с помощью GPT...")
            analysis = analyze_content_with_gpt(
                full_transcript, api_key, video_name, screenshots, 
                video_path=video_path, cache_manager=cache_manager
            )
        
        # Обрабатываем хронологические данные, если нужно
        if chronological_mode and CHRONOLOGICAL_AVAILABLE and api_key:
            print("\n🎬 Создаем хронологический анализ...")
            
            # Создаем контекст видео для хронологического процессора
            video_context = {
                'meeting_type': 'discussion',
                'main_topics': [],
                'visual_content_probability': 0.5
            }
            
            # Преобразуем скриншоты в нужный формат
            screenshots_formatted = []
            if screenshots:
                if isinstance(screenshots[0], dict):
                    # Новый формат из AdaptiveScreenshotExtractor
                    screenshots_formatted = [
                        (shot['path'], shot['timestamp'], shot.get('description'), shot.get('decision', {}).get('reason', 'screenshot'))
                        for shot in screenshots
                    ]
                else:
                    # Старый формат
                    screenshots_formatted = screenshots
            
            # Запускаем хронологическую обработку
            processor = ChronologicalTranscriptProcessor(api_key)
            chronological_data = processor.process_video_meeting(
                transcript_segments, screenshots_formatted, video_context
            )
            
            # Сохраняем хронологические результаты
            chron_md, chron_json = save_chronological_results(video_name, chronological_data, output_dir)
            
            print(f"\n✅ Хронологический анализ готов!")
            print(f"  🎬 Хронологический отчет: {chron_md}")
            print(f"  📋 Подробные данные: {chron_json}")
        
        elif chronological_mode and not CHRONOLOGICAL_AVAILABLE:
            print("⚠️  Хронологический режим недоступен. Переименуйте chronological-transcript-processor.py")
        
        elif chronological_mode and not api_key:
            print("⚠️  Для хронологического анализа нужен API ключ OpenAI")
        
        # Сохраняем обычные результаты (если не только хронологический режим)
        if not chronological_mode or not CHRONOLOGICAL_AVAILABLE or not api_key:
            md_file, json_file = save_results(video_name, transcript_segments, full_transcript, analysis, screenshots, output_dir)
        
        print("\n✅ Готово!")
        print(f"Результаты сохранены:")
        print(f"  📄 Markdown: {md_file}")
        print(f"  📋 JSON: {json_file}")
        if screenshots:
            print(f"  📸 Скриншоты: {len(screenshots)} файлов")
            if screenshot_mode == "transcript":
                print(f"     🧠 На основе анализа транскрипта найдено {len(screenshots)} ключевых моментов")
            elif screenshot_mode == "smart":
                ai_count = len([s for s in screenshots if s[3] != "periodic"])
                print(f"     🤖 ИИ определил {ai_count} важных моментов")
            elif screenshot_mode == "both":
                transcript_count = len([s for s in screenshots if getattr(s, 'method', 'transcript') == 'transcript'])
                periodic_count = len([s for s in screenshots if s[3] == "periodic"])
                print(f"     🧠 Транскрипт: {transcript_count}, 📸 Периодические: {periodic_count}")
        
        # Сохраняем метаданные обработки
        metadata = {
            'whisper_model': whisper_model,
            'screenshot_mode': screenshot_mode,
            'chronological_mode': chronological_mode,
            'transcript_length': len(full_transcript),
            'segments_count': len(transcript_segments),
            'screenshots_count': len(screenshots) if screenshots else 0
        }
        cache_manager.save_metadata(video_path, metadata)
        
        # Очищаем временный аудиофайл (НЕ из кэша)
        if audio_path and os.path.exists(audio_path) and not audio_path.startswith(cache_manager.cache_dir):
            os.remove(audio_path)
            print("\n🧹 Временный аудиофайл удален")
        
    except Exception as e:
        print(f"❌ Произошла ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
