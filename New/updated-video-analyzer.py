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
import tempfile
import time # Добавляем для mlx_whisper

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

def get_optimal_device():
    """Определяет оптимальное устройство для выполнения (GPU для M1/M2 Mac или CPU)"""
    import torch
    import platform
    
    # Проверяем платформу
    system = platform.system()
    machine = platform.machine()
    
    # Для Apple Silicon Mac
    if system == "Darwin" and machine == "arm64":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            print("🚀 Обнаружен Apple Silicon с поддержкой Metal Performance Shaders (MPS)")
            print("   Whisper будет использовать GPU для ускорения обработки")
            return "mps"
    
    # Для других систем с CUDA
    if torch.cuda.is_available():
        print("🚀 Обнаружена CUDA GPU")
        return "cuda"
    
    print("💻 Используется CPU (для ускорения рекомендуется GPU)")
    return "cpu"

def load_whisper_optimized(model_size="base"):
    """Загружает модель Whisper с оптимизацией для конкретного устройства"""
    import torch
    
    device = get_optimal_device()
    
    print(f"📥 Загрузка модели Whisper ({model_size}) на {device}...")
    
    # Загружаем модель
    model = whisper.load_model(model_size, device=device)
    
    # Для Apple Silicon с MPS нужна специальная обработка
    if device == "mps":
        # Whisper может не полностью поддерживать MPS, поэтому используем смешанный подход
        print("   Применяем оптимизации для Apple Silicon...")
        # Модель остается на CPU, но операции будут ускорены через Metal
        # Это связано с особенностями работы Whisper
        
        # Устанавливаем потоки для оптимальной работы на M1 Max
        torch.set_num_threads(8)  # 8 производительных ядер на M1 Max
        
        # Включаем оптимизации Metal
        if hasattr(torch.backends, 'mps'):
            torch.backends.mps.enable_fallback = True
    
    elif device == "cuda":
        # Переносим модель на GPU
        model = model.to(device)
        print(f"   Модель перенесена на GPU")
    
    return model, device

def create_transcript_with_mlx_whisper(audio_file, model_size="base", video_path=None, cache_manager=None):
    """Создание транскрипта с помощью MLX Whisper (GPU-ускоренная версия для Apple Silicon)"""
    
    # Проверяем кэш, если менеджер кэша предоставлен
    if cache_manager and video_path:
        cached_transcript = cache_manager.get_cached_transcript(video_path)
        if cached_transcript:
            return cached_transcript
    
    try:
        # Импортируем mlx_whisper
        import mlx_whisper
        
        print(f"🚀 Загрузка MLX Whisper ({model_size}) с GPU ускорением...")
        print("💡 Используется оптимизированная версия для Apple Silicon")
        
        # Определяем путь к модели в зависимости от размера
        model_path = f"mlx-community/whisper-{model_size}-mlx"
        
        # Специальные пути для некоторых моделей
        model_paths = {
            "tiny": "mlx-community/whisper-tiny-mlx",
            "base": "mlx-community/whisper-base-mlx", 
            "small": "mlx-community/whisper-small-mlx",
            "medium": "mlx-community/whisper-medium-mlx",
            "large": "mlx-community/whisper-large-v3-mlx",
            "large-v3": "mlx-community/whisper-large-v3-mlx"
        }
        
        if model_size in model_paths:
            model_path = model_paths[model_size]
        
        print(f"📦 Загрузка модели: {model_path}")
        print("⚙️ Настройка для длинных сегментов и лучшего контекста...")
        
        # Транскрибируем с MLX Whisper с улучшенными параметрами
        start_time = time.time()
        result = mlx_whisper.transcribe(
            audio_file,
            path_or_hf_repo=model_path,
            language="ru",
            verbose=True
        )
        
        elapsed_time = time.time() - start_time
        print(f"✅ Транскрипция завершена за {elapsed_time:.1f} секунд")
        
        # Обрабатываем результаты
        segments = result.get("segments", [])
        
        # Объединяем короткие сегменты в более длинные для сохранения контекста
        merged_segments = []
        current_segment = None
        min_segment_duration = 10.0  # Минимальная длительность сегмента в секундах
        
        for segment in segments:
            # Пропускаем пустые или очень короткие сегменты
            if not segment.get("text", "").strip() or len(segment.get("text", "").strip()) < 3:
                continue
                
            if current_segment is None:
                current_segment = {
                    "text": segment["text"],
                    "start": segment["start"],
                    "end": segment["end"]
                }
            else:
                # Проверяем, нужно ли объединить с текущим сегментом
                duration = segment["end"] - current_segment["start"]
                if duration < min_segment_duration:
                    # Объединяем
                    current_segment["text"] += " " + segment["text"]
                    current_segment["end"] = segment["end"]
                else:
                    # Сохраняем текущий и начинаем новый
                    merged_segments.append({
                        "text": current_segment["text"].strip(),
                        "start": current_segment["start"],
                        "duration": current_segment["end"] - current_segment["start"]
                    })
                    current_segment = {
                        "text": segment["text"],
                        "start": segment["start"],
                        "end": segment["end"]
                    }
        
        # Добавляем последний сегмент
        if current_segment:
            merged_segments.append({
                "text": current_segment["text"].strip(),
                "start": current_segment["start"],
                "duration": current_segment["end"] - current_segment["start"]
            })
        
        print(f"📊 Объединено {len(segments)} коротких сегментов в {len(merged_segments)} длинных")
        
        # Фильтруем повторения и артефакты
        filtered_segments = filter_repetitive_segments(merged_segments)
        
        # Дополнительная фильтрация артефактов типа "Zhi Zhi Zhi"
        final_segments = []
        for seg in filtered_segments:
            text = seg["text"]
            # Проверяем на повторяющиеся короткие слова
            words = text.split()
            if len(words) > 3:
                # Проверяем, не состоит ли текст из повторений одного слова
                unique_words = set(words)
                if len(unique_words) > len(words) * 0.2:  # Хотя бы 20% уникальных слов
                    final_segments.append(seg)
            elif len(text.strip()) > 10:  # Или если текст достаточно длинный
                final_segments.append(seg)
        
        # Собираем полный текст
        full_text = " ".join([seg["text"] for seg in final_segments])
        
        print(f"🎯 Финальное количество сегментов: {len(final_segments)}")
        
        # Сохраняем в кэш
        if cache_manager and video_path:
            cache_manager.save_transcript_cache(video_path, final_segments, full_text)
        
        return final_segments, full_text
        
    except ImportError:
        print("⚠️ mlx_whisper не установлен, переключаюсь на стандартный Whisper...")
        return create_transcript_with_whisper(audio_file, model_size, video_path, cache_manager)
    except Exception as e:
        print(f"❌ Ошибка MLX Whisper: {str(e)}")
        print("⚠️ Переключаюсь на стандартный Whisper...")
        return create_transcript_with_whisper(audio_file, model_size, video_path, cache_manager)

def create_transcript_with_whisper(audio_file, model_size="base", video_path=None, cache_manager=None):
    """Создание транскрипта с помощью Whisper"""
    
    # Проверяем кэш, если менеджер кэша предоставлен
    if cache_manager and video_path:
        cached_transcript = cache_manager.get_cached_transcript(video_path)
        if cached_transcript:
            return cached_transcript
    
    try:
        # Загружаем модель с оптимизациями для устройства
        model, device = load_whisper_optimized(model_size)
        
        print("🎯 Транскрибирование аудио...")
        # Улучшенные параметры для лучшего качества распознавания русской речи
        # Добавляем fp16 для M1 Max для ускорения
        use_fp16 = device == "mps" or device == "cuda"
        
        result = model.transcribe(
            audio_file, 
            language="ru",
            initial_prompt="Это деловая встреча на русском языке с обсуждением технических вопросов, оборудования и планов.",
            temperature=0.0,  # Минимальная случайность для более точного распознавания
            beam_size=5,      # Больше вариантов для выбора лучшего
            best_of=5,        # Выбирать лучший из 5 вариантов
            compression_ratio_threshold=2.4,  # Фильтр от бессмыслицы
            logprob_threshold=-1.0,  # Фильтр по вероятности
            no_speech_threshold=0.6,   # Порог тишины
            fp16=use_fp16     # Используем fp16 для ускорения на GPU/MPS
        )
        
        # Возвращаем сегменты с временными метками
        segments = []
        for segment in result["segments"]:
            segments.append({
                "text": segment["text"],
                "start": segment["start"],
                "duration": segment["end"] - segment["start"]
            })
        
        # Фильтруем повторяющиеся сегменты
        original_count = len(segments)
        segments = filter_repetitive_segments(segments)
        
        if len(segments) < original_count:
            print(f"⚠️  Отфильтровано {original_count - len(segments)} повторяющихся сегментов")
            # Пересобираем полный текст из отфильтрованных сегментов
            full_text = " ".join([s['text'] for s in segments if s.get('text')])
        else:
            full_text = result["text"]
        
        # Сохраняем в кэш, если менеджер кэша предоставлен
        if cache_manager and video_path:
            cache_manager.save_transcript_cache(video_path, segments, full_text)
        
        return segments, full_text
    except Exception as e:
        print(f"Ошибка при создании транскрипта с Whisper: {e}")
        return None, None

def create_clean_transcript_with_whisper(audio_file, model_size="base", video_path=None, cache_manager=None, chunk_duration=180):
    """Создание чистого транскрипта крупными блоками без временных меток"""
    
    # Проверяем кэш, если менеджер кэша предоставлен
    if cache_manager and video_path:
        cached_transcript = cache_manager.get_cached_transcript(video_path)
        if cached_transcript and isinstance(cached_transcript, tuple) and len(cached_transcript) > 1:
            # Если есть кэшированный транскрипт, возвращаем только полный текст
            return cached_transcript[1]
    
    try:
        # Загружаем модель с оптимизациями для устройства
        model, device = load_whisper_optimized(model_size)
        
        print(f"🎧 Транскрибирование аудио блоками по {chunk_duration} секунд...")
        
        # Загружаем аудио и получаем длительность
        import librosa
        audio_data, sr = librosa.load(audio_file, sr=16000)
        duration = len(audio_data) / sr
        
        full_transcript_parts = []
        
        # Обрабатываем аудио блоками
        for start_time in range(0, int(duration), chunk_duration):
            end_time = min(start_time + chunk_duration, duration)
            
            # Извлекаем часть аудио
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            chunk_audio = audio_data[start_sample:end_sample]
            
            # Создаем временный файл для блока
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                import soundfile as sf
                sf.write(tmp_file.name, chunk_audio, sr)
                
                # Транскрибируем блок
                print(f"  ⏱️  Обработка блока {int(start_time/60)}:{start_time%60:02d} - {int(end_time/60)}:{end_time%60:02d}")
                result = model.transcribe(
                    tmp_file.name,
                    language="ru",
                    initial_prompt="Транскрипция технической встречи на русском языке. "
                )
                
                # Добавляем текст блока
                if result['text'].strip():
                    full_transcript_parts.append(result['text'].strip())
                
                # Удаляем временный файл
                os.unlink(tmp_file.name)
        
        # Объединяем все части
        full_transcript = "\n\n".join(full_transcript_parts)
        
        print(f"✅ Транскрипт создан: {len(full_transcript)} символов")
        
        # Кэшируем результат
        if cache_manager and video_path:
            # Сохраняем в формате совместимом с обычным транскриптом
            cache_manager.save_transcript_cache(video_path, [], full_transcript)
        
        return full_transcript
        
    except Exception as e:
        print(f"❌ Ошибка при создании чистого транскрипта: {e}")
        import traceback
        traceback.print_exc()
        return None

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

def filter_repetitive_segments(segments, max_repetitions=5):
    """Фильтрует повторяющиеся сегменты из транскрипта Whisper"""
    if not segments:
        return segments
    
    filtered_segments = []
    repetition_count = 0
    last_text = None
    
    for segment in segments:
        segment_text = segment.get('text', '').strip()
        
        # Проверяем на повторение
        if segment_text == last_text and len(segment_text) < 20:  # Короткие повторяющиеся фразы
            repetition_count += 1
            if repetition_count > max_repetitions:
                continue  # Пропускаем повторяющийся сегмент
        else:
            repetition_count = 0
            last_text = segment_text
        
        filtered_segments.append(segment)
    
    # Проверяем, не удалили ли мы слишком много
    if len(filtered_segments) < len(segments) * 0.5:
        print(f"⚠️  Обнаружено зацикливание Whisper: {len(segments) - len(filtered_segments)} повторяющихся сегментов удалено")
    
    return filtered_segments

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
                    minutes = int(timestamp // 60)
                    seconds = int(timestamp % 60)
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
                    minutes = int(timestamp // 60)
                    seconds = int(timestamp % 60)
                    f.write(f"#### {minutes}:{seconds:02d}\n\n")
                    
                    image_base64 = get_image_base64(image_path)
                    if image_base64:
                        f.write(f"![Скриншот]({image_base64})\n\n")
                    
                    if description:
                        f.write(f"{description}\n\n")
                    
                    f.write("---\n\n")
        
        f.write("\n\n## Полный транскрипт\n\n")
        f.write(full_transcript)
        
        # Добавляем транскрипт с временными метками только если есть сегменты
        if transcript_segments and len(transcript_segments) > 0:
            f.write("\n\n## Транскрипт с временными метками\n\n")
            for segment in transcript_segments:
                start_time = segment['start']
                minutes = int(start_time // 60)
                seconds = int(start_time % 60)
                f.write(f"**{minutes}:{seconds:02d}** - {segment['text']}\n\n")
        elif not transcript_segments:
            f.write("\n\n---\n*Транскрипт создан в чистом режиме без временных меток*")
    
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
        print("  python updated_video_analyzer.py <путь_к_видео> [опции]")
        print("\nОпции:")
        print("  --whisper-model MODEL      Модель Whisper (tiny, base, small, medium, large)")
        print("  --screenshot-interval N    Интервал между скриншотами в секундах (по умолчанию: 30)")
        print("  --no-screenshots          Не извлекать скриншоты")
        print("  --smart-screenshots       Использовать ИИ для умного извлечения скриншотов")
        print("  --screenshot-mode MODE    Режим скриншотов: 'periodic', 'smart', 'transcript', 'both' (по умолчанию: transcript)")
        print("  --clean-transcript [N]     Создать чистый транскрипт блоками по N секунд (по умолчанию: 120)")
        print("  --basic-mode              Использовать базовый режим (без агентных улучшений)")
        print("  --output DIR              Директория для результатов (по умолчанию: results)")
        print("  --clear-cache             Очистить кэш для этого видео")
        print("  --force-refresh           Принудительно обновить все данные (игнорировать кэш)")
        print("  --cache-status            Показать только статус кэша и выйти")
        print("  --test-first-10min        Обработать только первые 10 минут для тестирования")
        print("\nПо умолчанию:")
        print("  🤖 АГЕНТНЫЙ РЕЖИМ - детальный анализ с сохранением технических деталей")
        print("\nПримеры:")
        print("  python updated_video_analyzer.py video.mp4 --test-first-10min")
        print("  python updated_video_analyzer.py video.mp4 --clean-transcript 180 --no-screenshots")
        print("  python updated_video_analyzer.py video.mp4 --clean-transcript --whisper-model large")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    # Парсим аргументы
    whisper_model = "base"
    screenshot_interval = 30
    extract_screenshots_flag = True
    screenshot_mode = "transcript"  # По умолчанию используем анализ транскрипта
    clean_transcript = False  # Флаг для чистого транскрипта
    chunk_duration = 120  # Длительность блоков в секундах
    basic_mode = False  # По умолчанию используем агентный режим!
    output_dir = "results"
    clear_cache = False
    force_refresh = False
    cache_status_only = False
    test_first_10min = False
    
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
        elif sys.argv[i] == "--clean-transcript":
            clean_transcript = True
            # Проверяем есть ли следующий аргумент и является ли он числом
            if i + 1 < len(sys.argv) and sys.argv[i + 1].isdigit():
                chunk_duration = int(sys.argv[i + 1])
                i += 2
            else:
                i += 1
        elif sys.argv[i] == "--basic-mode":
            basic_mode = True
            i += 1
        elif sys.argv[i] == "--output" and i + 1 < len(sys.argv):
            output_dir = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--clear-cache":
            clear_cache = True
            i += 1
        elif sys.argv[i] == "--force-refresh":
            force_refresh = True
            i += 1
        elif sys.argv[i] == "--cache-status":
            cache_status_only = True
            i += 1
        elif sys.argv[i] == "--test-first-10min":
            test_first_10min = True
            i += 1
        else:
            i += 1
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ API ключ OpenAI не найден!")
        print("Для работы агентной системы добавьте OPENAI_API_KEY в .env файл")
        print("Переключаемся на базовый режим...")
        basic_mode = True
        if screenshot_mode in ["smart", "transcript"]:
            screenshot_mode = "periodic"  # Переключаемся на периодический режим без API
    
    if not os.path.exists(video_path):
        print(f"Ошибка: Видеофайл {video_path} не найден")
        sys.exit(1)
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Режим работы
    if basic_mode:
        print(f"📊 БАЗОВЫЙ режим анализа видео: {video_name}")
        chronological_mode = False
    else:
        print(f"🤖 АГЕНТНЫЙ режим анализа видео: {video_name}")
        print("✨ Включен детальный анализ с сохранением технических деталей")
        chronological_mode = True  # Агентный режим всегда использует хронологический процессор
    
    print(f"🎬 Режим скриншотов: {screenshot_mode}")
    if clean_transcript:
        print(f"📝 ЧИСТЫЙ транскрипт: блоки по {chunk_duration} секунд (БЕЗ временных меток)")
    if test_first_10min:
        print("⏱️  ТЕСТОВЫЙ режим: обрабатываем только первые 10 минут")

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
        # Проверяем, на Apple Silicon ли мы
        import platform
        use_mlx = platform.system() == "Darwin" and platform.machine() == "arm64"
        
        if clean_transcript:
            print(f"📝 Создание ЧИСТОГО транскрипта блоками по {chunk_duration} секунд...")
            full_transcript = create_clean_transcript_with_whisper(
                audio_path, whisper_model, video_path=video_path, 
                cache_manager=cache_manager, chunk_duration=chunk_duration
            )
            transcript_segments = None  # Нет сегментов в чистом режиме
            if not full_transcript:
                print("Не удалось создать чистый транскрипт. Завершение работы.")
                sys.exit(1)
        else:
            # Используем MLX Whisper на Apple Silicon
            if use_mlx:
                print("🍎 Обнаружен Apple Silicon - используем MLX Whisper с GPU ускорением")
                transcript_segments, full_transcript = create_transcript_with_mlx_whisper(
                    audio_path, whisper_model, video_path=video_path, cache_manager=cache_manager
                )
            else:
                transcript_segments, full_transcript = create_transcript_with_whisper(
                    audio_path, whisper_model, video_path=video_path, cache_manager=cache_manager
                )
            
            if not full_transcript:
                print("Не удалось создать транскрипт. Завершение работы.")
                sys.exit(1)
        
        # Ограничиваем первыми 10 минутами для тестирования
        if test_first_10min and transcript_segments:
            print("⏱️  Ограничиваем анализ первыми 10 минутами...")
            original_segments_count = len(transcript_segments)
            transcript_segments = [s for s in transcript_segments if s['start'] <= 600]  # 10 минут = 600 сек
            
            # Обрезаем полный транскрипт тоже
            if transcript_segments:
                last_segment_end = transcript_segments[-1]['start'] + transcript_segments[-1].get('duration', 0)
                # Находим приблизительную позицию в полном тексте
                chars_per_second = len(full_transcript) / (last_segment_end if last_segment_end > 0 else 1)
                cut_position = int(chars_per_second * 600)
                full_transcript = full_transcript[:cut_position] + "\n\n[Анализ ограничен первыми 10 минутами для тестирования]"
            
            print(f"📊 Сегментов до ограничения: {original_segments_count}")
            print(f"📊 Сегментов после ограничения: {len(transcript_segments)}")
        elif test_first_10min and clean_transcript:
            print("⏱️  В режиме чистого транскрипта ограничение на 10 минут не применяется")
            print("💡 Используйте --chunk-duration для настройки размера блоков")
        
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
        
        # 🤖 АГЕНТНЫЙ ХРОНОЛОГИЧЕСКИЙ АНАЛИЗ (по умолчанию в агентном режиме)
        if chronological_mode and CHRONOLOGICAL_AVAILABLE and api_key:
            print("\n🤖 Запускаем АГЕНТНЫЙ хронологический анализ...")
            print("✨ Детальное восстановление смысла с сохранением технических деталей")
            
            # Создаем расширенный контекст видео для агентной системы
            video_context = {
                'meeting_type': 'technical_discussion',  # Предполагаем техническую встречу
                'main_topics': ['technical_details', 'equipment', 'data_processing'],
                'visual_content_probability': 0.8,  # Высокая вероятность технического контента
                'use_agent_mode': True,  # Флаг для агентного режима
                'preserve_details': True,  # Сохранять все детали
                'test_mode': test_first_10min  # Передаем информацию о тестовом режиме
            }
            
            # Преобразуем скриншоты в нужный формат с ограничением для тестового режима
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
                
                # Ограничиваем скриншоты первыми 10 минутами для тестового режима
                if test_first_10min:
                    screenshots_formatted = [s for s in screenshots_formatted if s[1] <= 600]
                    print(f"📸 Ограничено скриншотов для тестирования: {len(screenshots_formatted)}")
            
            # Запускаем агентную хронологическую обработку
            processor = ChronologicalTranscriptProcessor(api_key)
            chronological_data = processor.process_video_meeting(
                transcript_segments, screenshots_formatted, video_context
            )
            
            # Сохраняем результаты агентного анализа
            chron_md, chron_json = save_chronological_results(video_name, chronological_data, output_dir)
            
            print(f"\n🎉 АГЕНТНЫЙ анализ завершен!")
            print(f"  🎬 Детальный отчет: {chron_md}")
            print(f"  📋 Технические данные: {chron_json}")
            
            # Показываем статистику агентных улучшений
            timeline = chronological_data.get('timeline', [])
            corrected_count = sum(1 for event in timeline 
                                if event.type == 'transcript' and 
                                event.content.get('corrected_text') != event.content.get('text'))
            technical_details_count = sum(1 for event in timeline 
                                        if event.type == 'transcript' and 
                                        event.content.get('technical_details'))
            screen_references_count = sum(1 for event in timeline 
                                        if event.type == 'transcript' and 
                                        event.content.get('screen_references'))
            
            print(f"\n📊 СТАТИСТИКА АГЕНТНЫХ УЛУЧШЕНИЙ:")
            print(f"  ✏️  Whisper коррекций: {corrected_count}")
            print(f"  ⚙️  Технических деталей: {technical_details_count}")
            print(f"  📺 Ссылок на экран: {screen_references_count}")
        
        elif chronological_mode and not CHRONOLOGICAL_AVAILABLE:
            print("⚠️  Агентный режим недоступен. Проверьте chronological_transcript_processor.py")
            print("Переключаемся на базовый режим...")
            chronological_mode = False
        
        elif chronological_mode and not api_key:
            print("⚠️  Для агентного анализа нужен API ключ OpenAI")
            print("Переключаемся на базовый режим...")
            chronological_mode = False
        
        # Определяем какие файлы сохранять
        if chronological_mode and 'chron_md' in locals():
            # Агентный режим - используем детальные результаты
            md_file = chron_md
            json_file = chron_json
            mode_description = "🤖 АГЕНТНЫЙ"
        else:
            # Базовый режим - обычные результаты
            md_file, json_file = save_results(video_name, transcript_segments, full_transcript, analysis, screenshots, output_dir)
            mode_description = "📊 БАЗОВЫЙ"
        
        print(f"\n🎉 {mode_description} анализ завершен!")
        print(f"Результаты сохранены:")
        print(f"  📄 Отчет: {md_file}")
        print(f"  📋 Данные: {json_file}")
        
        if test_first_10min:
            print(f"  ⏱️  Проанализировано: первые 10 минут видео")
        
        if screenshots:
            screenshot_count = len(screenshots)
            if test_first_10min:
                screenshot_count = len([s for s in screenshots if s[1] <= 600])
            
            print(f"  📸 Скриншоты: {screenshot_count} файлов")
            if screenshot_mode == "transcript":
                print(f"     🧠 На основе анализа транскрипта найдено {screenshot_count} ключевых моментов")
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
            'clean_transcript_mode': clean_transcript,
            'chunk_duration': chunk_duration if clean_transcript else None,
            'transcript_length': len(full_transcript),
            'segments_count': len(transcript_segments) if transcript_segments else 0,
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
