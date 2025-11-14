#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Умный экстрактор скриншотов на основе анализа транскрипта
Анализирует текст разговора и определяет ключевые моменты для скриншотов
"""

import os
import json
import cv2
import numpy as np
from openai import OpenAI
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import logging
import re

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TranscriptMoment:
    """Важный момент в транскрипте для скриншота"""
    timestamp: float
    reason: str
    importance: float
    keywords: List[str]
    context: str
    screenshot_type: str  # "demo", "slide", "code", "diagram", "ui"

class SmartTranscriptExtractor:
    """Умный экстрактор на основе анализа текста"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
        
        # Ключевые слова для разных типов демонстраций
        self.demo_keywords = {
            "screen_share": ["покажу", "смотрите", "видите", "вот здесь", "на экране", "давайте посмотрим"],
            "code": ["код", "функция", "метод", "переменная", "алгоритм", "класс", "скрипт", "файл"],
            "presentation": ["слайд", "презентация", "следующий", "видим", "показано", "диаграмма"],
            "demo": ["демо", "демонстрация", "пример", "работает", "запустим", "посмотрим как"],
            "diagram": ["схема", "диаграмма", "структура", "архитектура", "поток", "процесс"],
            "ui": ["интерфейс", "кнопка", "меню", "форма", "страница", "окно", "элемент"],
            "document": ["документ", "файл", "текст", "содержание", "пункт", "раздел"],
            "discussion": ["обсуждаем", "мнение", "думаю", "считаю", "предлагаю", "вопрос"]
        }
        
        # Триггерные фразы для скриншотов
        self.trigger_phrases = [
            r"вот\s+(здесь|тут|это)",
            r"смотрите\s+(на|сюда)",
            r"видите\s+(это|здесь)",
            r"покажу\s+(вам|как)",
            r"давайте\s+посмотрим",
            r"например\s+(здесь|это)",
            r"переходим\s+к",
            r"следующий\s+(слайд|пункт|раздел)",
            r"открываю\s+(файл|документ|код)",
            r"запускаю\s+(программу|скрипт)",
            r"вызываю\s+(функцию|метод)",
            r"нажимаю\s+(кнопку|ссылку)"
        ]
    
    def extract_screenshots(self, video_path: str, output_dir: str, 
                           transcript_segments: List[Dict]) -> List[Tuple]:
        """Основной метод извлечения скриншотов"""
        
        logger.info(f"🧠 Анализируем транскрипт для поиска ключевых моментов...")
        
        # Создаем директорию для скриншотов
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Шаг 1: Анализируем весь транскрипт и находим ключевые моменты
        key_moments = self.analyze_transcript_for_moments(transcript_segments)
        
        logger.info(f"📍 Найдено {len(key_moments)} ключевых моментов")
        
        # Шаг 2: Группируем близкие моменты и удаляем дубликаты
        optimized_moments = self.optimize_moments(key_moments)
        
        logger.info(f"✨ После оптимизации: {len(optimized_moments)} моментов")
        
        # Шаг 3: Извлекаем скриншоты только в найденные моменты
        screenshots = self.extract_targeted_screenshots(
            video_path, output_dir, optimized_moments, transcript_segments
        )
        
        logger.info(f"📸 Извлечено {len(screenshots)} скриншотов")
        
        return screenshots
    
    def analyze_transcript_for_moments(self, transcript_segments: List[Dict]) -> List[TranscriptMoment]:
        """Анализирует транскрипт и находит ключевые моменты"""
        
        # Сначала пробуем проанализировать весь транскрипт одним запросом
        full_text = self.prepare_full_transcript_text(transcript_segments)
        
        # Если транскрипт слишком большой (>15000 символов), разбиваем на крупные блоки
        if len(full_text) > 15000:
            logger.info(f"📄 Транскрипт большой ({len(full_text)} символов), разбиваем на блоки по 10 минут")
            text_blocks = self.group_transcript_blocks(transcript_segments, block_size=600)  # 10 минут
        else:
            logger.info(f"📄 Анализируем весь транскрипт целиком ({len(full_text)} символов)")
            # Создаем один блок из всего транскрипта
            text_blocks = [{
                'start_time': transcript_segments[0]['start'] if transcript_segments else 0,
                'end_time': transcript_segments[-1]['start'] + transcript_segments[-1]['duration'] if transcript_segments else 0,
                'text': full_text,
                'segments': transcript_segments
            }]
        
        all_moments = []
        
        logger.info(f"🧠 Обрабатываем {len(text_blocks)} блок(ов) через OpenAI API...")
        
        for i, block in enumerate(text_blocks, 1):
            logger.info(f"📝 Анализ блока {i}/{len(text_blocks)} ({block['start_time']:.1f}с - {block['end_time']:.1f}с)")
            
            # Анализируем каждый блок с помощью AI
            moments = self.analyze_text_block(block)
            all_moments.extend(moments)
            
            # Добавляем также эвристические моменты
            heuristic_moments = self.find_heuristic_moments(block)
            all_moments.extend(heuristic_moments)
        
        logger.info(f"✅ Анализ завершен. Найдено {len(all_moments)} потенциальных моментов")
        return all_moments

    def prepare_full_transcript_text(self, transcript_segments: List[Dict]) -> str:
        """Подготавливает полный текст транскрипта с таймкодами"""
        text_parts = []
        
        for segment in transcript_segments:
            timestamp = f"[{segment['start']:.1f}с]"
            text_parts.append(f"{timestamp} {segment['text']}")
        
        return '\n'.join(text_parts)
    
    def group_transcript_blocks(self, transcript_segments: List[Dict], 
                               block_size: float = 45) -> List[Dict]:
        """Группирует транскрипт по временным блокам"""
        
        blocks = []
        current_block = {
            'start_time': 0,
            'end_time': 0,
            'text': '',
            'segments': []
        }
        
        for segment in transcript_segments:
            segment_start = segment['start']
            
            # Начинаем новый блок если прошло больше block_size секунд
            if segment_start > current_block['start_time'] + block_size:
                if current_block['text']:
                    blocks.append(current_block)
                
                current_block = {
                    'start_time': segment_start,
                    'end_time': segment_start + segment['duration'],
                    'text': segment['text'],
                    'segments': [segment]
                }
            else:
                # Добавляем к текущему блоку
                current_block['end_time'] = segment_start + segment['duration']
                current_block['text'] += ' ' + segment['text']
                current_block['segments'].append(segment)
        
        # Добавляем последний блок
        if current_block['text']:
            blocks.append(current_block)
        
        return blocks
    
    def analyze_text_block(self, block: Dict) -> List[TranscriptMoment]:
        """Анализирует текстовый блок с помощью AI"""
        
        # Определяем, это полный транскрипт или блок
        is_full_transcript = len(block['text']) > 10000
        duration_minutes = (block['end_time'] - block['start_time']) / 60
        
        # Ограничиваем текст для модели (8000 символов)
        text_for_analysis = block['text'][:8000] + ("..." if len(block['text']) > 8000 else "")
        
        prompt = f"""Проанализируй {"весь транскрипт" if is_full_transcript else "фрагмент транскрипта"} встречи и найди ключевые моменты для создания скриншотов.

{"Полная встреча" if is_full_transcript else "Временной отрезок"}: {block['start_time']:.1f}с - {block['end_time']:.1f}с ({duration_minutes:.1f} мин)

Текст с таймкодами:
{text_for_analysis}

ВАЖНО: Текст содержит таймкоды в формате [123.4с]. Используй эти ТОЧНЫЕ таймкоды для определения времени.

Найди моменты где участники:
1. Показывают экран, код, документ, презентацию ("вот здесь", "смотрите на экран", "покажу")
2. Демонстрируют работу программы/интерфейса ("запускаю", "нажимаю", "открываю")
3. Объясняют диаграмму, схему ("схема показывает", "на диаграмме видно")
4. Переходят к новой теме/слайду ("следующий слайд", "перейдем к", "теперь рассмотрим")
5. Показывают результаты, примеры ("результат такой", "например вот", "получается")

Для каждого момента укажи:
- timestamp: ТОЧНОЕ время из таймкода в тексте (например, если видишь [245.7с], используй 245.7)
- reason: что именно показывается (1-2 предложения)
- importance: важность 0.1-1.0 (1.0 = обязательный скриншот, 0.7+ = желательный, 0.5- = опциональный)
- keywords: ключевые слова из речи (2-5 слов)
- screenshot_type: тип (demo/code/presentation/diagram/ui/document/discussion)

Отбирай только ЗНАЧИМЫЕ моменты. Лучше {"3-8 важных" if is_full_transcript else "1-3 важных"}, чем много случайных.

JSON ответ:
{{
  "moments": [
    {{
      "timestamp": 123.5,
      "reason": "Показывает код функции обработки данных",
      "importance": 0.9,
      "keywords": ["код", "функция", "обработка"],
      "screenshot_type": "code"
    }}
  ]
}}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            # Безопасный парсинг JSON
            try:
                result = json.loads(response.choices[0].message.content)
            except json.JSONDecodeError as e:
                logger.error(f"Ошибка JSON парсинга в анализе транскрипта: {e}")
                return []
            moments = []
            
            for moment_data in result.get('moments', []):
                moment = TranscriptMoment(
                    timestamp=moment_data['timestamp'],
                    reason=moment_data['reason'],
                    importance=moment_data['importance'],
                    keywords=moment_data['keywords'],
                    context=block['text'][:200],
                    screenshot_type=moment_data['screenshot_type']
                )
                moments.append(moment)
                
            return moments
            
        except Exception as e:
            logger.error(f"Ошибка при анализе блока: {e}")
            return []
    
    def find_heuristic_moments(self, block: Dict) -> List[TranscriptMoment]:
        """Находит моменты с помощью эвристических правил"""
        
        text = block['text'].lower()
        moments = []
        
        # Ищем триггерные фразы
        for pattern in self.trigger_phrases:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Приблизительно определяем время
                char_position = match.start()
                time_ratio = char_position / len(text)
                timestamp = block['start_time'] + (block['end_time'] - block['start_time']) * time_ratio
                
                moment = TranscriptMoment(
                    timestamp=timestamp,
                    reason=f"Триггерная фраза: '{match.group()}'",
                    importance=0.7,
                    keywords=[match.group()],
                    context=text[max(0, char_position-50):char_position+50],
                    screenshot_type="demo"
                )
                moments.append(moment)
        
        # Ищем ключевые слова по категориям
        for category, keywords in self.demo_keywords.items():
            if category == "discussion":  # Пропускаем обычные обсуждения
                continue
                
            for keyword in keywords:
                if keyword in text:
                    # Находим позицию ключевого слова
                    keyword_pos = text.find(keyword)
                    time_ratio = keyword_pos / len(text)
                    timestamp = block['start_time'] + (block['end_time'] - block['start_time']) * time_ratio
                    
                    moment = TranscriptMoment(
                        timestamp=timestamp,
                        reason=f"Ключевое слово '{keyword}' указывает на {category}",
                        importance=0.6,
                        keywords=[keyword],
                        context=text[max(0, keyword_pos-30):keyword_pos+30],
                        screenshot_type=category
                    )
                    moments.append(moment)
                    break  # Один момент на категорию в блоке
        
        return moments
    
    def optimize_moments(self, moments: List[TranscriptMoment]) -> List[TranscriptMoment]:
        """Оптимизирует список моментов - убирает дубликаты и группирует близкие"""
        
        if not moments:
            return []
        
        # Сортируем по времени
        moments.sort(key=lambda m: m.timestamp)
        
        optimized = []
        current_group = [moments[0]]
        
        for i in range(1, len(moments)):
            current_moment = moments[i]
            
            # Если момент близко к предыдущим (в пределах 10 секунд)
            if current_moment.timestamp - current_group[-1].timestamp < 10:
                current_group.append(current_moment)
            else:
                # Завершаем текущую группу и берем лучший момент
                best_moment = max(current_group, key=lambda m: m.importance)
                optimized.append(best_moment)
                
                # Начинаем новую группу
                current_group = [current_moment]
        
        # Добавляем последнюю группу
        if current_group:
            best_moment = max(current_group, key=lambda m: m.importance)
            optimized.append(best_moment)
        
        # Фильтруем по важности (оставляем только важные моменты)
        high_importance = [m for m in optimized if m.importance >= 0.7]
        
        # Если важных моментов мало, добавляем средние
        if len(high_importance) < 3:
            medium_importance = [m for m in optimized if 0.5 <= m.importance < 0.7]
            high_importance.extend(medium_importance[:5])  # Максимум 5 средних
        
        # Ограничиваем общее количество
        final_moments = sorted(high_importance, key=lambda m: m.importance, reverse=True)[:15]
        final_moments.sort(key=lambda m: m.timestamp)  # Сортируем обратно по времени
        
        return final_moments
    
    def extract_targeted_screenshots(self, video_path: str, output_dir: str,
                                   moments: List[TranscriptMoment],
                                   transcript_segments: List[Dict]) -> List[Tuple]:
        """Извлекает скриншоты только в найденные моменты"""
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        screenshots = []
        
        for i, moment in enumerate(moments):
            logger.info(f"📸 {i+1}/{len(moments)}: {moment.timestamp:.1f}с - {moment.reason}")
            
            # Устанавливаем позицию видео
            frame_number = int(moment.timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Не удалось получить кадр в {moment.timestamp:.1f}с")
                continue
            
            # Выбираем лучший кадр в окрестности (±2 секунды)
            best_frame, best_timestamp = self.find_best_frame_nearby(
                cap, moment.timestamp, fps, window_seconds=2
            )
            
            # Сохраняем скриншот
            filename = f"screenshot_{i+1:03d}_{best_timestamp:.1f}s.png"
            screenshot_path = os.path.join(output_dir, filename)
            
            cv2.imwrite(screenshot_path, best_frame)
            
            # Создаем описание
            description = self.create_screenshot_description(
                moment, best_timestamp, transcript_segments
            )
            
            screenshot_info = {
                'path': screenshot_path,
                'timestamp': best_timestamp,
                'description': description,
                'moment': moment,
                'type': moment.screenshot_type
            }
            
            screenshots.append((screenshot_path, best_timestamp, description, moment.reason))
        
        cap.release()
        return screenshots
    
    def find_best_frame_nearby(self, cap: cv2.VideoCapture, target_time: float,
                              fps: float, window_seconds: float = 2) -> Tuple[np.ndarray, float]:
        """Находит лучший кадр в окрестности целевого времени"""
        
        # Собираем кадры в окне
        start_time = max(0, target_time - window_seconds)
        end_time = target_time + window_seconds
        
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        best_frame = None
        best_time = target_time
        best_score = -1
        
        for frame_num in range(start_frame, end_frame + 1, int(fps * 0.5)):  # Каждые 0.5 сек
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Оцениваем качество кадра
            score = self.evaluate_frame_quality(frame)
            
            if score > best_score:
                best_score = score
                best_frame = frame.copy()
                best_time = frame_num / fps
        
        return best_frame if best_frame is not None else frame, best_time
    
    def evaluate_frame_quality(self, frame: np.ndarray) -> float:
        """Оценивает качество кадра"""
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Оценка резкости (вариация Лапласиана)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Оценка контраста
        contrast = gray.std()
        
        # Оценка информативности (количество краев)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Комбинированная оценка
        quality_score = (
            laplacian_var * 0.4 +
            contrast * 0.3 +
            edge_density * 100 * 0.3
        )
        
        return quality_score
    
    def create_screenshot_description(self, moment: TranscriptMoment,
                                    timestamp: float,
                                    transcript_segments: List[Dict]) -> str:
        """Создает описание скриншота с использованием GPT-4V"""

        # Находим ближайшие сегменты транскрипта
        context_text = self.get_transcript_context(transcript_segments, timestamp, window=15)

        # Базовое описание без GPT-4V (используется как fallback)
        basic_description = f"""
**Время:** {timestamp:.1f}с
**Причина:** {moment.reason}
**Тип:** {moment.screenshot_type}
**Ключевые слова:** {', '.join(moment.keywords)}
**Контекст:** {context_text}
        """.strip()

        return basic_description
    
    def get_transcript_context(self, transcript_segments: List[Dict],
                             timestamp: float, window: float = 15) -> str:
        """Получает контекст из транскрипта вокруг заданного времени"""
        
        context_segments = []
        
        for segment in transcript_segments:
            segment_start = segment['start']
            segment_end = segment_start + segment['duration']
            
            # Проверяем пересечение с окном
            if (segment_start <= timestamp + window and 
                segment_end >= timestamp - window):
                context_segments.append(segment['text'])
        
        return ' '.join(context_segments)


def create_smart_transcript_extractor(video_path: str, output_dir: str,
                                    transcript_segments: List[Dict],
                                    api_key: str) -> List[Tuple]:
    """Интеграционная функция для умного экстрактора"""
    
    extractor = SmartTranscriptExtractor(api_key)
    return extractor.extract_screenshots(video_path, output_dir, transcript_segments) 