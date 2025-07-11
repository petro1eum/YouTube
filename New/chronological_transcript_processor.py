#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Процессор для создания интегрированного хронологического транскрипта
с определением участников и коррекцией текста
"""

import os
import json
import re
import base64
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from openai import OpenAI
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Speaker:
    """Информация об участнике"""
    id: str
    name: Optional[str] = None
    role: Optional[str] = None
    characteristics: List[str] = None
    voice_segments: List[Tuple[float, float]] = None

@dataclass
class TranscriptSegment:
    """Сегмент транскрипта с дополнительной информацией"""
    start: float
    end: float
    text: str
    speaker_id: Optional[str] = None
    corrected_text: Optional[str] = None
    context_notes: Optional[str] = None
    related_screenshot: Optional[str] = None
    confidence: float = 1.0

@dataclass
class TimelineEvent:
    """Событие на временной линии"""
    timestamp: float
    type: str  # 'transcript', 'screenshot', 'topic_change', 'speaker_change'
    content: Dict
    importance: float = 0.5

class ChronologicalTranscriptProcessor:
    """Обработчик для создания интегрированного хронологического транскрипта"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
        self.speakers = {}
        self.timeline_events = []
        self.topics = []
        self.context_buffer = []
        
    def process_video_meeting(self, transcript_segments: List[Dict], 
                            screenshots: List[Tuple],
                            video_context: Dict) -> Dict:
        """Основной метод обработки видео встречи"""
        
        logger.info("Начинаем обработку хронологического транскрипта...")
        
        # 1. Анализируем участников
        speakers = self.identify_speakers(transcript_segments)
        logger.info(f"Обнаружено участников: {len(speakers)}")
        
        # 2. Создаем временную линию событий
        timeline = self.create_timeline(transcript_segments, screenshots)
        
        # 3. Корректируем транскрипт с учетом контекста
        corrected_timeline = self.correct_transcript_with_context(
            timeline, speakers, video_context
        )
        
        # 4. Группируем по темам и участникам
        structured_content = self.structure_content(corrected_timeline, speakers)
        
        # 5. Генерируем финальный отчет
        report = self.generate_chronological_report(structured_content, speakers)
        
        return {
            'speakers': speakers,
            'timeline': corrected_timeline,
            'structured_content': structured_content,
            'report': report
        }
    
    def identify_speakers(self, transcript_segments: List[Dict]) -> Dict[str, Speaker]:
        """Определяет участников встречи на основе транскрипта"""
        
        # Собираем первые 10 минут для анализа
        early_segments = []
        for segment in transcript_segments:
            if segment['start'] < 600:  # 10 минут
                early_segments.append(segment)
        
        early_text = '\n'.join([s['text'] for s in early_segments])
        
        prompt = f"""Проанализируй транскрипт встречи и определи участников.

Транскрипт первых минут:
{early_text[:3000]}

Определи:
1. Количество участников (speakers)
2. Их вероятные имена (если упоминаются)
3. Их роли (ведущий, докладчик, участник и т.д.)
4. Характерные фразы или стиль речи каждого
5. Паттерны смены говорящих

Ответь в JSON формате:
{{
    "speakers": [
        {{
            "id": "speaker1",
            "probable_name": "имя если известно или null",
            "role": "роль",
            "characteristics": ["характерные особенности речи"],
            "speech_patterns": ["примеры фраз"]
        }}
    ],
    "speaker_change_indicators": ["индикаторы смены говорящего"],
    "meeting_format": "формат встречи (презентация/дискуссия/интервью)"
}}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Создаем объекты Speaker
            speakers = {}
            for speaker_data in result.get('speakers', []):
                speaker = Speaker(
                    id=speaker_data['id'],
                    name=speaker_data.get('probable_name'),
                    role=speaker_data.get('role', 'участник'),
                    characteristics=speaker_data.get('characteristics', []),
                    voice_segments=[]
                )
                speakers[speaker.id] = speaker
            
            # Сохраняем индикаторы смены говорящих
            self.speaker_change_indicators = result.get('speaker_change_indicators', [])
            self.meeting_format = result.get('meeting_format', 'discussion')
            
            # Теперь проходим по всему транскрипту и назначаем говорящих
            self.assign_speakers_to_segments(transcript_segments, speakers)
            
            return speakers
            
        except Exception as e:
            logger.error(f"Ошибка при определении участников: {e}")
            # Возвращаем одного дефолтного участника
            default_speaker = Speaker(id="speaker1", name="Участник", role="speaker")
            return {"speaker1": default_speaker}
    
    def assign_speakers_to_segments(self, transcript_segments: List[Dict], 
                                  speakers: Dict[str, Speaker]):
        """Назначает говорящих для каждого сегмента транскрипта"""
        
        current_speaker = "speaker1"
        context_window = []
        
        for i, segment in enumerate(transcript_segments):
            # Собираем контекст
            context = self.get_segment_context(transcript_segments, i, window_size=5)
            
            # Определяем говорящего для сегмента
            speaker_id = self.detect_speaker_for_segment(
                segment, context, speakers, current_speaker
            )
            
            segment['speaker_id'] = speaker_id
            
            # Обновляем информацию о сегментах говорящего
            if speaker_id in speakers:
                speakers[speaker_id].voice_segments.append(
                    (segment['start'], segment.get('end', segment['start'] + segment['duration']))
                )
            
            current_speaker = speaker_id
    
    def detect_speaker_for_segment(self, segment: Dict, context: List[Dict],
                                 speakers: Dict[str, Speaker], 
                                 current_speaker: str) -> str:
        """Определяет говорящего для конкретного сегмента"""
        
        # Простая эвристика для начала
        text_lower = segment['text'].lower()
        
        # Ищем индикаторы смены говорящего
        for indicator in self.speaker_change_indicators:
            if indicator.lower() in text_lower:
                # Вероятно смена говорящего
                # Выбираем следующего по циклу
                speaker_ids = list(speakers.keys())
                current_idx = speaker_ids.index(current_speaker)
                next_idx = (current_idx + 1) % len(speaker_ids)
                return speaker_ids[next_idx]
        
        # Проверяем характерные фразы каждого говорящего
        best_match = current_speaker
        best_score = 0
        
        for speaker_id, speaker in speakers.items():
            score = 0
            for characteristic in speaker.characteristics:
                if characteristic.lower() in text_lower:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_match = speaker_id
        
        # Если нет явных признаков, используем длину пауз
        if len(context) > 0:
            prev_segment = context[-1]
            pause = segment['start'] - (prev_segment['start'] + prev_segment.get('duration', 0))
            if pause > 2.0:  # Пауза больше 2 секунд - возможно смена говорящего
                # Вероятно смена
                speaker_ids = list(speakers.keys())
                if len(speaker_ids) > 1 and best_score == 0:
                    current_idx = speaker_ids.index(current_speaker)
                    next_idx = (current_idx + 1) % len(speaker_ids)
                    return speaker_ids[next_idx]
        
        return best_match if best_score > 0 else current_speaker
    
    def get_segment_context(self, segments: List[Dict], current_idx: int, 
                          window_size: int = 5) -> List[Dict]:
        """Получает контекст вокруг сегмента"""
        start_idx = max(0, current_idx - window_size)
        end_idx = min(len(segments), current_idx + window_size + 1)
        return segments[start_idx:end_idx]
    
    def create_timeline(self, transcript_segments: List[Dict], 
                       screenshots: List[Tuple]) -> List[TimelineEvent]:
        """Создает единую временную линию из транскрипта и скриншотов"""
        
        timeline = []
        
        # Добавляем сегменты транскрипта
        for segment in transcript_segments:
            event = TimelineEvent(
                timestamp=segment['start'],
                type='transcript',
                content={
                    'text': segment['text'],
                    'duration': segment.get('duration', 0),
                    'speaker_id': segment.get('speaker_id', 'unknown'),
                    'original_segment': segment
                }
            )
            timeline.append(event)
        
        # Добавляем скриншоты
        for screenshot_path, timestamp, description, reason in screenshots:
            event = TimelineEvent(
                timestamp=float(timestamp),
                type='screenshot',
                content={
                    'path': screenshot_path,
                    'description': description,
                    'reason': reason
                },
                importance=0.8  # Скриншоты обычно важны
            )
            timeline.append(event)
        
        # Сортируем по времени
        timeline.sort(key=lambda x: x.timestamp)
        
        # Определяем изменения тем
        self.detect_topic_changes(timeline)
        
        return timeline
    
    def detect_topic_changes(self, timeline: List[TimelineEvent]):
        """Определяет моменты смены темы обсуждения"""
        
        # Группируем текстовые события для анализа
        text_groups = []
        current_group = []
        
        for event in timeline:
            if event.type == 'transcript':
                current_group.append(event)
                # Анализируем каждые 10 сегментов
                if len(current_group) >= 10:
                    self.analyze_topic_group(current_group, timeline)
                    current_group = current_group[-5:]  # Оставляем перекрытие
        
        # Анализируем последнюю группу
        if current_group:
            self.analyze_topic_group(current_group, timeline)
    
    def analyze_topic_group(self, text_events: List[TimelineEvent], 
                          timeline: List[TimelineEvent]):
        """Анализирует группу текстовых событий на предмет темы"""
        
        texts = [e.content['text'] for e in text_events]
        combined_text = ' '.join(texts)
        
        # Простой анализ ключевых слов для определения темы
        # В реальности здесь можно использовать более сложные методы
        topic_keywords = {
            'введение': ['привет', 'добрый день', 'начнем', 'представлюсь'],
            'презентация': ['слайд', 'покажу', 'демонстрация', 'пример'],
            'код': ['код', 'функция', 'метод', 'переменная', 'класс'],
            'обсуждение': ['вопрос', 'что думаете', 'мнение', 'обсудим'],
            'заключение': ['итог', 'вывод', 'резюме', 'спасибо', 'вопросы']
        }
        
        detected_topic = None
        max_score = 0
        
        for topic, keywords in topic_keywords.items():
            score = sum(1 for kw in keywords if kw in combined_text.lower())
            if score > max_score:
                max_score = score
                detected_topic = topic
        
        if detected_topic and max_score > 2:
            # Добавляем событие смены темы
            topic_event = TimelineEvent(
                timestamp=text_events[0].timestamp,
                type='topic_change',
                content={'topic': detected_topic},
                importance=0.6
            )
            timeline.append(topic_event)
    
    def extract_terminology_from_screenshots(self, screenshot_events: List[TimelineEvent]) -> Dict:
        """АГЕНТ извлекает терминологию из скриншотов для коррекции Whisper"""
        
        logger.info("🤖 АГЕНТ анализирует скриншоты для извлечения терминологии...")
        
        terminology_dict = {'by_timestamp': {}, 'all_terms': set()}
        
        for event in screenshot_events:
            detailed_content = event.content.get('detailed_content', {})
            if not detailed_content:
                continue
            
            timestamp = event.timestamp
            
            # АГЕНТ анализирует содержимое каждого скриншота
            prompt = f"""Ты - эксперт по извлечению терминологии из деловых документов. Проанализируй содержимое скриншота и извлеки ВСЮ точную терминологию.

СОДЕРЖИМОЕ СКРИНШОТА:
Тип контента: {detailed_content.get('main_content_type', 'неизвестно')}
Видимый текст: {detailed_content.get('visible_text', '')}
Код/команды: {detailed_content.get('code_snippets', [])}
Данные таблиц: {detailed_content.get('table_data', [])}
UI элементы: {detailed_content.get('ui_elements', [])}
Технические детали: {detailed_content.get('technical_details', [])}

ТВОЯ ЗАДАЧА:
Извлеки ВСЕ термины, которые могут быть неправильно распознаны Whisper:
- Названия полей, таблиц, переменных
- Имена файлов, функций, методов
- Технические термины и аббревиатуры
- Точные числовые значения и даты
- UI элементы (кнопки, поля)
- Специфические названия проектов/систем

ВАЖНО: Учитывай, что Whisper часто:
- Переводит английские термины на русский
- Искажает техническую терминологию
- Неправильно распознает числа и даты
- Путает похожие по звучанию слова

Верни JSON с терминами, которые важно исправить в речи:
{{
    "critical_terms": ["список самых важных терминов"],
    "field_names": ["названия полей/переменных"],
    "exact_values": ["точные значения, числа, даты"],
    "file_names": ["имена файлов"],
    "ui_elements": ["элементы интерфейса"],
    "whisper_errors": [
        {{
            "correct": "правильный термин",
            "likely_errors": ["возможные ошибки Whisper"]
        }}
    ]
}}"""

            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.1
                )
                
                result = json.loads(response.choices[0].message.content)
                
                # Сохраняем по времени
                terminology_dict['by_timestamp'][timestamp] = result
                
                # Добавляем в общий словарь
                for category, terms in result.items():
                    if category != 'whisper_errors' and isinstance(terms, list):
                        terminology_dict['all_terms'].update(terms)
                
                logger.info(f"✅ Извлечено {len(result.get('critical_terms', []))} критических терминов в {timestamp:.1f}с")
                
            except Exception as e:
                logger.error(f"Ошибка при анализе терминологии: {e}")
        
        total_terms = len(terminology_dict['all_terms'])
        logger.info(f"🎯 АГЕНТ извлек {total_terms} уникальных терминов из {len(screenshot_events)} скриншотов")
        
        return terminology_dict
    

    
    def correct_whisper_with_terminology(self, text: str, terminology_dict: Dict, 
                                       timestamp: float) -> str:
        """АГЕНТ корректирует ошибки Whisper используя терминологию из скриншотов"""
        
        if not text or not terminology_dict:
            return text
            
        # Получаем релевантные термины для этого времени (±30 секунд)
        relevant_terminology = []
        for ts, terms_data in terminology_dict.get('by_timestamp', {}).items():
            if abs(ts - timestamp) <= 30:  # В пределах 30 секунд
                relevant_terminology.append({
                    'timestamp': ts,
                    'terms': terms_data
                })
        
        if not relevant_terminology:
            return text
        
        # АГЕНТ корректирует текст
        prompt = f"""Ты - эксперт по исправлению ошибок распознавания речи Whisper. У тебя есть ТОЧНАЯ терминология из скриншотов.

ИСХОДНЫЙ ТЕКСТ (с ошибками Whisper):
"{text}"

ТОЧНАЯ ТЕРМИНОЛОГИЯ ИЗ СКРИНШОТОВ:
{json.dumps(relevant_terminology, ensure_ascii=False, indent=2)}

ТВОЯ ЗАДАЧА:
Исправь ошибки Whisper, используя ТОЧНУЮ терминологию со скриншотов:

1. 🔍 Найди искаженные термины в тексте
2. ✏️  Замени их на ТОЧНЫЕ версии из скриншотов  
3. 📝 Исправь только явные ошибки, не меняй смысл
4. 🎯 Приоритет - названия полей, файлов, техническим терминам

ТИПИЧНЫЕ ОШИБКИ WHISPER:
- "дата басе" → "database"  
- "файл таблицы" → название конкретной таблицы
- "кнопка сохранить" → точное название кнопки
- числа словами → цифры

Верни JSON:
{{
    "corrected_text": "исправленный текст",
    "corrections": [
        {{
            "original": "ошибочный фрагмент",
            "corrected": "правильный термин",
            "source": "откуда взят правильный термин"
        }}
    ]
}}

ВАЖНО: Исправляй только то, что есть в терминологии!"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            result = json.loads(response.choices[0].message.content)
            corrected_text = result.get('corrected_text', text)
            corrections = result.get('corrections', [])
            
            if corrections:
                logger.info(f"🤖 АГЕНТ исправил {len(corrections)} ошибок Whisper в {timestamp:.1f}с")
                for correction in corrections[:3]:  # Показываем первые 3
                    logger.info(f"   ✏️  '{correction['original']}' → '{correction['corrected']}'")
            
            return corrected_text
            
        except Exception as e:
            logger.error(f"Ошибка при коррекции АГЕНТОМ: {e}")
            return text
    

    
    def correct_transcript_with_context(self, timeline: List[TimelineEvent],
                                      speakers: Dict[str, Speaker],
                                      video_context: Dict) -> List[TimelineEvent]:
        """Корректирует транскрипт с учетом ПОЛНОГО контекста + терминология из скриншотов"""
        
        logger.info("🔧 Улучшенная коррекция с терминологией из скриншотов...")
        
        # Собираем события по типам
        transcript_events = [e for e in timeline if e.type == 'transcript']
        screenshot_events = [e for e in timeline if e.type == 'screenshot']
        
        if not transcript_events:
            return timeline
        
        # 1. НОВОЕ: Дополняем скриншоты детальным анализом содержимого
        enhanced_screenshots = self.enhance_screenshots_with_content(screenshot_events)
        
        # 2. НОВОЕ: Извлекаем терминологию из скриншотов
        terminology_dict = self.extract_terminology_from_screenshots(enhanced_screenshots)
        
        # 3. НОВОЕ: Предварительно корректируем Whisper ошибки с помощью терминологии
        for event in transcript_events:
            original_text = event.content['text']
            corrected_text = self.correct_whisper_with_terminology(
                original_text, terminology_dict, event.timestamp
            )
            event.content['text'] = corrected_text
            event.content['whisper_corrections'] = corrected_text != original_text
        
        # 4. Основная коррекция с контекстными блоками
        corrected_events = self.correct_with_context_blocks(
            transcript_events, enhanced_screenshots, speakers, video_context, terminology_dict
        )
        
        # Создаем новый timeline с откорректированными событиями
        corrected_timeline = []
        transcript_map = {e.timestamp: e for e in corrected_events}
        
        for event in timeline:
            if event.type == 'transcript' and event.timestamp in transcript_map:
                corrected_timeline.append(transcript_map[event.timestamp])
            elif event.type == 'screenshot':
                # Используем улучшенные скриншоты
                enhanced_event = next((e for e in enhanced_screenshots if e.timestamp == event.timestamp), event)
                corrected_timeline.append(enhanced_event)
            else:
                corrected_timeline.append(event)
        
        # Сортируем по времени
        corrected_timeline.sort(key=lambda x: x.timestamp)
        
        return corrected_timeline
    
    def create_processing_groups(self, timeline: List[TimelineEvent], 
                               group_duration: float = 30.0) -> List[List[TimelineEvent]]:
        """Создает группы событий для обработки"""
        
        groups = []
        current_group = []
        group_start = 0
        
        for event in timeline:
            if not current_group:
                group_start = event.timestamp
                current_group.append(event)
            elif event.timestamp - group_start <= group_duration:
                current_group.append(event)
            else:
                groups.append(current_group)
                current_group = [event]
                group_start = event.timestamp
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def correct_whole_transcript(self, transcript_events: List[TimelineEvent],
                               screenshot_events: List[TimelineEvent],
                               speakers: Dict[str, Speaker],
                               video_context: Dict) -> List[TimelineEvent]:
        """Корректирует ВЕСЬ транскрипт целиком с полным контекстом"""
        
        logger.info("🔧 Улучшенная коррекция с сохранением деталей...")
        
        # Собираем ПОЛНЫЙ диалог по времени
        full_dialogue = []
        speaker_profiles = {}
        
        for event in transcript_events:
            speaker_id = event.content.get('speaker_id', 'unknown')
            speaker_name = speakers.get(speaker_id, Speaker(id=speaker_id)).name or f"Участник {speaker_id[-1] if speaker_id != 'unknown' else '1'}"
            
            # Строим профили спикеров
            if speaker_name not in speaker_profiles:
                speaker_profiles[speaker_name] = {
                    'total_speech': [],
                    'topics': set(),
                    'style': 'формальный'
                }
            
            text = event.content['text']
            speaker_profiles[speaker_name]['total_speech'].append(text)
            
            full_dialogue.append({
                'time': event.timestamp,
                'speaker': speaker_name,
                'text': text,
                'original_event': event
            })
        
        # Собираем информацию о скриншотах
        visual_context = ""
        if screenshot_events:
            visual_context = "\n\nВизуальная информация в видео:\n"
            for ss_event in screenshot_events:
                time_str = f"{int(ss_event.timestamp//60)}:{int(ss_event.timestamp%60):02d}"
                desc = ss_event.content.get('description', '')
                reason = ss_event.content.get('reason', 'изменение')
                visual_context += f"[{time_str}] {reason}: {desc}\n"
        
        # 🚀 НОВЫЙ ПОДХОД: разбиваем на перекрывающиеся блоки
        corrected_events = self.correct_with_context_blocks(
            full_dialogue, speaker_profiles, visual_context, video_context
        )
        
        logger.info(f"✅ Откорректировано {len(corrected_events)} реплик с сохранением деталей")
        return corrected_events
    
    def correct_with_context_blocks(self, transcript_events: List[TimelineEvent], 
                                   enhanced_screenshots: List[TimelineEvent],
                                   speakers: Dict[str, Speaker],
                                   video_context: Dict, terminology_dict: Dict) -> List[TimelineEvent]:
        """Новый метод коррекции с контекстными блоками + терминология из скриншотов"""
        
        # Преобразуем события в диалог для обработки
        full_dialogue = []
        speaker_profiles = {}
        
        for event in transcript_events:
            speaker_id = event.content.get('speaker_id', 'unknown')
            speaker_name = speakers.get(speaker_id, Speaker(id=speaker_id)).name or f"Участник {speaker_id[-1] if speaker_id != 'unknown' else '1'}"
            
            # Строим профили спикеров
            if speaker_name not in speaker_profiles:
                speaker_profiles[speaker_name] = {
                    'total_speech': [],
                    'topics': set(),
                    'style': 'формальный'
                }
            
            text = event.content['text']
            speaker_profiles[speaker_name]['total_speech'].append(text)
            
            full_dialogue.append({
                'time': event.timestamp,
                'speaker': speaker_name,
                'text': text,
                'original_event': event
            })
        
        # Определяем размер блока (примерно 15-20 реплик)
        block_size = 20
        overlap_size = 5  # Перекрытие для сохранения контекста
        
        corrected_events = []
        
        for block_start in range(0, len(full_dialogue), block_size - overlap_size):
            block_end = min(block_start + block_size, len(full_dialogue))
            
            # Текущий блок
            current_block = full_dialogue[block_start:block_end]
            
            # Контекст до блока (предыдущие 3 реплики)
            context_before = full_dialogue[max(0, block_start-3):block_start]
            
            # Контекст после блока (следующие 3 реплики) 
            context_after = full_dialogue[block_end:min(len(full_dialogue), block_end+3)]
            
            # Определяем тематический контекст блока
            block_theme = self.analyze_block_theme(current_block)
            
            # Определяем временной диапазон блока
            block_start_time = current_block[0]['time'] if current_block else 0
            block_end_time = current_block[-1]['time'] if current_block else 0
            
            logger.info(f"🔍 Обрабатываем блок {block_start//block_size + 1}: реплики {block_start+1}-{block_end} (тема: {block_theme})")
            
            # Получаем детальное содержимое скриншотов для этого блока
            visual_context = self.get_screenshot_content_for_time(
                enhanced_screenshots, (block_start_time + block_end_time) / 2, window=60
            )
            
            # Корректируем блок с полным контекстом + терминология
            corrected_block = self.correct_context_block(
                current_block, context_before, context_after,
                block_theme, speaker_profiles, visual_context, video_context, terminology_dict
            )
            
            # Добавляем откорректированные события (избегаем дубликатов при перекрытии)
            start_idx = overlap_size if block_start > 0 else 0
            for i in range(start_idx, len(corrected_block)):
                corrected_events.append(corrected_block[i])
        
        return corrected_events
    
    def analyze_block_theme(self, block: List[Dict]) -> str:
        """Анализирует тематику блока диалога"""
        
        block_text = ' '.join([item['text'] for item in block]).lower()
        
        # Простая эвристика для определения темы
        themes = {
            'техническое обсуждение': ['оборудование', 'система', 'данные', 'база', 'таблица', 'схема'],
            'планирование': ['план', 'задача', 'сроки', 'дедлайн', 'график'],
            'анализ проблем': ['проблема', 'ошибка', 'исправить', 'решение', 'баг'],
            'демонстрация': ['показать', 'смотреть', 'экран', 'слайд', 'код'],
            'обсуждение процессов': ['процесс', 'этап', 'шаг', 'порядок', 'последовательность']
        }
        
        best_theme = 'общее обсуждение'
        max_score = 0
        
        for theme, keywords in themes.items():
            score = sum(1 for keyword in keywords if keyword in block_text)
            if score > max_score:
                max_score = score
                best_theme = theme
        
        return best_theme
    
    def correct_context_block(self, current_block: List[Dict],
                             context_before: List[Dict], context_after: List[Dict],
                             block_theme: str, speaker_profiles: Dict,
                             visual_context: str, video_context: Dict,
                             terminology_dict: Dict) -> List[TimelineEvent]:
        """Корректирует блок с учетом полного контекста"""
        
        # Формируем контекстную информацию
        context_info = ""
        if context_before:
            context_info += "ПРЕДШЕСТВУЮЩИЙ КОНТЕКСТ:\n"
            for item in context_before:
                time_str = f"{int(item['time']//60)}:{int(item['time']%60):02d}"
                context_info += f"[{time_str}] {item['speaker']}: {item['text']}\n"
            context_info += "\n"
        
        # Текущий блок
        current_text = ""
        for item in current_block:
            time_str = f"{int(item['time']//60)}:{int(item['time']%60):02d}"
            current_text += f"[{time_str}] {item['speaker']}: {item['text']}\n"
        
        # Последующий контекст
        if context_after:
            context_info += "\nПОСЛЕДУЮЩИЙ КОНТЕКСТ:\n"
            for item in context_after:
                time_str = f"{int(item['time']//60)}:{int(item['time']%60):02d}"
                context_info += f"[{time_str}] {item['speaker']}: {item['text']}\n"
        
        # 🤖 АГЕНТНЫЙ ПРОМПТ с использованием терминологии из скриншотов
        prompt = f"""Ты - эксперт по восстановлению точного смысла деловых встреч. У тебя есть ДЕТАЛЬНАЯ информация с экрана + точная терминология.

КОНТЕКСТ ВСТРЕЧИ:
- Тип: {video_context.get('meeting_type', 'деловая встреча')}
- Тематика блока: {block_theme}
- Участники: {', '.join(speaker_profiles.keys())}

{context_info}

ОСНОВНОЙ БЛОК ДЛЯ КОРРЕКЦИИ:
{current_text}

ДЕТАЛЬНАЯ ИНФОРМАЦИЯ С ЭКРАНА:
{visual_context}

ИЗВЛЕЧЕННАЯ ТЕРМИНОЛОГИЯ ИЗ СКРИНШОТОВ:
{json.dumps(terminology_dict, default=lambda x: list(x) if isinstance(x, set) else x, ensure_ascii=False, indent=2)[:2000]}

ТВОЯ ГЛАВНАЯ ЗАДАЧА:
🎯 Восстанови ТОЧНЫЙ смысл диалога, используя:
1. Детальное содержимое скриншотов (что реально видно на экране)
2. Точную терминологию (правильные названия полей, файлов, систем)
3. Техническую информацию (коды, данные, команды)
4. Контекст разговора (что было до и после)

ОСОБОЕ ВНИМАНИЕ:
💡 Если участник говорит общие фразы типа "вот здесь", "это поле", "эта таблица" - 
   замени на КОНКРЕТНЫЕ названия из скриншотов!

🔍 Если Whisper исказил технические термины - исправь по терминологии!

📊 Если упоминаются данные/числа - используй точные значения с экрана!

ПРИМЕРЫ УЛУЧШЕНИЙ:
❌ "Там разобрали собрали снова протестировали" 
✅ "Оборудование model_X123 разобрали, протестировали систему inventory_tracking, выявили ошибки в поле date_received, затем собрали и отправили"

❌ "Здесь соответственно следующая строчка по оборудованию"
✅ "В таблице equipment_log следующая запись показывает статус 'completed' для единицы с ID 15847"

Верни JSON с МАКСИМАЛЬНО ДЕТАЛЬНЫМИ репликами:
{{
    "corrected_dialogue": [
        {{
            "timestamp": временная_метка,
            "speaker": "имя_спикера",
            "corrected_text": "МАКСИМАЛЬНО ДЕТАЛЬНАЯ реплика с конкретными названиями, числами, терминами",
            "screen_references": "что конкретно видно на экране в этот момент",
            "technical_details": "извлеченные технические детали",
            "context_connection": "связь с общим контекстом"
        }}
    ],
    "block_summary": "подробное резюме технических аспектов блока"
}}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",  # Используем мощную модель
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.05,  # Минимальная креативность для точности
                max_tokens=6000    # Увеличили лимит для деталей
            )
            
            result = json.loads(response.choices[0].message.content)
            corrected_dialogue = result.get('corrected_dialogue', [])
            block_summary = result.get('block_summary', '')
            
            # Создаем откорректированные события
            corrected_events = []
            for i, original_item in enumerate(current_block):
                if i < len(corrected_dialogue):
                    corrected = corrected_dialogue[i]
                    
                    original_event = original_item['original_event']
                    
                    # Создаем новый event с детальной информацией
                    new_event = TimelineEvent(
                        timestamp=original_event.timestamp,
                        type='transcript',
                        content={
                            'text': original_event.content['text'],  # Оригинал
                            'corrected_text': corrected.get('corrected_text', original_event.content['text']),
                            'speaker_id': original_event.content.get('speaker_id'),
                            'speaker_name': corrected.get('speaker'),
                            'technical_details': corrected.get('technical_details', ''),
                            'context_connection': corrected.get('context_connection', ''),
                            'duration': original_event.content.get('duration', 0),
                            'block_summary': block_summary if i == 0 else None,
                            'theme': block_theme
                        },
                        importance=original_event.importance
                    )
                    corrected_events.append(new_event)
                else:
                    # Если не хватило откорректированных - берем оригинал
                    corrected_events.append(original_item['original_event'])
            
            return corrected_events
            
        except Exception as e:
            logger.error(f"Ошибка при коррекции блока: {e}")
            # Возвращаем исходные события в случае ошибки
            return [item['original_event'] for item in current_block]
    
    def correct_transcript_group(self, transcript_events: List[TimelineEvent],
                               screenshot_events: List[TimelineEvent],
                               speakers: Dict[str, Speaker],
                               video_context: Dict) -> List[TimelineEvent]:
        """Корректирует группу транскриптов с учетом скриншотов и создает ОСМЫСЛЕННЫЙ текст"""
        
        # Собираем контекст с привязкой к говорящим
        speakers_texts = {}
        timeline_context = []
        
        for event in transcript_events:
            speaker_id = event.content.get('speaker_id', 'unknown')
            speaker_name = speakers.get(speaker_id, Speaker(id=speaker_id)).name or f"Участник {speaker_id[-1] if speaker_id != 'unknown' else '1'}"
            
            if speaker_name not in speakers_texts:
                speakers_texts[speaker_name] = []
            
            speakers_texts[speaker_name].append(event.content['text'])
            timeline_context.append({
                'time': event.timestamp,
                'speaker': speaker_name,
                'text': event.content['text']
            })
        
        # Информация о визуальном контексте
        visual_context = ""
        if screenshot_events:
            visual_context = "\n\nВИЗУАЛЬНЫЙ КОНТЕКСТ на экране в этот период:\n"
            for ss_event in screenshot_events:
                time_str = f"{int(ss_event.timestamp//60)}:{int(ss_event.timestamp%60):02d}"
                desc = ss_event.content.get('description', '')
                reason = ss_event.content.get('reason', 'изменение')
                visual_context += f"[{time_str}] {reason}: {desc}\n"
        
        # Создаем детализированный промпт для ОСМЫСЛЕННОЙ коррекции
        prompt = f"""Ты - эксперт по анализу видео встреч. Твоя задача - создать ОСМЫСЛЕННЫЙ, контекстно-связанный транскрипт.

ТИП ВСТРЕЧИ: {video_context.get('meeting_type', 'обсуждение')}
ОСНОВНЫЕ ТЕМЫ: {', '.join(video_context.get('main_topics', []))}

УЧАСТНИКИ:
{chr(10).join([f"- {name}: {len(texts)} реплик" for name, texts in speakers_texts.items()])}

ИСХОДНЫЙ ДИАЛОГ (по времени):
{chr(10).join([f"[{int(item['time']//60)}:{int(item['time']%60):02d}] {item['speaker']}: {item['text']}" for item in timeline_context])}
{visual_context}

ЗАДАЧИ КОРРЕКЦИИ:
1. **Исправь ошибки распознавания речи** - восстанови правильные слова и термины
2. **Добавь пунктуацию и структуру** - сделай текст читаемым
3. **Учти визуальный контекст** - если говорят "смотрите сюда" или "как видите", добавь пояснение [показывает на экране...]
4. **Сохрани хронологию** - порядок реплик должен остаться тем же
5. **Добавь контекстные связки** - если реплики связаны, покажи это
6. **Убери повторы и паузы** - оставь только смысловое содержание
7. **Добавь пояснения в скобках** - если упоминаются технические термины или визуальные элементы

ВАЖНО: Создай связный, осмысленный диалог, а не набор отдельных фраз!

Верни JSON с откорректированными сегментами:
{{
    "corrected_segments": [
        {{
            "speaker": "имя_говорящего",
            "corrected_text": "откорректированный осмысленный текст",
            "context_explanation": "пояснение контекста если нужно",
            "visual_reference": "ссылка на визуальный элемент если упоминается"
        }}
    ],
    "overall_context": "общий контекст происходящего в этом временном отрезке"
}}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=3000
            )
            
            # Парсим ответ
            result = json.loads(response.choices[0].message.content)
            corrected_segments = result.get('corrected_segments', [])
            overall_context = result.get('overall_context', '')
            
            # Обновляем события с улучшенной информацией
            corrected_events = []
            for i, event in enumerate(transcript_events):
                if i < len(corrected_segments):
                    corrected = corrected_segments[i]
                    
                    # Создаем улучшенный event
                    new_event = TimelineEvent(
                        timestamp=event.timestamp,
                        type='transcript',
                        content={
                            'text': event.content['text'],  # Оригинальный текст
                            'corrected_text': corrected.get('corrected_text', event.content['text']),
                            'speaker_id': event.content.get('speaker_id'),
                            'speaker_name': corrected.get('speaker'),
                            'context_explanation': corrected.get('context_explanation'),
                            'visual_reference': corrected.get('visual_reference'),
                            'duration': event.content.get('duration', 0),
                            'overall_context': overall_context if i == 0 else None  # Добавляем общий контекст только к первому
                        },
                        importance=event.importance
                    )
                    corrected_events.append(new_event)
                else:
                    corrected_events.append(event)
            
            return corrected_events
            
        except Exception as e:
            logger.error(f"Ошибка при коррекции транскрипта: {e}")
            # Возвращаем исходные события в случае ошибки
            return transcript_events
    
    def structure_content(self, timeline: List[TimelineEvent], 
                         speakers: Dict[str, Speaker]) -> Dict:
        """Структурирует контент по темам и участникам"""
        
        structured = {
            'sections': [],
            'speaker_stats': defaultdict(lambda: {
                'word_count': 0,
                'segment_count': 0,
                'topics_discussed': set()
            }),
            'key_moments': []
        }
        
        current_section = {
            'start_time': 0,
            'end_time': 0,
            'topic': 'Начало',
            'events': [],
            'speakers': set()
        }
        
        for event in timeline:
            # Обработка смены темы
            if event.type == 'topic_change':
                # Завершаем текущую секцию
                if current_section['events']:
                    current_section['end_time'] = current_section['events'][-1].timestamp
                    structured['sections'].append(current_section)
                
                # Начинаем новую секцию
                current_section = {
                    'start_time': event.timestamp,
                    'end_time': event.timestamp,
                    'topic': event.content['topic'],
                    'events': [],
                    'speakers': set()
                }
            else:
                current_section['events'].append(event)
                
                # Обновляем статистику
                if event.type == 'transcript':
                    speaker_id = event.content.get('speaker_id', 'unknown')
                    current_section['speakers'].add(speaker_id)
                    
                    # Статистика говорящего
                    text = event.content.get('corrected_text', event.content.get('text', ''))
                    structured['speaker_stats'][speaker_id]['word_count'] += len(text.split())
                    structured['speaker_stats'][speaker_id]['segment_count'] += 1
                    structured['speaker_stats'][speaker_id]['topics_discussed'].add(
                        current_section['topic']
                    )
                
                # Отмечаем ключевые моменты
                if event.type == 'screenshot' or event.importance > 0.7:
                    structured['key_moments'].append({
                        'timestamp': event.timestamp,
                        'type': event.type,
                        'description': self.get_event_description(event)
                    })
        
        # Добавляем последнюю секцию
        if current_section['events']:
            current_section['end_time'] = current_section['events'][-1].timestamp
            structured['sections'].append(current_section)
        
        return structured
    
    def get_event_description(self, event: TimelineEvent) -> str:
        """Получает описание события"""
        
        if event.type == 'screenshot':
            return event.content.get('description', 'Скриншот')
        elif event.type == 'transcript':
            text = event.content.get('corrected_text', event.content.get('text', ''))
            return text[:100] + '...' if len(text) > 100 else text
        elif event.type == 'topic_change':
            return f"Переход к теме: {event.content.get('topic', 'неизвестно')}"
        else:
            return "Событие"
    
    def generate_chronological_report(self, structured_content: Dict,
                                    speakers: Dict[str, Speaker]) -> str:
        """Генерирует финальный хронологический отчет"""
        
        report = []
        
        # Заголовок
        report.append("# Хронологический отчет встречи\n")
        
        # Информация об участниках
        report.append("## Участники\n")
        for speaker_id, speaker in speakers.items():
            stats = structured_content['speaker_stats'].get(speaker_id, {})
            name = speaker.name or f"Участник {speaker_id[-1]}"
            role = speaker.role or "участник"
            
            report.append(f"### {name} ({role})")
            report.append(f"- Сказано слов: {stats.get('word_count', 0)}")
            report.append(f"- Количество реплик: {stats.get('segment_count', 0)}")
            topics = list(stats.get('topics_discussed', set()))
            if topics:
                report.append(f"- Участвовал в обсуждении: {', '.join(topics)}")
            report.append("")
        
        # Ключевые моменты
        if structured_content['key_moments']:
            report.append("\n## Ключевые моменты\n")
            for moment in structured_content['key_moments'][:10]:  # Топ 10
                time = self.format_time(moment['timestamp'])
                report.append(f"- **{time}** - {moment['description']}")
            report.append("")
        
        # Хронологический транскрипт по секциям
        report.append("\n## Хронологический транскрипт\n")
        
        for section in structured_content['sections']:
            # Заголовок секции
            start_time = self.format_time(section['start_time'])
            end_time = self.format_time(section['end_time'])
            report.append(f"\n### {section['topic']} ({start_time} - {end_time})\n")
            
            # Добавляем тематический контекст блоков в секции
            block_themes = set()
            for event in section['events']:
                if event.type == 'transcript':
                    theme = event.content.get('theme')
                    if theme and theme != 'общее обсуждение':
                        block_themes.add(theme)
            
            if block_themes:
                report.append(f"🎯 **Тематика:** {', '.join(block_themes)}\n")
            
            # Собираем резюме блоков в секции
            block_summaries = []
            for event in section['events']:
                if event.type == 'transcript':
                    block_summary = event.content.get('block_summary')
                    if block_summary and block_summary not in block_summaries:
                        block_summaries.append(block_summary)
            
            if block_summaries:
                report.append("📋 **Технические аспекты:**")
                for summary in block_summaries:
                    report.append(f"  • {summary}")
                report.append("")
            
            # События в секции
            for event in section['events']:
                if event.type == 'transcript':
                    # Транскрипт - ТОЛЬКО исправленная версия
                    time = self.format_time(event.timestamp)
                    speaker_id = event.content.get('speaker_id', 'unknown')
                    speaker_name = event.content.get('speaker_name') or \
                                 speakers.get(speaker_id, Speaker(id=speaker_id)).name or \
                                 f"Спикер {speaker_id[-1] if speaker_id != 'unknown' else '1'}"
                    
                    # Используем ТОЛЬКО откорректированный текст
                    corrected_text = event.content.get('corrected_text')
                    if corrected_text and corrected_text.strip():
                        report.append(f"**[{time}] {speaker_name}:** {corrected_text}")
                    else:
                        # Fallback на оригинал только если нет исправленного
                        original_text = event.content.get('text', '').strip()
                        if original_text:
                            report.append(f"**[{time}] {speaker_name}:** {original_text}")
                    
                    # Добавляем НОВЫЕ поля от агента
                    screen_references = event.content.get('screen_references', '')
                    if screen_references and screen_references.strip():
                        report.append(f"  📺 **На экране:** {screen_references}")
                    
                    technical_details = event.content.get('technical_details', '')
                    if technical_details and technical_details.strip():
                        report.append(f"  ⚙️ **Технические детали:** {technical_details}")
                    
                    context_connection = event.content.get('context_connection', '')
                    if context_connection and context_connection.strip():
                        report.append(f"  🔗 **Контекстная связь:** {context_connection}")
                    
                    # Старые контекстные пояснения (для совместимости)
                    context_note = event.content.get('context_explanation')
                    if context_note and context_note.strip():
                        report.append(f"  📝 **Контекст:** {context_note}")
                    
                    report.append("")
                    
                elif event.type == 'screenshot':
                    # Скриншот
                    time = self.format_time(event.timestamp)
                    description = event.content.get('description', 'Скриншот')
                    reason = event.content.get('reason', '')
                    path = event.content.get('path', '')
                    
                    report.append(f"\n📸 **[{time}] Скриншот** - {reason}")
                    report.append(f"*{description}*")
                    
                    # Конвертируем изображение в base64 для встраивания
                    if path and os.path.exists(path):
                        try:
                            with open(path, "rb") as image_file:
                                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                                ext = os.path.splitext(path)[1].lstrip('.')
                                base64_url = f"data:image/{ext};base64,{encoded_string}"
                                report.append(f"![Скриншот]({base64_url})\n")
                        except Exception as e:
                            logger.error(f"Ошибка при кодировании изображения: {e}")
                            report.append(f"![Скриншот]({path})\n")
                    else:
                        report.append("")
        
        return '\n'.join(report)
    
    def format_time(self, seconds: float) -> str:
        """Форматирует время в читаемый вид"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    def analyze_screenshot_content(self, screenshot_path: str, timestamp: float) -> Dict:
        """Детально анализирует содержимое скриншота"""
        
        try:
            # Кодируем изображение в base64
            with open(screenshot_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Промпт для детального извлечения
            prompt = """Детально проанализируй содержимое этого скриншота из деловой встречи.

ИЗВЛЕКИ ВСЁ:
1. **Текст на экране** - ТОЧНО скопируй весь видимый текст
2. **Коды/команды** - если есть программный код, SQL, команды - скопируй точно
3. **Данные из таблиц** - числа, даты, статусы, значения
4. **Схемы/диаграммы** - опиши структуру и элементы
5. **UI элементы** - кнопки, поля, меню с их названиями
6. **Технические детали** - версии, параметры, настройки

Верни подробную информацию в JSON:
{
    "visible_text": "весь видимый текст дословно",
    "code_snippets": ["фрагменты кода/команд если есть"],
    "table_data": ["данные из таблиц/списков"],
    "technical_details": ["версии, параметры, технические значения"],
    "ui_elements": ["названия кнопок, полей, меню"],
    "diagrams_schemas": "описание схем/диаграмм",
    "main_content_type": "тип содержимого (код/документ/таблица/диаграмма/презентация)",
    "key_information": "самая важная информация с экрана"
}

КРИТИЧЕСКИ ВАЖНО: Копируй текст и данные ТОЧНО, не обобщай!"""

            response = self.client.chat.completions.create(
                model="gpt-4o",  # Используем мощную модель для анализа изображений
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
                max_tokens=2000
            )
            
            result = json.loads(response.choices[0].message.content)
            logger.info(f"✅ Детально проанализирован скриншот в {timestamp:.1f}с")
            return result
            
        except Exception as e:
            logger.error(f"Ошибка при анализе содержимого скриншота: {e}")
            return {}
    
    def enhance_screenshots_with_content(self, screenshot_events: List[TimelineEvent]) -> List[TimelineEvent]:
        """Дополняет скриншоты детальным анализом содержимого"""
        
        enhanced_events = []
        
        for event in screenshot_events:
            screenshot_path = event.content.get('path', '')
            
            if screenshot_path and os.path.exists(screenshot_path):
                # Анализируем содержимое скриншота
                detailed_content = self.analyze_screenshot_content(
                    screenshot_path, event.timestamp
                )
                
                # Создаем улучшенное событие
                enhanced_content = event.content.copy()
                enhanced_content['detailed_content'] = detailed_content
                
                enhanced_event = TimelineEvent(
                    timestamp=event.timestamp,
                    type=event.type,
                    content=enhanced_content,
                    importance=event.importance
                )
                enhanced_events.append(enhanced_event)
            else:
                enhanced_events.append(event)
        
        return enhanced_events
    
    def get_screenshot_content_for_time(self, screenshot_events: List[TimelineEvent], 
                                      target_time: float, window: float = 30) -> str:
        """Получает детальное содержимое скриншотов для определенного времени"""
        
        relevant_screenshots = []
        
        for event in screenshot_events:
            if abs(event.timestamp - target_time) <= window:
                detailed = event.content.get('detailed_content', {})
                
                if detailed:
                    time_str = f"{int(event.timestamp//60)}:{int(event.timestamp%60):02d}"
                    content_info = f"\n📸 СКРИНШОТ в {time_str}:\n"
                    
                    # Основной тип контента
                    content_type = detailed.get('main_content_type', 'неизвестно')
                    content_info += f"   Тип: {content_type}\n"
                    
                    # Видимый текст
                    visible_text = detailed.get('visible_text', '')
                    if visible_text and visible_text.strip():
                        content_info += f"   📄 Текст на экране: {visible_text[:200]}...\n"
                    
                    # Код/команды
                    code_snippets = detailed.get('code_snippets', [])
                    if code_snippets:
                        content_info += f"   💻 Код/команды: {'; '.join(code_snippets[:3])}\n"
                    
                    # Данные таблиц
                    table_data = detailed.get('table_data', [])
                    if table_data:
                        content_info += f"   📊 Данные таблиц: {'; '.join(table_data[:3])}\n"
                    
                    # Технические детали
                    tech_details = detailed.get('technical_details', [])
                    if tech_details:
                        content_info += f"   ⚙️ Технические детали: {'; '.join(tech_details[:3])}\n"
                    
                    # Ключевая информация
                    key_info = detailed.get('key_information', '')
                    if key_info and key_info.strip():
                        content_info += f"   🎯 Ключевая информация: {key_info}\n"
                    
                    relevant_screenshots.append(content_info)
        
        if relevant_screenshots:
            return "\n".join(relevant_screenshots)
        else:
            return ""

def integrate_chronological_processor(transcript_segments: List[Dict],
                                    screenshots: List[Tuple],
                                    video_context: Dict,
                                    api_key: str) -> Dict:
    """Интегрирует хронологический процессор в основной pipeline"""
    
    processor = ChronologicalTranscriptProcessor(api_key)
    result = processor.process_video_meeting(
        transcript_segments, screenshots, video_context
    )
    
    return result

# Пример использования
if __name__ == "__main__":
    # Тестовые данные
    test_segments = [
        {"text": "Всем привет, давайте начнем нашу встречу", "start": 0, "duration": 3},
        {"text": "Меня зовут Иван, я буду вести презентацию", "start": 3, "duration": 4},
        {"text": "Сегодня мы обсудим новый проект", "start": 7, "duration": 3},
        {"text": "У кого есть вопросы?", "start": 30, "duration": 2},
        {"text": "Да, у меня вопрос по архитектуре", "start": 35, "duration": 3},
    ]
    
    test_screenshots = [
        ("screenshot_00010s.jpg", 10, "Слайд с заголовком презентации", "presentation_slide"),
        ("screenshot_00040s.jpg", 40, "Диаграмма архитектуры системы", "diagram"),
    ]
    
    test_context = {
        'meeting_type': 'presentation',
        'main_topics': ['новый проект', 'архитектура', 'планы']
    }
    
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        processor = ChronologicalTranscriptProcessor(api_key)
        result = processor.process_video_meeting(
            test_segments, test_screenshots, test_context
        )
        
        print("=== ОТЧЕТ ===")
        print(result['report'])
    else:
        print("Требуется OPENAI_API_KEY")
