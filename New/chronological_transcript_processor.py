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
    
    def correct_transcript_with_context(self, timeline: List[TimelineEvent],
                                      speakers: Dict[str, Speaker],
                                      video_context: Dict) -> List[TimelineEvent]:
        """Корректирует транскрипт с учетом контекста и скриншотов"""
        
        logger.info("Корректируем транскрипт с учетом контекста...")
        
        # Группируем события для пакетной обработки
        processing_groups = self.create_processing_groups(timeline)
        
        corrected_timeline = []
        
        for group in processing_groups:
            # Находим скриншоты в группе
            screenshots_in_group = [e for e in group if e.type == 'screenshot']
            transcript_in_group = [e for e in group if e.type == 'transcript']
            
            if not transcript_in_group:
                corrected_timeline.extend(group)
                continue
            
            # Корректируем транскрипты в группе
            corrected_events = self.correct_transcript_group(
                transcript_in_group, screenshots_in_group, speakers, video_context
            )
            
            # Объединяем с остальными событиями
            for event in group:
                if event.type == 'transcript':
                    # Находим откорректированную версию
                    for corrected in corrected_events:
                        if corrected.timestamp == event.timestamp:
                            corrected_timeline.append(corrected)
                            break
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
            
            # События в секции
            for event in section['events']:
                if event.type == 'transcript':
                    # Транскрипт
                    time = self.format_time(event.timestamp)
                    speaker_id = event.content.get('speaker_id', 'unknown')
                    speaker_name = event.content.get('speaker_name') or \
                                 speakers.get(speaker_id, Speaker(id=speaker_id)).name or \
                                 f"Участник {speaker_id[-1]}"
                    
                    text = event.content.get('corrected_text', event.content.get('text', ''))
                    context_note = event.content.get('context_explanation')
                    
                    report.append(f"**[{time}] {speaker_name}:** {text}")
                    if context_note:
                        report.append(f"*[{context_note}]*")
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
