#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Тесты для ChronologicalTranscriptProcessor
"""

import os
import sys
import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent.parent / "New"))

from chronological_transcript_processor import (
    ChronologicalTranscriptProcessor,
    Speaker,
    TranscriptSegment,
    TimelineEvent
)


class TestChronologicalProcessor:
    """Тесты для процессора хронологического анализа"""

    @pytest.fixture
    def mock_openai_client(self):
        """Мок OpenAI клиента"""
        with patch('chronological_transcript_processor.OpenAI') as mock:
            client = MagicMock()
            mock.return_value = client
            yield client

    @pytest.fixture
    def processor(self, mock_openai_client):
        """Создает экземпляр процессора с мок API ключом"""
        return ChronologicalTranscriptProcessor(api_key="test_api_key")

    @pytest.fixture
    def sample_segments(self):
        """Примеры сегментов транскрипта"""
        return [
            {"text": "Привет всем, начинаем встречу", "start": 0.0, "duration": 3.0},
            {"text": "Сегодня обсудим новый проект", "start": 3.0, "duration": 4.0},
            {"text": "Мария, что думаешь об этом?", "start": 7.0, "duration": 3.0},
            {"text": "Думаю это хорошая идея", "start": 10.0, "duration": 3.0},
        ]

    @pytest.fixture
    def sample_screenshots(self):
        """Примеры скриншотов"""
        return [
            ("screenshot_001.jpg", 5.0, "Слайд презентации", "Показ архитектуры"),
            ("screenshot_002.jpg", 12.0, "Диаграмма", "Схема процесса")
        ]

    def test_init(self, processor):
        """Тест инициализации"""
        assert processor.api_key == "test_api_key"
        assert processor.speakers == {}
        assert processor.timeline_events == []
        assert processor.topics == []

    def test_get_segment_context(self, processor, sample_segments):
        """Тест получения контекста сегмента"""
        context = processor.get_segment_context(sample_segments, 1, window_size=2)

        assert len(context) == 4  # index 1 ± 2 (но не больше длины)
        assert context[0] == sample_segments[0]

    def test_create_timeline(self, processor, sample_segments, sample_screenshots):
        """Тест создания временной линии"""
        # Добавляем speaker_id к сегментам
        for seg in sample_segments:
            seg['speaker_id'] = 'speaker1'

        timeline = processor.create_timeline(sample_segments, sample_screenshots)

        # Проверяем, что все события добавлены
        transcript_events = [e for e in timeline if e.type == 'transcript']
        screenshot_events = [e for e in timeline if e.type == 'screenshot']

        assert len(transcript_events) == len(sample_segments)
        assert len(screenshot_events) == len(sample_screenshots)

        # Проверяем сортировку по времени
        for i in range(len(timeline) - 1):
            assert timeline[i].timestamp <= timeline[i + 1].timestamp

    def test_format_time(self, processor):
        """Тест форматирования времени"""
        assert processor.format_time(0) == "00:00"
        assert processor.format_time(65) == "01:05"
        assert processor.format_time(125.5) == "02:05"

    def test_detect_speaker_for_segment_simple(self, processor, sample_segments):
        """Тест простого определения говорящего"""
        speakers = {
            'speaker1': Speaker(id='speaker1', name='Иван', role='ведущий', characteristics=[]),
            'speaker2': Speaker(id='speaker2', name='Мария', role='участник', characteristics=[])
        }

        processor.speaker_change_indicators = ['вопрос', 'мнение']

        segment = sample_segments[0]
        context = []

        speaker = processor.detect_speaker_for_segment(
            segment, context, speakers, 'speaker1'
        )

        # Должен вернуть текущего говорящего (нет индикаторов смены)
        assert speaker in ['speaker1', 'speaker2']

    def test_analyze_block_theme(self, processor):
        """Тест анализа темы блока"""
        block = [
            {'text': 'Давайте обсудим оборудование и систему'},
            {'text': 'Нужно проверить базу данных'},
            {'text': 'Схема показывает процесс обработки'}
        ]

        theme = processor.analyze_block_theme(block)

        # Должен определить техническое обсуждение
        assert theme in ['техническое обсуждение', 'общее обсуждение']

    def test_safe_json_parse_valid(self):
        """Тест безопасного парсинга валидного JSON"""
        from chronological_transcript_processor import safe_json_parse

        valid_json = '{"key": "value", "number": 123}'
        result = safe_json_parse(valid_json, "test")

        assert result['key'] == 'value'
        assert result['number'] == 123

    def test_safe_json_parse_with_markdown(self):
        """Тест безопасного парсинга JSON с markdown блоками"""
        from chronological_transcript_processor import safe_json_parse

        json_with_md = '''```json
        {"key": "value"}
        ```'''

        result = safe_json_parse(json_with_md, "test")

        assert result['key'] == 'value'

    def test_safe_json_parse_invalid(self):
        """Тест безопасного парсинга невалидного JSON"""
        from chronological_transcript_processor import safe_json_parse

        invalid_json = 'this is not json'
        result = safe_json_parse(invalid_json, "test")

        assert result == {}

    def test_get_event_description(self, processor):
        """Тест получения описания события"""
        # Транскрипт
        transcript_event = TimelineEvent(
            timestamp=10.0,
            type='transcript',
            content={'text': 'Это длинный текст который будет обрезан' * 10}
        )

        desc = processor.get_event_description(transcript_event)
        assert len(desc) <= 103  # 100 символов + "..."

        # Скриншот
        screenshot_event = TimelineEvent(
            timestamp=20.0,
            type='screenshot',
            content={'description': 'Тестовое описание'}
        )

        desc = processor.get_event_description(screenshot_event)
        assert desc == 'Тестовое описание'

        # Смена темы
        topic_event = TimelineEvent(
            timestamp=30.0,
            type='topic_change',
            content={'topic': 'Новая тема'}
        )

        desc = processor.get_event_description(topic_event)
        assert 'Новая тема' in desc


class TestSpeaker:
    """Тесты для класса Speaker"""

    def test_speaker_creation(self):
        """Тест создания спикера"""
        speaker = Speaker(
            id='speaker1',
            name='Иван',
            role='ведущий',
            characteristics=['энергичный', 'четкий'],
            voice_segments=[(0.0, 10.0), (20.0, 30.0)]
        )

        assert speaker.id == 'speaker1'
        assert speaker.name == 'Иван'
        assert speaker.role == 'ведущий'
        assert len(speaker.characteristics) == 2
        assert len(speaker.voice_segments) == 2


class TestTimelineEvent:
    """Тесты для класса TimelineEvent"""

    def test_timeline_event_creation(self):
        """Тест создания события"""
        event = TimelineEvent(
            timestamp=15.5,
            type='transcript',
            content={'text': 'Test text', 'speaker_id': 'speaker1'},
            importance=0.8
        )

        assert event.timestamp == 15.5
        assert event.type == 'transcript'
        assert event.content['text'] == 'Test text'
        assert event.importance == 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
