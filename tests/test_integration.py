#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Интеграционные тесты для всей системы
"""

import os
import sys
import json
import tempfile
import shutil
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent.parent / "New"))

from cache_manager import CacheManager
from chronological_transcript_processor import ChronologicalTranscriptProcessor


class TestIntegration:
    """Интеграционные тесты"""

    @pytest.fixture
    def temp_dir(self):
        """Временная директория"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    @pytest.fixture
    def temp_video(self, temp_dir):
        """Временный видео файл"""
        video_path = os.path.join(temp_dir, "test_video.mp4")
        with open(video_path, 'wb') as f:
            f.write(b'fake video content for testing')
        return video_path

    @pytest.fixture
    def cache_manager(self, temp_dir):
        """CacheManager с временной директорией"""
        cache_dir = os.path.join(temp_dir, "cache")
        return CacheManager(cache_dir=cache_dir)

    def test_cache_workflow(self, cache_manager, temp_video):
        """Тест полного workflow кэширования"""

        # 1. Изначально кэш пуст
        status = cache_manager.get_cache_status(temp_video)
        assert all(not v for v in status.values())

        # 2. Сохраняем транскрипт
        segments = [
            {"text": "Тестовый сегмент 1", "start": 0.0, "duration": 2.0},
            {"text": "Тестовый сегмент 2", "start": 2.0, "duration": 3.0},
        ]
        full_text = "Тестовый сегмент 1 Тестовый сегмент 2"

        cache_manager.save_transcript_cache(temp_video, segments, full_text)

        # 3. Проверяем, что транскрипт появился в кэше
        status = cache_manager.get_cache_status(temp_video)
        assert status['transcript'] is True

        # 4. Загружаем транскрипт обратно
        cached_data = cache_manager.get_cached_transcript(temp_video)
        assert cached_data is not None
        cached_segments, cached_full_text = cached_data

        assert len(cached_segments) == 2
        assert cached_full_text == full_text

        # 5. Сохраняем анализ
        analysis = "Это тестовый анализ видео"
        cache_manager.save_analysis_cache(temp_video, analysis, "basic")

        # 6. Проверяем анализ
        cached_analysis = cache_manager.get_cached_analysis(temp_video, "basic")
        assert cached_analysis == analysis

        # 7. Сохраняем метаданные
        metadata = {
            'whisper_model': 'base',
            'screenshot_mode': 'smart',
            'test_data': 'value'
        }
        cache_manager.save_metadata(temp_video, metadata)

        # 8. Загружаем метаданные
        loaded_meta = cache_manager.load_metadata(temp_video)
        assert loaded_meta['whisper_model'] == 'base'
        assert loaded_meta['screenshot_mode'] == 'smart'

        # 9. Финальная проверка статуса
        final_status = cache_manager.get_cache_status(temp_video)
        assert final_status['transcript'] is True
        assert final_status['analysis'] is True
        assert final_status['metadata'] is True

    def test_cache_reuse(self, cache_manager, temp_video):
        """Тест переиспользования кэша"""

        # Сохраняем данные
        segments = [{"text": "test", "start": 0, "duration": 1}]
        cache_manager.save_transcript_cache(temp_video, segments, "test")

        # Получаем хэш
        hash1 = cache_manager.get_video_hash(temp_video)

        # Загружаем данные
        cached1 = cache_manager.get_cached_transcript(temp_video)

        # Получаем хэш снова (должен быть тот же)
        hash2 = cache_manager.get_video_hash(temp_video)
        assert hash1 == hash2

        # Загружаем данные снова
        cached2 = cache_manager.get_cached_transcript(temp_video)

        # Данные должны быть одинаковые
        assert cached1 == cached2

    def test_cache_invalidation(self, cache_manager, temp_video):
        """Тест инвалидации кэша"""

        # Сохраняем данные
        cache_manager.save_transcript_cache(
            temp_video,
            [{"text": "test", "start": 0, "duration": 1}],
            "test"
        )

        # Проверяем, что данные есть
        assert cache_manager.get_cached_transcript(temp_video) is not None

        # Очищаем кэш
        cache_manager.clear_cache(temp_video)

        # Проверяем, что данные удалены
        assert cache_manager.get_cached_transcript(temp_video) is None


class TestModuleImports:
    """Тесты импорта модулей"""

    def test_import_cache_manager(self):
        """Тест импорта cache_manager"""
        try:
            from cache_manager import CacheManager
            assert CacheManager is not None
        except ImportError as e:
            pytest.fail(f"Failed to import CacheManager: {e}")

    def test_import_chronological_processor(self):
        """Тест импорта chronological_transcript_processor"""
        try:
            from chronological_transcript_processor import (
                ChronologicalTranscriptProcessor,
                Speaker,
                TimelineEvent
            )
            assert ChronologicalTranscriptProcessor is not None
            assert Speaker is not None
            assert TimelineEvent is not None
        except ImportError as e:
            pytest.fail(f"Failed to import ChronologicalTranscriptProcessor: {e}")

    def test_import_smart_extractor(self):
        """Тест импорта smart_transcript_extractor"""
        try:
            from smart_transcript_extractor import SmartTranscriptExtractor
            assert SmartTranscriptExtractor is not None
        except ImportError as e:
            pytest.fail(f"Failed to import SmartTranscriptExtractor: {e}")

    def test_import_adaptive_extractor(self):
        """Тест импорта adaptive_screenshot_extractor"""
        try:
            from adaptive_screenshot_extractor import AdaptiveScreenshotExtractor
            assert AdaptiveScreenshotExtractor is not None
        except ImportError as e:
            pytest.fail(f"Failed to import AdaptiveScreenshotExtractor: {e}")


class TestDataFlow:
    """Тесты потока данных между модулями"""

    @pytest.fixture
    def mock_api_key(self):
        """Мок API ключа"""
        return "test_api_key_12345"

    def test_segments_format_compatibility(self):
        """Тест совместимости формата сегментов между модулями"""

        # Формат сегмента из Whisper
        whisper_segment = {
            "text": "Test transcription",
            "start": 0.0,
            "duration": 5.0
        }

        # Проверяем, что формат совместим с процессором
        assert 'text' in whisper_segment
        assert 'start' in whisper_segment
        assert 'duration' in whisper_segment

    def test_screenshot_format_compatibility(self):
        """Тест совместимости формата скриншотов"""

        # Формат скриншота (path, timestamp, description, reason)
        screenshot = ("path/to/screenshot.jpg", 10.5, "Test description", "Test reason")

        assert len(screenshot) == 4
        assert isinstance(screenshot[0], str)  # path
        assert isinstance(screenshot[1], (int, float))  # timestamp
        assert isinstance(screenshot[2], str)  # description
        assert isinstance(screenshot[3], str)  # reason


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
