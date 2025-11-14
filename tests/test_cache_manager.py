#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Тесты для CacheManager
"""

import os
import sys
import json
import tempfile
import shutil
import pytest
from pathlib import Path

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent.parent / "New"))

from cache_manager import CacheManager


class TestCacheManager:
    """Тесты для менеджера кэша"""

    @pytest.fixture
    def temp_cache_dir(self):
        """Создает временную директорию для кэша"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Очистка после теста
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    @pytest.fixture
    def temp_video_file(self, temp_cache_dir):
        """Создает временный видео файл для тестов"""
        video_path = os.path.join(temp_cache_dir, "test_video.mp4")
        with open(video_path, 'wb') as f:
            f.write(b'fake video content')
        yield video_path

    @pytest.fixture
    def cache_manager(self, temp_cache_dir):
        """Создает экземпляр CacheManager"""
        return CacheManager(cache_dir=temp_cache_dir)

    def test_init(self, temp_cache_dir):
        """Тест инициализации"""
        manager = CacheManager(cache_dir=temp_cache_dir)
        assert os.path.exists(temp_cache_dir)
        assert manager.cache_dir == temp_cache_dir

    def test_get_video_hash(self, cache_manager, temp_video_file):
        """Тест генерации хэша видео"""
        hash1 = cache_manager.get_video_hash(temp_video_file)
        hash2 = cache_manager.get_video_hash(temp_video_file)

        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 хэш

    def test_get_cache_paths(self, cache_manager, temp_video_file):
        """Тест получения путей к кэшу"""
        paths = cache_manager.get_cache_paths(temp_video_file)

        assert 'audio' in paths
        assert 'transcript_segments' in paths
        assert 'transcript_full' in paths
        assert 'analysis' in paths
        assert 'screenshots' in paths
        assert 'metadata' in paths

        # Проверяем, что все пути содержат хэш
        video_hash = cache_manager.get_video_hash(temp_video_file)
        for path in paths.values():
            assert video_hash in path

    def test_save_and_load_metadata(self, cache_manager, temp_video_file):
        """Тест сохранения и загрузки метаданных"""
        metadata = {
            'whisper_model': 'base',
            'screenshot_mode': 'smart',
            'test_field': 'test_value'
        }

        cache_manager.save_metadata(temp_video_file, metadata)
        loaded_metadata = cache_manager.load_metadata(temp_video_file)

        assert loaded_metadata is not None
        assert loaded_metadata['whisper_model'] == 'base'
        assert loaded_metadata['screenshot_mode'] == 'smart'
        assert loaded_metadata['test_field'] == 'test_value'
        assert 'processed_at' in loaded_metadata
        assert 'video_path' in loaded_metadata

    def test_save_and_get_transcript(self, cache_manager, temp_video_file):
        """Тест сохранения и получения транскрипта"""
        segments = [
            {"text": "Hello world", "start": 0.0, "duration": 2.0},
            {"text": "Test segment", "start": 2.0, "duration": 3.0}
        ]
        full_text = "Hello world Test segment"

        cache_manager.save_transcript_cache(temp_video_file, segments, full_text)
        cached_data = cache_manager.get_cached_transcript(temp_video_file)

        assert cached_data is not None
        cached_segments, cached_full_text = cached_data

        assert len(cached_segments) == 2
        assert cached_segments[0]['text'] == "Hello world"
        assert cached_full_text == full_text

    def test_save_and_get_analysis(self, cache_manager, temp_video_file):
        """Тест сохранения и получения анализа"""
        analysis_text = "This is a test analysis of the video content."

        cache_manager.save_analysis_cache(temp_video_file, analysis_text, "basic")
        cached_analysis = cache_manager.get_cached_analysis(temp_video_file, "basic")

        assert cached_analysis is not None
        assert cached_analysis == analysis_text

    def test_get_cache_status(self, cache_manager, temp_video_file):
        """Тест получения статуса кэша"""
        # Изначально все должно быть False
        status = cache_manager.get_cache_status(temp_video_file)

        assert status['audio'] is False
        assert status['transcript'] is False
        assert status['analysis'] is False

        # Сохраняем транскрипт
        cache_manager.save_transcript_cache(
            temp_video_file,
            [{"text": "test", "start": 0, "duration": 1}],
            "test"
        )

        # Проверяем снова
        status = cache_manager.get_cache_status(temp_video_file)
        assert status['transcript'] is True

    def test_clear_cache(self, cache_manager, temp_video_file):
        """Тест очистки кэша"""
        # Создаем некоторые кэшированные данные
        cache_manager.save_transcript_cache(
            temp_video_file,
            [{"text": "test", "start": 0, "duration": 1}],
            "test"
        )
        cache_manager.save_metadata(temp_video_file, {"test": "data"})

        # Проверяем, что данные есть
        status_before = cache_manager.get_cache_status(temp_video_file)
        assert status_before['transcript'] is True
        assert status_before['metadata'] is True

        # Очищаем кэш
        cache_manager.clear_cache(temp_video_file)

        # Проверяем, что данные удалены
        status_after = cache_manager.get_cache_status(temp_video_file)
        assert status_after['transcript'] is False
        assert status_after['metadata'] is False

    def test_cleanup_old_cache(self, cache_manager, temp_cache_dir):
        """Тест очистки старых файлов кэша"""
        # Создаем старый файл
        old_file = os.path.join(temp_cache_dir, "old_file.txt")
        with open(old_file, 'w') as f:
            f.write("old content")

        # Изменяем время модификации на очень старое
        import time
        old_time = time.time() - (10 * 24 * 60 * 60)  # 10 дней назад
        os.utime(old_file, (old_time, old_time))

        # Создаем новый файл
        new_file = os.path.join(temp_cache_dir, "new_file.txt")
        with open(new_file, 'w') as f:
            f.write("new content")

        # Очищаем старые файлы (старше 7 дней)
        removed_count = cache_manager.cleanup_old_cache(days=7)

        assert removed_count >= 1
        assert not os.path.exists(old_file)
        assert os.path.exists(new_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
