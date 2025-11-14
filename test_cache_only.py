#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Специальный тест только для CacheManager (без внешних зависимостей)
"""

import sys
import os
from pathlib import Path
import tempfile
import shutil

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent / "New"))

from cache_manager import CacheManager


def run_all_cache_tests():
    """Запуск всех тестов CacheManager"""
    print("=" * 70)
    print("🧪 ПОЛНОЕ ТЕСТИРОВАНИЕ CACHE MANAGER")
    print("=" * 70)

    temp_dir = tempfile.mkdtemp()
    passed = 0
    failed = 0

    try:
        cache = CacheManager(cache_dir=temp_dir)
        video_path = os.path.join(temp_dir, "test.mp4")

        # Создаем тестовый видео файл
        with open(video_path, 'wb') as f:
            f.write(b'test video content')

        tests = [
            ("1. Инициализация", test_init, (cache, temp_dir)),
            ("2. Генерация хэша", test_hash, (cache, video_path)),
            ("3. Пути к кэшу", test_paths, (cache, video_path)),
            ("4. Сохранение метаданных", test_metadata, (cache, video_path)),
            ("5. Сохранение транскрипта", test_transcript, (cache, video_path)),
            ("6. Сохранение анализа", test_analysis, (cache, video_path)),
            ("7. Статус кэша", test_status, (cache, video_path)),
            ("8. Очистка кэша", test_clear, (cache, video_path)),
            ("9. Переиспользование кэша", test_reuse, (cache, video_path)),
            ("10. Cleanup старых файлов", test_cleanup, (cache,)),
        ]

        print()
        for name, test_func, args in tests:
            try:
                print(f"Тест {name}...", end=" ")
                result = test_func(*args)
                if result:
                    print("✅ PASSED")
                    passed += 1
                else:
                    print("❌ FAILED")
                    failed += 1
            except Exception as e:
                print(f"❌ FAILED: {e}")
                failed += 1

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    print("\n" + "=" * 70)
    print(f"ИТОГО: {passed} пройдено, {failed} провалено из {len(tests)} тестов")
    print("=" * 70)

    return failed == 0


def test_init(cache, temp_dir):
    """Тест инициализации"""
    assert os.path.exists(temp_dir)
    assert cache.cache_dir == temp_dir
    return True


def test_hash(cache, video_path):
    """Тест генерации хэша"""
    hash1 = cache.get_video_hash(video_path)
    hash2 = cache.get_video_hash(video_path)
    assert hash1 == hash2
    assert len(hash1) == 32
    return True


def test_paths(cache, video_path):
    """Тест получения путей"""
    paths = cache.get_cache_paths(video_path)
    assert 'audio' in paths
    assert 'transcript_segments' in paths
    assert 'transcript_full' in paths
    assert 'analysis' in paths
    assert 'screenshots' in paths
    assert 'metadata' in paths
    return True


def test_metadata(cache, video_path):
    """Тест метаданных"""
    metadata = {'test_key': 'test_value', 'number': 42}
    cache.save_metadata(video_path, metadata)
    loaded = cache.load_metadata(video_path)
    assert loaded is not None
    assert loaded['test_key'] == 'test_value'
    assert loaded['number'] == 42
    assert 'processed_at' in loaded
    return True


def test_transcript(cache, video_path):
    """Тест транскрипта"""
    segments = [
        {"text": "Test segment 1", "start": 0.0, "duration": 2.0},
        {"text": "Test segment 2", "start": 2.0, "duration": 3.0}
    ]
    full_text = "Test segment 1 Test segment 2"

    cache.save_transcript_cache(video_path, segments, full_text)
    loaded = cache.get_cached_transcript(video_path)

    assert loaded is not None
    loaded_segments, loaded_text = loaded
    assert len(loaded_segments) == 2
    assert loaded_text == full_text
    assert loaded_segments[0]['text'] == "Test segment 1"
    return True


def test_analysis(cache, video_path):
    """Тест анализа"""
    analysis = "This is test analysis"
    cache.save_analysis_cache(video_path, analysis, "test_type")
    loaded = cache.get_cached_analysis(video_path, "test_type")
    assert loaded == analysis
    return True


def test_status(cache, video_path):
    """Тест статуса кэша"""
    status = cache.get_cache_status(video_path)
    assert status['transcript'] is True
    assert status['metadata'] is True
    assert status['analysis'] is True
    return True


def test_clear(cache, video_path):
    """Тест очистки кэша"""
    # Проверяем, что данные есть
    status_before = cache.get_cache_status(video_path)
    assert status_before['transcript'] is True

    # Очищаем
    cache.clear_cache(video_path)

    # Проверяем, что данные удалены
    status_after = cache.get_cache_status(video_path)
    assert status_after['transcript'] is False
    assert status_after['metadata'] is False
    return True


def test_reuse(cache, video_path):
    """Тест переиспользования кэша"""
    # Сохраняем данные
    cache.save_transcript_cache(
        video_path,
        [{"text": "reuse test", "start": 0, "duration": 1}],
        "reuse test"
    )

    # Загружаем дважды
    data1 = cache.get_cached_transcript(video_path)
    data2 = cache.get_cached_transcript(video_path)

    # Должны быть идентичны
    assert data1 == data2
    return True


def test_cleanup(cache):
    """Тест cleanup старых файлов"""
    # Создаем старый файл
    import time
    old_file = os.path.join(cache.cache_dir, "old_test.txt")
    with open(old_file, 'w') as f:
        f.write("old content")

    # Делаем его старым
    old_time = time.time() - (10 * 24 * 60 * 60)
    os.utime(old_file, (old_time, old_time))

    # Очищаем
    removed = cache.cleanup_old_cache(days=7)

    # Проверяем, что старый файл удален
    assert not os.path.exists(old_file)
    assert removed >= 1
    return True


if __name__ == "__main__":
    success = run_all_cache_tests()
    sys.exit(0 if success else 1)
