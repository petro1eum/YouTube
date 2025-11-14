#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Базовые тесты без pytest для проверки работоспособности
"""

import sys
import os
from pathlib import Path

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent / "New"))

def test_imports():
    """Тест импорта всех модулей"""
    print("=" * 60)
    print("ТЕСТ 1: Проверка импортов модулей")
    print("=" * 60)

    modules_to_test = [
        ("cache_manager", "CacheManager"),
        ("chronological_transcript_processor", "ChronologicalTranscriptProcessor"),
        ("adaptive_screenshot_extractor", "AdaptiveScreenshotExtractor"),
        ("smart_transcript_extractor", "SmartTranscriptExtractor"),
    ]

    success_count = 0
    fail_count = 0

    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name)
            cls = getattr(module, class_name)
            print(f"✅ {module_name}.{class_name} - OK")
            success_count += 1
        except Exception as e:
            print(f"❌ {module_name}.{class_name} - FAILED: {e}")
            fail_count += 1

    print(f"\nРезультат: {success_count} успешно, {fail_count} провалено")
    return fail_count == 0


def test_cache_manager():
    """Тест CacheManager"""
    print("\n" + "=" * 60)
    print("ТЕСТ 2: Проверка CacheManager")
    print("=" * 60)

    try:
        from cache_manager import CacheManager
        import tempfile
        import shutil

        # Создаем временную директорию
        temp_dir = tempfile.mkdtemp()

        try:
            # Создаем менеджер кэша
            cache = CacheManager(cache_dir=temp_dir)
            print(f"✅ CacheManager создан: {temp_dir}")

            # Создаем тестовый видео файл
            video_path = os.path.join(temp_dir, "test.mp4")
            with open(video_path, 'wb') as f:
                f.write(b'test video content')

            # Проверяем генерацию хэша
            hash1 = cache.get_video_hash(video_path)
            hash2 = cache.get_video_hash(video_path)
            assert hash1 == hash2, "Хэши должны совпадать"
            print(f"✅ Генерация хэша работает: {hash1}")

            # Проверяем сохранение/загрузку транскрипта
            test_segments = [{"text": "test", "start": 0, "duration": 1}]
            cache.save_transcript_cache(video_path, test_segments, "test")

            loaded = cache.get_cached_transcript(video_path)
            assert loaded is not None, "Транскрипт должен загрузиться"
            print("✅ Сохранение/загрузка транскрипта работает")

            # Проверяем статус кэша
            status = cache.get_cache_status(video_path)
            assert status['transcript'] is True, "Транскрипт должен быть в кэше"
            print("✅ Статус кэша работает")

            # Проверяем очистку кэша
            cache.clear_cache(video_path)
            status_after = cache.get_cache_status(video_path)
            assert status_after['transcript'] is False, "Кэш должен быть очищен"
            print("✅ Очистка кэша работает")

            print("\n✅ Все тесты CacheManager пройдены!")
            return True

        finally:
            # Удаляем временную директорию
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    except Exception as e:
        print(f"\n❌ Тест CacheManager провален: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_json_parser():
    """Тест safe_json_parse"""
    print("\n" + "=" * 60)
    print("ТЕСТ 3: Проверка safe_json_parse")
    print("=" * 60)

    try:
        from chronological_transcript_processor import safe_json_parse

        # Тест 1: Валидный JSON
        result = safe_json_parse('{"key": "value"}', "test1")
        assert result.get('key') == 'value', "Должен распарсить валидный JSON"
        print("✅ Валидный JSON распознан")

        # Тест 2: JSON с markdown блоками
        result = safe_json_parse('```json\n{"key": "value"}\n```', "test2")
        assert result.get('key') == 'value', "Должен распарсить JSON в markdown"
        print("✅ JSON в markdown блоке распознан")

        # Тест 3: Невалидный JSON
        result = safe_json_parse('invalid json', "test3")
        assert result == {}, "Должен вернуть пустой dict для невалидного JSON"
        print("✅ Невалидный JSON обработан корректно")

        print("\n✅ Все тесты safe_json_parse пройдены!")
        return True

    except Exception as e:
        print(f"\n❌ Тест safe_json_parse провален: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_structures():
    """Тест структур данных"""
    print("\n" + "=" * 60)
    print("ТЕСТ 4: Проверка структур данных")
    print("=" * 60)

    try:
        from chronological_transcript_processor import Speaker, TimelineEvent

        # Тест Speaker
        speaker = Speaker(
            id="speaker1",
            name="Тестовый спикер",
            role="presenter",
            characteristics=["энергичный"],
            voice_segments=[(0.0, 10.0)]
        )
        assert speaker.id == "speaker1"
        assert speaker.name == "Тестовый спикер"
        print("✅ Speaker создан корректно")

        # Тест TimelineEvent
        event = TimelineEvent(
            timestamp=10.5,
            type="transcript",
            content={"text": "Test text"},
            importance=0.8
        )
        assert event.timestamp == 10.5
        assert event.type == "transcript"
        print("✅ TimelineEvent создан корректно")

        print("\n✅ Все тесты структур данных пройдены!")
        return True

    except Exception as e:
        print(f"\n❌ Тест структур данных провален: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Запуск всех тестов"""
    print("\n🧪 ЗАПУСК БАЗОВЫХ ТЕСТОВ\n")

    results = []

    # Запускаем тесты
    results.append(("Импорты модулей", test_imports()))
    results.append(("CacheManager", test_cache_manager()))
    results.append(("safe_json_parse", test_json_parser()))
    results.append(("Структуры данных", test_data_structures()))

    # Итоги
    print("\n" + "=" * 60)
    print("ИТОГИ ТЕСТИРОВАНИЯ")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    failed = sum(1 for _, result in results if not result)

    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{name}: {status}")

    print(f"\nВсего: {len(results)} тестов")
    print(f"Пройдено: {passed}")
    print(f"Провалено: {failed}")

    if failed == 0:
        print("\n🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
        return 0
    else:
        print(f"\n⚠️  {failed} тест(ов) провалено")
        return 1


if __name__ == "__main__":
    sys.exit(main())
