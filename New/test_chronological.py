#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Диагностический скрипт для проверки хронологического процессора
"""

import json
import os
import sys
from pprint import pprint
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

# Добавляем папку в путь
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chronological_transcript_processor import ChronologicalTranscriptProcessor

def test_speaker_identification():
    """Тестирует определение участников"""
    print("=" * 50)
    print("ТЕСТ 1: Определение участников")
    print("=" * 50)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Нет API ключа OpenAI")
        return
    
    # Загружаем тестовые сегменты транскрипта
    test_segments = [
        {
            "start": 0.0,
            "end": 11.0,
            "text": "Вот есть куитом, где у нас просто вот одним валом насыпанную, все возможно, все нет, нет, не открылось.",
            "duration": 11.0
        },
        {
            "start": 11.0,
            "end": 20.0,
            "text": "Навалина все наше оборудование, одним бесконечным списком есть табличка. Действие с этим оборудованием также валом Навалина.",
            "duration": 9.0
        },
        {
            "start": 20.0,
            "end": 33.0,
            "text": "Есть табличка ActivitySN, ID, которым все это дело соединяется, чем мне такое шли бы шои.",
            "duration": 13.0
        }
    ]
    
    processor = ChronologicalTranscriptProcessor(api_key)
    
    try:
        speakers = processor.identify_speakers(test_segments)
        print(f"✅ Обнаружено участников: {len(speakers)}")
        for speaker_id, speaker in speakers.items():
            print(f"\n{speaker_id}:")
            print(f"  Имя: {speaker.name}")
            print(f"  Роль: {speaker.role}")
            print(f"  Характеристики: {speaker.characteristics}")
        
        # Проверяем назначение говорящих
        print("\n\nПроверка назначения говорящих сегментам:")
        for seg in test_segments[:3]:
            print(f"\n[{seg['start']:.1f}s] \"{seg['text'][:50]}...\"")
            print(f"  Говорящий: {seg.get('speaker_id', 'НЕ НАЗНАЧЕН')}")
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

def test_timeline_creation():
    """Тестирует создание временной линии"""
    print("\n" + "=" * 50)
    print("ТЕСТ 2: Создание временной линии")
    print("=" * 50)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Нет API ключа OpenAI")
        return
        
    processor = ChronologicalTranscriptProcessor(api_key)
    
    # Тестовые данные
    test_segments = [
        {"start": 0.0, "text": "Начало встречи", "duration": 5.0},
        {"start": 5.0, "text": "Обсуждение темы", "duration": 10.0}
    ]
    
    test_screenshots = [
        ("screenshot_001.jpg", 3.0, "Презентация", "Начало презентации"),
        ("screenshot_002.jpg", 10.0, "Диаграмма", "Показ диаграммы")
    ]
    
    try:
        timeline = processor.create_timeline(test_segments, test_screenshots)
        print(f"✅ Создано событий: {len(timeline)}")
        
        for event in timeline[:5]:
            print(f"\n{event.timestamp:.1f}s - {event.type}")
            if event.type == 'transcript':
                print(f"  Текст: {event.content['text'][:50]}...")
            elif event.type == 'screenshot':
                print(f"  Скриншот: {event.content['path']}")
                print(f"  Описание: {event.content['description']}")
                
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

def test_full_processing():
    """Тестирует полный процесс обработки"""
    print("\n" + "=" * 50)
    print("ТЕСТ 3: Полная обработка мини-версии")
    print("=" * 50)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Нет API ключа OpenAI")
        return
        
    # Проверяем кэшированные данные
    cache_file = "results/Запись встречи 09.07.2025 10-52-18 - запись_analysis.json"
    if os.path.exists(cache_file):
        print(f"📁 Используем кэшированные данные из {cache_file}")
        with open(cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Берем первые 20 сегментов для теста
        test_segments = data['transcript_segments'][:20]
        print(f"✅ Загружено {len(test_segments)} сегментов")
    else:
        # Используем простые тестовые данные
        test_segments = [
            {"start": 0.0, "text": "Привет всем! Меня зовут Иван.", "duration": 3.0},
            {"start": 3.0, "text": "Сегодня мы обсудим новый проект.", "duration": 4.0},
            {"start": 7.0, "text": "Мария, что ты думаешь об этом?", "duration": 3.0},
            {"start": 10.0, "text": "Думаю, это хорошая идея, Иван.", "duration": 3.0},
            {"start": 13.0, "text": "Но есть несколько вопросов.", "duration": 2.0}
        ]
    
    test_screenshots = [
        ("screenshot_001.jpg", 5.0, "Слайд презентации", "Показ архитектуры"),
        ("screenshot_002.jpg", 12.0, "Диаграмма", "Схема процесса")
    ]
    
    video_context = {
        "title": "Тестовая встреча",
        "duration": 60.0,
        "format": "webm"
    }
    
    processor = ChronologicalTranscriptProcessor(api_key)
    
    try:
        print("\n🚀 Запускаем полную обработку...")
        result = processor.process_video_meeting(
            test_segments,
            test_screenshots,
            video_context
        )
        
        print("\n✅ Обработка завершена!")
        print(f"Участников: {len(result['speakers'])}")
        print(f"Событий на timeline: {len(result['timeline'])}")
        
        # Проверяем отчет
        if 'report' in result:
            print("\n📄 Сгенерирован отчет:")
            print(result['report'][:500] + "...")
            
            # Сохраняем для анализа
            with open("test_chronological_output.md", "w", encoding="utf-8") as f:
                f.write(result['report'])
            print("\n💾 Отчет сохранен в test_chronological_output.md")
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

def analyze_problem():
    """Анализирует проблему с повторяющимся текстом"""
    print("\n" + "=" * 50)
    print("ТЕСТ 4: Анализ проблемы с 'Роману.'")
    print("=" * 50)
    
    problem_file = "results/Запись встречи 09.07.2025 10-52-18 - запись_INTEGRATED_chronological.md"
    
    # Используем grep для поиска проблемы
    print("\n🔍 Ищем повторения 'Роману.'...")
    os.system(f"grep -c 'Роману\\.' \"{problem_file}\" | head -1")
    
    print("\n🔍 Проверяем структуру файла...")
    os.system(f"head -n 50 \"{problem_file}\" | grep -E '(##|Участники|speaker)'")

if __name__ == "__main__":
    print("🧪 ДИАГНОСТИКА ХРОНОЛОГИЧЕСКОГО ПРОЦЕССОРА\n")
    
    # Запускаем тесты
    test_speaker_identification()
    test_timeline_creation()
    test_full_processing()
    analyze_problem()
    
    print("\n✅ Диагностика завершена!") 