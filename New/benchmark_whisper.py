#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Бенчмарк производительности Whisper на M1 Max
Сравнивает скорость работы разных моделей и настроек
"""

import time
import whisper
import torch
import numpy as np
import importlib.util
spec = importlib.util.spec_from_file_location("updated_video_analyzer", "updated-video-analyzer.py")
updated_video_analyzer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(updated_video_analyzer)
get_optimal_device = updated_video_analyzer.get_optimal_device
load_whisper_optimized = updated_video_analyzer.load_whisper_optimized

def benchmark_whisper():
    """Тестирует производительность разных моделей Whisper"""
    
    print("🏃 Бенчмарк Whisper на вашем устройстве")
    print("=" * 50)
    
    # Создаем тестовый аудио (30 секунд тишины для теста)
    test_audio = np.zeros(16000 * 30, dtype=np.float32)  # 30 секунд
    test_file = "temp_benchmark.wav"
    
    import soundfile as sf
    sf.write(test_file, test_audio, 16000)
    
    models_to_test = ["tiny", "base", "small"]
    results = {}
    
    for model_size in models_to_test:
        print(f"\n📊 Тестирование модели: {model_size}")
        
        # Тест с оптимизацией
        start_time = time.time()
        model, device = load_whisper_optimized(model_size)
        load_time = time.time() - start_time
        
        # Прогрев модели
        print("   Прогрев модели...")
        _ = model.transcribe(test_file, language="ru", fp16=(device != "cpu"))
        
        # Основной тест
        print("   Транскрипция...")
        start_time = time.time()
        for i in range(3):
            result = model.transcribe(
                test_file, 
                language="ru", 
                fp16=(device != "cpu"),
                temperature=0.0,
                beam_size=5
            )
        
        avg_time = (time.time() - start_time) / 3
        
        results[model_size] = {
            "load_time": load_time,
            "transcribe_time": avg_time,
            "real_time_factor": 30.0 / avg_time  # Во сколько раз быстрее реального времени
        }
        
        print(f"   ✅ Загрузка: {load_time:.2f}с")
        print(f"   ✅ Транскрипция 30с аудио: {avg_time:.2f}с")
        print(f"   ✅ Скорость: {results[model_size]['real_time_factor']:.1f}x реального времени")
        
        # Освобождаем память
        del model
        if device == "mps" or device == "cuda":
            torch.cuda.empty_cache() if device == "cuda" else None
    
    # Удаляем временный файл
    import os
    os.unlink(test_file)
    
    # Рекомендации
    print("\n" + "=" * 50)
    print("📋 РЕКОМЕНДАЦИИ для вашего M1 Max:")
    print()
    
    # Находим оптимальную модель
    best_model = None
    best_rtf = 0
    
    for model, stats in results.items():
        if stats['real_time_factor'] > best_rtf and stats['real_time_factor'] > 5:
            best_model = model
            best_rtf = stats['real_time_factor']
    
    if best_model:
        print(f"✅ Рекомендуемая модель: {best_model}")
        print(f"   - Скорость: {best_rtf:.1f}x реального времени")
        print(f"   - Час записи обработается за ~{60/best_rtf:.1f} минут")
    
    print("\n💡 Советы по ускорению:")
    print("   1. Используйте модель 'base' для баланса качества/скорости")
    print("   2. Модель 'large' работает ОЧЕНЬ медленно на CPU")
    print("   3. Включите fp16=True для ускорения на 20-30%")
    print("   4. Для длинных записей используйте --clean-transcript")
    
    return results

if __name__ == "__main__":
    benchmark_whisper() 