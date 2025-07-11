#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Мониторинг прогресса анализа видео
"""

import time
import os
import json
from datetime import datetime

def monitor_analysis():
    """Отслеживает прогресс анализа"""
    
    print("📊 Мониторинг анализа видео")
    print("=" * 50)
    
    # Папки для проверки
    cache_dir = "cache"
    results_dir = "optimized_results"
    
    # Файлы для мониторинга
    video_id = "c716d82418a0df0bdc9a276c1ea9ac24"
    
    files_to_check = {
        "📝 Транскрипт": f"{cache_dir}/{video_id}_transcript_segments.json",
        "📄 Полный текст": f"{cache_dir}/{video_id}_transcript_full.txt",
        "📊 Анализ": f"{cache_dir}/{video_id}_analysis.json",
        "🎯 Результаты": f"{results_dir}/Запись встречи 09.07.2025 10-52-18 - запись_analysis.md"
    }
    
    last_status = {}
    
    while True:
        os.system('clear' if os.name == 'posix' else 'cls')
        print(f"📊 Мониторинг анализа видео - {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 50)
        
        all_done = True
        
        for name, filepath in files_to_check.items():
            if os.path.exists(filepath):
                size = os.path.getsize(filepath) / 1024 / 1024  # В MB
                mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                
                # Проверяем изменения
                is_new = filepath not in last_status
                is_growing = last_status.get(filepath, {}).get('size', 0) < size
                
                status = "✅ Готово"
                if is_new:
                    status = "🆕 Создан"
                elif is_growing:
                    status = "📈 Обновляется"
                    all_done = False
                
                print(f"{name}: {status} ({size:.1f} MB) - {mtime.strftime('%H:%M:%S')}")
                
                last_status[filepath] = {'size': size, 'mtime': mtime}
            else:
                print(f"{name}: ⏳ Ожидание...")
                all_done = False
        
        # Проверяем процесс
        print("\n" + "-" * 50)
        ps_output = os.popen("ps aux | grep 'python.*updated-video-analyzer' | grep -v grep").read()
        if ps_output:
            print("🚀 Процесс активен")
            # Извлекаем использование CPU
            cpu_usage = ps_output.split()[2]
            print(f"   CPU: {cpu_usage}%")
        else:
            print("❌ Процесс не найден")
            if all_done:
                print("\n✅ АНАЛИЗ ЗАВЕРШЕН!")
                break
        
        # Показываем последние строки логов если есть
        if os.path.exists(f"{cache_dir}/{video_id}_transcript_full.txt"):
            print("\n📜 Последний обработанный фрагмент:")
            with open(f"{cache_dir}/{video_id}_transcript_full.txt", 'r', encoding='utf-8') as f:
                content = f.read()
                last_200_chars = content[-200:] if len(content) > 200 else content
                print(f"...{last_200_chars}")
        
        time.sleep(5)  # Обновление каждые 5 секунд
    
    print("\n✅ Анализ завершен!")
    print(f"📁 Результаты в папке: {results_dir}/")

if __name__ == "__main__":
    monitor_analysis() 