#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Скрипт быстрого запуска анализатора видео с оптимальными настройками
"""

import os
import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description='Быстрый анализ видео встречи с ИИ',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  %(prog)s video.mp4                    # Быстрый анализ
  %(prog)s video.mp4 --full             # Полный анализ с хронологией
  %(prog)s video.mp4 --fast             # Быстрый режим (tiny модель)
  %(prog)s folder/                      # Анализ всех видео в папке
        """
    )
    
    parser.add_argument('input', help='Путь к видео файлу или папке')
    parser.add_argument('--full', action='store_true', 
                       help='Полный анализ с хронологическим отчетом')
    parser.add_argument('--fast', action='store_true', 
                       help='Быстрый режим (tiny whisper модель)')
    parser.add_argument('--economical', action='store_true',
                       help='Экономичный режим (минимум API вызовов)')
    parser.add_argument('--output', default='results', 
                       help='Папка для результатов (по умолчанию: results)')
    
    args = parser.parse_args()
    
    # Проверяем наличие API ключа
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Ошибка: Не найден OPENAI_API_KEY")
        print("Создайте файл .env и добавьте туда:")
        print("OPENAI_API_KEY=your_api_key_here")
        sys.exit(1)
    
    input_path = Path(args.input)
    
    # Определяем список файлов для обработки
    video_files = []
    if input_path.is_file():
        video_files = [input_path]
    elif input_path.is_dir():
        # Ищем видео файлы в папке
        video_extensions = {'.mp4', '.avi', '.mov', '.webm', '.mkv', '.flv'}
        video_files = [f for f in input_path.iterdir() 
                      if f.suffix.lower() in video_extensions]
        if not video_files:
            print(f"❌ В папке {input_path} не найдено видео файлов")
            sys.exit(1)
    else:
        print(f"❌ Путь {input_path} не существует")
        sys.exit(1)
    
    # Определяем параметры на основе режима
    if args.fast:
        whisper_model = "tiny"
        screenshot_mode = "periodic"
        chronological = False
        print("⚡ Быстрый режим: tiny модель, периодические скриншоты")
    elif args.economical:
        whisper_model = "tiny"
        screenshot_mode = "periodic"
        chronological = False
        print("💰 Экономичный режим: минимум API вызовов, ~$1-3 за час видео")
    elif args.full:
        whisper_model = "base"
        screenshot_mode = "smart"
        chronological = True
        print("🔍 Полный анализ: base модель, умные скриншоты, хронология (~$15-25 за час)")
    else:
        whisper_model = "base"
        screenshot_mode = "smart"
        chronological = False
        print("🤖 Стандартный анализ: base модель, умные скриншоты (~$8-12 за час)")
    
    # Обрабатываем каждый файл
    for i, video_file in enumerate(video_files, 1):
        print(f"\n{'='*60}")
        print(f"Обработка {i}/{len(video_files)}: {video_file.name}")
        print('='*60)
        
        # Формируем команду
        cmd = [
            sys.executable,
            "updated-video-analyzer.py",
            str(video_file),
            "--whisper-model", whisper_model,
            "--screenshot-mode", screenshot_mode,
            "--output", args.output
        ]
        
        if chronological:
            cmd.append("--chronological")
        
        # Запускаем анализ
        import subprocess
        result = subprocess.run(cmd)
        
        if result.returncode != 0:
            print(f"⚠️  Ошибка при обработке {video_file.name}")
        else:
            print(f"✅ {video_file.name} обработан успешно")
    
    print(f"\n{'='*60}")
    print(f"✨ Обработка завершена!")
    print(f"📁 Результаты сохранены в: {args.output}/")
    
    # Показываем сводку
    if len(video_files) > 1:
        print(f"\nОбработано файлов: {len(video_files)}")

if __name__ == "__main__":
    main()
