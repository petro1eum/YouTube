#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Скрипт для анализа текста/транскрипта с помощью OpenAI
"""

import os
import sys
import json
import requests
from dotenv import load_dotenv

def analyze_with_openai(text, api_key, title="Анализируемый текст"):
    """Анализ текста с помощью OpenAI API"""
    prompt = f"""
    Проанализируй следующий текст/транскрипт.
    
    Текст:
    {text}
    
    Создай следующее:
    1. Краткое содержание (3-5 предложений)
    2. Основные тезисы и ключевые мысли (8-10 пунктов в виде маркированного списка). Выдели самые важные идеи, концепции и примеры из текста. Рассмотри различные аспекты обсуждаемой темы.
    """
    
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "Ты - аналитик контента. Твоя задача - выделять ключевые идеи и создавать подробный анализ содержания."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 2000
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions", 
            headers=headers, 
            json=payload
        )
        result = response.json()
        
        # Получаем ответ от API
        if "choices" in result and len(result["choices"]) > 0:
            analysis = result["choices"][0]["message"]["content"]
            return analysis
        else:
            print(f"Ошибка при анализе: {result}")
            return None
    except Exception as e:
        print(f"Ошибка при анализе текста: {e}")
        return None

def save_results(text, analysis, output_file="analysis_result"):
    """Сохранение результатов анализа"""
    output_dir = "results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Сохраняем в формате Markdown
    md_file = f"{output_dir}/{output_file}.md"
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(f"# Анализ текста\n\n")
        f.write(analysis)
        f.write("\n\n## Исходный текст\n\n")
        f.write(text)
        
    print(f"Анализ сохранен в Markdown файл: {md_file}")
    
    # Сохраняем JSON
    json_file = f"{output_dir}/{output_file}.json"
    results = {
        "text": text,
        "analysis": analysis
    }
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    print(f"Результаты также сохранены в JSON: {json_file}")
    
    return md_file, json_file

def main():
    # Загружаем переменные окружения из .env файла
    load_dotenv()
    
    if len(sys.argv) < 2:
        print("Использование:")
        print("  python analyze_text.py <текст_для_анализа>")
        print("  python analyze_text.py --file <путь_к_файлу>")
        print("\nПримеры:")
        print("  python analyze_text.py \"Ваш текст для анализа\"")
        print("  python analyze_text.py --file transcript.txt")
        sys.exit(1)
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("Ошибка: API ключ OpenAI не найден в переменной окружения OPENAI_API_KEY")
        sys.exit(1)
    
    try:
        # Определяем источник текста
        if sys.argv[1] == "--file":
            if len(sys.argv) < 3:
                print("Ошибка: Укажите путь к файлу после --file")
                sys.exit(1)
            
            file_path = sys.argv[2]
            if not os.path.exists(file_path):
                print(f"Ошибка: Файл {file_path} не найден")
                sys.exit(1)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            output_name = os.path.splitext(os.path.basename(file_path))[0] + "_analysis"
        else:
            text = sys.argv[1]
            output_name = "text_analysis"
        
        if not text.strip():
            print("Ошибка: Текст для анализа пуст")
            sys.exit(1)
        
        print(f"Текст получен. Длина: {len(text)} символов")
        
        # Анализируем текст
        print("Анализ текста с помощью OpenAI...")
        analysis = analyze_with_openai(text, api_key)
        
        if not analysis:
            print("Не удалось выполнить анализ текста.")
            sys.exit(1)
        
        # Сохраняем результаты
        md_file, json_file = save_results(text, analysis, output_name)
        
        print("Готово!")
        print(f"Анализ: {md_file}")
        print(f"JSON-файл с результатами: {json_file}")
        
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 