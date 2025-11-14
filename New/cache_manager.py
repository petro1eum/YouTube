#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Система кэширования для анализатора видео
Позволяет переиспользовать результаты предыдущих запусков
"""

import os
import json
import hashlib
from typing import Optional, Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class CacheManager:
    """Менеджер кэширования для анализатора видео"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    
    def get_video_hash(self, video_path: str) -> str:
        """Генерирует хэш видеофайла для идентификации"""
        # Используем размер файла + модификационное время для быстрого хэша
        stat = os.stat(video_path)
        hash_string = f"{video_path}_{stat.st_size}_{stat.st_mtime}"
        return hashlib.md5(hash_string.encode()).hexdigest()
    
    def get_cache_paths(self, video_path: str) -> Dict[str, str]:
        """Возвращает пути к файлам кэша для видео"""
        video_hash = self.get_video_hash(video_path)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        return {
            'audio': os.path.join(self.cache_dir, f"{video_hash}_audio.wav"),
            'transcript_segments': os.path.join(self.cache_dir, f"{video_hash}_transcript_segments.json"),
            'transcript_full': os.path.join(self.cache_dir, f"{video_hash}_transcript_full.txt"),
            'analysis': os.path.join(self.cache_dir, f"{video_hash}_analysis.json"),
            'screenshots': os.path.join(self.cache_dir, f"{video_hash}_screenshots"),
            'metadata': os.path.join(self.cache_dir, f"{video_hash}_metadata.json")
        }
    
    def save_metadata(self, video_path: str, metadata: Dict):
        """Сохраняет метаданные обработки"""
        paths = self.get_cache_paths(video_path)
        
        # Добавляем информацию о времени создания
        metadata['created_at'] = os.path.getctime(video_path)
        metadata['processed_at'] = __import__('time').time()
        metadata['video_path'] = video_path
        metadata['video_name'] = os.path.splitext(os.path.basename(video_path))[0]
        
        with open(paths['metadata'], 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    def load_metadata(self, video_path: str) -> Optional[Dict]:
        """Загружает метаданные обработки"""
        paths = self.get_cache_paths(video_path)
        
        if not os.path.exists(paths['metadata']):
            return None
        
        try:
            with open(paths['metadata'], 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Ошибка загрузки метаданных: {e}")
            return None
    
    def get_cached_audio(self, video_path: str) -> Optional[str]:
        """Проверяет и возвращает путь к кэшированному аудио"""
        paths = self.get_cache_paths(video_path)
        
        if os.path.exists(paths['audio']):
            logger.info(f"🎵 Найдено кэшированное аудио: {paths['audio']}")
            return paths['audio']
        
        return None
    
    def save_audio_cache(self, video_path: str, audio_path: str) -> str:
        """Сохраняет аудио в кэш"""
        paths = self.get_cache_paths(video_path)
        cache_audio_path = paths['audio']
        
        # Копируем аудиофайл в кэш
        if audio_path != cache_audio_path:
            import shutil
            shutil.copy2(audio_path, cache_audio_path)
            logger.info(f"💾 Аудио сохранено в кэш: {cache_audio_path}")
        
        return cache_audio_path
    
    def get_cached_transcript(self, video_path: str) -> Optional[Tuple[List[Dict], str]]:
        """Проверяет и возвращает кэшированный транскрипт"""
        paths = self.get_cache_paths(video_path)
        
        if (os.path.exists(paths['transcript_segments']) and 
            os.path.exists(paths['transcript_full'])):
            
            try:
                # Загружаем сегменты
                with open(paths['transcript_segments'], 'r', encoding='utf-8') as f:
                    segments = json.load(f)
                
                # Загружаем полный текст
                with open(paths['transcript_full'], 'r', encoding='utf-8') as f:
                    full_text = f.read()
                
                logger.info(f"📝 Найден кэшированный транскрипт: {len(segments)} сегментов, {len(full_text)} символов")
                return segments, full_text
                
            except Exception as e:
                logger.error(f"Ошибка загрузки транскрипта: {e}")
                return None
        
        return None
    
    def save_transcript_cache(self, video_path: str, segments: List[Dict], full_text: str):
        """Сохраняет транскрипт в кэш"""
        paths = self.get_cache_paths(video_path)
        
        # Сохраняем сегменты
        with open(paths['transcript_segments'], 'w', encoding='utf-8') as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
        
        # Сохраняем полный текст
        with open(paths['transcript_full'], 'w', encoding='utf-8') as f:
            f.write(full_text)
        
        logger.info(f"💾 Транскрипт сохранен в кэш")
    
    def get_cached_analysis(self, video_path: str, analysis_type: str = "basic") -> Optional[str]:
        """Проверяет и возвращает кэшированный анализ"""
        paths = self.get_cache_paths(video_path)
        
        if os.path.exists(paths['analysis']):
            try:
                with open(paths['analysis'], 'r', encoding='utf-8') as f:
                    analysis_data = json.load(f)
                
                if analysis_type in analysis_data:
                    logger.info(f"📊 Найден кэшированный анализ типа '{analysis_type}'")
                    return analysis_data[analysis_type]
                
            except Exception as e:
                logger.error(f"Ошибка загрузки анализа: {e}")
        
        return None
    
    def save_analysis_cache(self, video_path: str, analysis: str, analysis_type: str = "basic"):
        """Сохраняет анализ в кэш"""
        paths = self.get_cache_paths(video_path)
        
        # Загружаем существующие анализы или создаем новый файл
        analysis_data = {}
        if os.path.exists(paths['analysis']):
            try:
                with open(paths['analysis'], 'r', encoding='utf-8') as f:
                    analysis_data = json.load(f)
            except:
                pass
        
        # Добавляем новый анализ
        analysis_data[analysis_type] = analysis
        analysis_data[f"{analysis_type}_timestamp"] = __import__('time').time()
        
        # Сохраняем
        with open(paths['analysis'], 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 Анализ типа '{analysis_type}' сохранен в кэш")
    
    def get_cached_screenshots(self, video_path: str, mode: str) -> Optional[List]:
        """Проверяет и возвращает кэшированные скриншоты"""
        paths = self.get_cache_paths(video_path)
        screenshots_info_file = os.path.join(paths['screenshots'], f"screenshots_{mode}.json")
        
        if os.path.exists(screenshots_info_file):
            try:
                with open(screenshots_info_file, 'r', encoding='utf-8') as f:
                    screenshots_data = json.load(f)
                
                # Проверяем, что все файлы скриншотов существуют
                all_exist = all(
                    os.path.exists(shot.get('path', '')) 
                    for shot in screenshots_data
                )
                
                if all_exist:
                    logger.info(f"📸 Найдены кэшированные скриншоты режима '{mode}': {len(screenshots_data)} файлов")
                    return screenshots_data
                else:
                    logger.warning(f"Некоторые файлы скриншотов отсутствуют, кэш недействителен")
                
            except Exception as e:
                logger.error(f"Ошибка загрузки скриншотов: {e}")
        
        return None
    
    def save_screenshots_cache(self, video_path: str, screenshots: List, mode: str):
        """Сохраняет информацию о скриншотах в кэш"""
        paths = self.get_cache_paths(video_path)
        
        if not os.path.exists(paths['screenshots']):
            os.makedirs(paths['screenshots'])
        
        screenshots_info_file = os.path.join(paths['screenshots'], f"screenshots_{mode}.json")
        
        # Подготавливаем данные для сохранения
        screenshots_data = []
        for shot in screenshots:
            if isinstance(shot, dict):
                screenshots_data.append(shot)
            elif isinstance(shot, (list, tuple)) and len(shot) >= 3:
                screenshots_data.append({
                    'path': shot[0],
                    'timestamp': shot[1],
                    'description': shot[2] if len(shot) > 2 else '',
                    'reason': shot[3] if len(shot) > 3 else 'screenshot'
                })
        
        # Сохраняем информацию
        with open(screenshots_info_file, 'w', encoding='utf-8') as f:
            json.dump(screenshots_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 Информация о скриншотах режима '{mode}' сохранена в кэш")
    
    def get_cache_status(self, video_path: str) -> Dict[str, bool]:
        """Возвращает статус кэшированных данных"""
        paths = self.get_cache_paths(video_path)
        
        status = {
            'audio': os.path.exists(paths['audio']),
            'transcript': (os.path.exists(paths['transcript_segments']) and 
                          os.path.exists(paths['transcript_full'])),
            'analysis': os.path.exists(paths['analysis']),
            'screenshots': os.path.exists(paths['screenshots']),
            'metadata': os.path.exists(paths['metadata'])
        }
        
        return status
    
    def print_cache_status(self, video_path: str):
        """Выводит статус кэша"""
        status = self.get_cache_status(video_path)
        metadata = self.load_metadata(video_path)
        
        print(f"\n🗃️  Статус кэша для видео:")
        print(f"   📱 Аудио: {'✅' if status['audio'] else '❌'}")
        print(f"   📝 Транскрипт: {'✅' if status['transcript'] else '❌'}")
        print(f"   📊 Анализ: {'✅' if status['analysis'] else '❌'}")
        print(f"   📸 Скриншоты: {'✅' if status['screenshots'] else '❌'}")
        
        if metadata:
            import time
            processed_time = time.strftime('%Y-%m-%d %H:%M:%S', 
                                         time.localtime(metadata.get('processed_at', 0)))
            print(f"   ⏰ Последняя обработка: {processed_time}")
            
            if 'whisper_model' in metadata:
                print(f"   🤖 Модель Whisper: {metadata['whisper_model']}")
            
            if 'screenshot_mode' in metadata:
                print(f"   📸 Режим скриншотов: {metadata['screenshot_mode']}")
    
    def clear_cache(self, video_path: str):
        """Очищает кэш для конкретного видео"""
        paths = self.get_cache_paths(video_path)
        
        import shutil
        for cache_type, path in paths.items():
            if os.path.exists(path):
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
                logger.info(f"🧹 Удален кэш: {cache_type}")
    
    def cleanup_old_cache(self, days: int = 7):
        """Удаляет старые файлы кэша"""
        import time
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        
        removed_count = 0
        for filename in os.listdir(self.cache_dir):
            filepath = os.path.join(self.cache_dir, filename)
            if os.path.getmtime(filepath) < cutoff_time:
                if os.path.isdir(filepath):
                    import shutil
                    shutil.rmtree(filepath)
                else:
                    os.remove(filepath)
                removed_count += 1
        
        if removed_count > 0:
            logger.info(f"🧹 Удалено {removed_count} старых файлов кэша")

        return removed_count 