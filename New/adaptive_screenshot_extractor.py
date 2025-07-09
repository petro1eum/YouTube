#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Адаптивный экстрактор скриншотов с обучением на лету
"""

import os
import json
import cv2
import numpy as np
import base64
from openai import OpenAI
from PIL import Image
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from collections import deque
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ScreenshotDecision:
    """Структура для хранения решения о скриншоте"""
    capture: bool
    reason: str
    importance: float
    confidence: float
    visual_features: Dict
    context_match: float

@dataclass
class VideoContext:
    """Контекст видео для адаптации параметров"""
    meeting_type: str
    main_topics: List[str]
    visual_content_probability: float
    recommended_strategy: str
    key_participants: List[str]
    expected_demonstrations: List[str]

class AdaptiveScreenshotExtractor:
    """Адаптивный экстрактор, который учится на основе обратной связи"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
        
        # История решений для обучения
        self.decision_history = deque(maxlen=100)
        
        # Адаптивные параметры
        self.importance_threshold = 0.6
        self.confidence_threshold = 0.7
        self.min_interval = 3.0  # минимальный интервал между скриншотами
        self.check_interval = 2.0  # как часто проверять
        
        # Веса для различных факторов
        self.weights = {
            "visual_change": 1.0,
            "transcript_relevance": 1.2,
            "content_type": 1.1,
            "demonstration_keywords": 1.5,
            "time_since_last": 0.8
        }
        
        # Кэш для оптимизации
        self.frame_cache = deque(maxlen=10)
        self.analysis_cache = {}
        
    def extract_screenshots(self, video_path: str, output_dir: str, 
                           transcript_segments: List[Dict]) -> List[Tuple]:
        """Основной метод извлечения с адаптацией"""
        
        logger.info(f"Начинаем анализ видео: {video_path}")
        
        # Создаем директорию для скриншотов
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Анализируем контекст видео
        video_context = self.analyze_video_context(video_path, transcript_segments)
        logger.info(f"Тип встречи: {video_context.meeting_type}")
        
        # Настраиваем параметры на основе контекста
        self.adjust_parameters(video_context)
        
        # Извлекаем скриншоты с учетом контекста
        screenshots = self.intelligent_extraction(
            video_path, output_dir, transcript_segments, video_context
        )
        
        # Постобработка и оптимизация результатов
        return self.post_process_screenshots(screenshots, video_context)
    
    def analyze_video_context(self, video_path: str, 
                            transcript_segments: List[Dict]) -> VideoContext:
        """Анализирует общий контекст видео"""
        
        # Собираем первые 5 минут транскрипта для анализа
        early_transcript = []
        for segment in transcript_segments:
            if segment['start'] < 300:  # первые 5 минут
                early_transcript.append(segment['text'])
            else:
                break
        
        full_early_text = " ".join(early_transcript)
        
        # Также анализируем несколько кадров видео
        visual_summary = self.analyze_video_sample(video_path)
        
        prompt = f"""Проанализируй видео встречи и определи её контекст.

Транскрипт первых минут:
{full_early_text[:3000]}

Визуальный анализ:
{visual_summary}

Определи:
1. meeting_type: тип встречи (presentation/code_review/discussion/demo/training/workshop)
2. main_topics: основные темы (список из 3-5 тем)
3. visual_content_probability: вероятность демонстрации визуального контента (0.0-1.0)
4. recommended_strategy: рекомендуемая стратегия скриншотов (frequent/moderate/sparse/contextual)
5. key_participants: ключевые участники (если упоминаются)
6. expected_demonstrations: что вероятно будет демонстрироваться (code/slides/documents/diagrams/ui)

Ответь в JSON формате."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            context_data = json.loads(response.choices[0].message.content)
            
            return VideoContext(
                meeting_type=context_data.get('meeting_type', 'discussion'),
                main_topics=context_data.get('main_topics', []),
                visual_content_probability=context_data.get('visual_content_probability', 0.5),
                recommended_strategy=context_data.get('recommended_strategy', 'moderate'),
                key_participants=context_data.get('key_participants', []),
                expected_demonstrations=context_data.get('expected_demonstrations', [])
            )
            
        except Exception as e:
            logger.error(f"Ошибка при анализе контекста: {e}")
            # Возвращаем дефолтный контекст
            return VideoContext(
                meeting_type='discussion',
                main_topics=[],
                visual_content_probability=0.5,
                recommended_strategy='moderate',
                key_participants=[],
                expected_demonstrations=[]
            )
    
    def analyze_video_sample(self, video_path: str) -> str:
        """Анализирует образцы кадров из видео"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Берем 5 кадров равномерно распределенных по видео
        sample_positions = np.linspace(0, total_frames - 1, 5, dtype=int)
        
        visual_info = []
        for pos in sample_positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if ret:
                info = self.analyze_frame_content(frame)
                visual_info.append(info)
        
        cap.release()
        
        # Суммируем информацию
        content_types = [info['content_type'] for info in visual_info]
        most_common = max(set(content_types), key=content_types.count)
        
        return f"Преобладающий тип контента: {most_common}, обнаружены типы: {set(content_types)}"
    
    def adjust_parameters(self, video_context: VideoContext):
        """Настраивает параметры на основе контекста видео"""
        
        # Адаптируем интервалы проверки (УВЕЛИЧИВАЕМ для экономии)
        strategy_intervals = {
            'frequent': 3.0,      # было 1.0
            'moderate': 5.0,      # было 2.0  
            'sparse': 10.0,       # было 5.0
            'contextual': 4.0     # было 1.5
        }
        self.check_interval = strategy_intervals.get(
            video_context.recommended_strategy, 5.0  # было 2.0
        )
        
        # Адаптируем пороги важности
        if video_context.meeting_type in ['presentation', 'demo', 'training']:
            self.importance_threshold = 0.6  # было 0.5 - повышаем порог
            self.weights['content_type'] = 1.3
        elif video_context.meeting_type == 'code_review':
            self.importance_threshold = 0.65  # было 0.55
            self.weights['visual_change'] = 1.2
            self.weights['demonstration_keywords'] = 1.6
        else:  # discussion
            self.importance_threshold = 0.8  # было 0.7 - значительно повышаем
            self.weights['transcript_relevance'] = 1.4
        
        # Учитываем вероятность визуального контента
        if video_context.visual_content_probability > 0.7:
            self.min_interval = 5.0  # было 2.0
        elif video_context.visual_content_probability < 0.3:
            self.min_interval = 10.0  # было 5.0
        else:
            self.min_interval = 7.0   # добавляем средний случай
        
        logger.info(f"ОПТИМИЗИРОВАННЫЕ параметры: interval={self.check_interval}, "
                   f"threshold={self.importance_threshold}, min_interval={self.min_interval}")
    
    def intelligent_extraction(self, video_path: str, output_dir: str,
                             transcript_segments: List[Dict],
                             video_context: VideoContext) -> List[Tuple]:
        """Умное извлечение с учетом контекста"""
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        logger.info(f"Видео: {duration:.1f} сек, {fps:.1f} FPS")
        
        screenshots = []
        frame_count = 0
        last_check = -self.check_interval
        last_screenshot_time = -self.min_interval
        
        # Состояние для отслеживания контекста
        state = {
            "last_content_type": None,
            "demonstration_mode": False,
            "important_section": False,
            "recent_screenshots": deque(maxlen=5),
            "scene_stability": 0,
            "transcript_buffer": deque(maxlen=10)
        }
        
        # Прогресс бар
        progress_interval = int(total_frames / 20)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = frame_count / fps
            
            # Показываем прогресс
            if frame_count % progress_interval == 0:
                progress = (frame_count / total_frames) * 100
                logger.info(f"Прогресс: {progress:.1f}%")
            
            # Обновляем кэш кадров
            self.frame_cache.append((frame.copy(), current_time))
            
            # Проверяем необходимость скриншота
            if current_time - last_check >= self.check_interval:
                # Обновляем состояние транскрипта
                self.update_transcript_state(state, transcript_segments, current_time)
                
                # Принимаем решение
                decision = self.make_screenshot_decision(
                    frame, current_time, transcript_segments,
                    video_context, state
                )
                
                # Логируем решение для обучения
                self.decision_history.append({
                    'time': current_time,
                    'decision': decision,
                    'state': state.copy()
                })
                
                # Проверяем, нужен ли скриншот
                if decision.capture and (current_time - last_screenshot_time) >= self.min_interval:
                    # Выбираем лучший кадр из кэша
                    best_frame, best_time = self.select_best_frame(decision)
                    
                    # Сохраняем скриншот
                    screenshot_path = self.save_screenshot(
                        best_frame, best_time, output_dir, decision
                    )
                    
                    # Генерируем описание
                    description = self.generate_contextual_description(
                        screenshot_path, best_time,
                        transcript_segments, video_context,
                        decision
                    )
                    
                    screenshots.append({
                        'path': screenshot_path,
                        'timestamp': best_time,
                        'description': description,
                        'decision': decision,
                        'context': video_context.meeting_type
                    })
                    
                    # Обновляем состояние
                    state['recent_screenshots'].append(best_time)
                    last_screenshot_time = best_time
                    
                    # Обучаемся на решении
                    self.learn_from_decision(decision, True)
                    
                    logger.info(f"📸 {best_time:.1f}с: {decision.reason} "
                              f"(важность: {decision.importance:.2f})")
                
                last_check = current_time
            
            frame_count += 1
        
        cap.release()
        logger.info(f"Извлечено {len(screenshots)} скриншотов")
        
        return screenshots
    
    def make_screenshot_decision(self, frame: np.ndarray, current_time: float,
                               transcript_segments: List[Dict],
                               video_context: VideoContext,
                               state: Dict) -> ScreenshotDecision:
        """Принимает решение о необходимости скриншота"""
        
        # Анализируем визуальные характеристики
        visual_features = self.analyze_frame_content(frame)
        
        # Получаем релевантный контекст транскрипта
        transcript_context = self.get_transcript_context(
            transcript_segments, current_time, window=10
        )
        
        # Вычисляем различные факторы важности
        factors = self.calculate_importance_factors(
            visual_features, transcript_context, state, 
            current_time, video_context
        )
        
        # Используем AI для финального решения
        prompt = f"""Анализируй момент видео и реши, нужен ли скриншот.

Время: {current_time:.1f} сек
Тип встречи: {video_context.meeting_type}

Визуальная информация:
- Тип контента: {visual_features['content_type']}
- Изменение сцены: {visual_features['scene_change']:.2f}
- Количество текста: {visual_features['text_amount']}
- Сложность изображения: {visual_features['complexity']:.2f}

Контекст транскрипта:
{transcript_context['text'][:500]}

Ключевые слова: {transcript_context['keywords']}
Демонстрация: {state.get('demonstration_mode', False)}

Факторы важности:
{json.dumps(factors, indent=2)}

Последний скриншот: {state['recent_screenshots'][-1] if state['recent_screenshots'] else 'нет'}

Нужен ли скриншот? Ответь в JSON:
{{
    "capture": true/false,
    "reason": "краткое объяснение",
    "importance": 0.0-1.0,
    "confidence": 0.0-1.0
}}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.2
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Применяем адаптивные веса
            weighted_importance = self.apply_weights(result['importance'], factors)
            
            return ScreenshotDecision(
                capture=result['capture'] and weighted_importance >= self.importance_threshold,
                reason=result['reason'],
                importance=weighted_importance,
                confidence=result.get('confidence', 0.8),
                visual_features=visual_features,
                context_match=factors.get('context_match', 0.5)
            )
            
        except Exception as e:
            logger.error(f"Ошибка при принятии решения: {e}")
            # Fallback на эвристику
            return self.heuristic_decision(visual_features, transcript_context, state)
    
    def analyze_frame_content(self, frame: np.ndarray) -> Dict:
        """Анализирует содержимое кадра"""
        
        # Конвертируем в grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Детекция краев для определения сложности
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Анализ цветов
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation_mean = np.mean(hsv[:, :, 1])
        value_mean = np.mean(hsv[:, :, 2])
        
        # Определение типа контента
        content_type = self.detect_content_type(frame, edge_density, saturation_mean)
        
        # Детекция изменений (если есть предыдущий кадр)
        scene_change = 0.0
        if len(self.frame_cache) > 1:
            prev_frame = self.frame_cache[-2][0]
            scene_change = self.calculate_scene_change(prev_frame, frame)
        
        # Простая оценка количества текста
        text_amount = self.estimate_text_amount(edges, edge_density)
        
        return {
            'content_type': content_type,
            'scene_change': scene_change,
            'text_amount': text_amount,
            'complexity': edge_density,
            'saturation': saturation_mean,
            'brightness': value_mean
        }
    
    def detect_content_type(self, frame: np.ndarray, edge_density: float, 
                          saturation: float) -> str:
        """Определяет тип контента на экране"""
        
        # Простые эвристики
        if edge_density > 0.15:
            if saturation < 30:
                return "code_editor"
            else:
                return "text_document"
        elif edge_density > 0.08:
            if saturation > 100:
                return "presentation_slide"
            else:
                return "diagram"
        elif edge_density < 0.05:
            if saturation > 50:
                return "video_share"
            else:
                return "blank_screen"
        else:
            return "mixed_content"
    
    def calculate_scene_change(self, prev_frame: np.ndarray, 
                              curr_frame: np.ndarray) -> float:
        """Вычисляет степень изменения между кадрами"""
        
        # Используем несколько метрик
        # 1. Простая разница
        diff = cv2.absdiff(prev_frame, curr_frame)
        mean_diff = np.mean(diff) / 255.0
        
        # 2. Гистограммы
        hist1 = cv2.calcHist([prev_frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([curr_frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()
        hist_corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        # Комбинируем метрики
        scene_change = mean_diff * 0.5 + (1 - hist_corr) * 0.5
        
        return min(scene_change, 1.0)
    
    def estimate_text_amount(self, edges: np.ndarray, edge_density: float) -> str:
        """Оценивает количество текста на изображении"""
        
        if edge_density > 0.12:
            return "high"
        elif edge_density > 0.06:
            return "medium"
        elif edge_density > 0.02:
            return "low"
        else:
            return "none"
    
    def get_transcript_context(self, transcript_segments: List[Dict],
                             current_time: float, window: float = 10) -> Dict:
        """Получает контекст транскрипта вокруг текущего времени"""
        
        relevant_segments = []
        keywords = []
        
        # Ключевые слова для детекции важных моментов
        important_keywords = [
            'покажу', 'показываю', 'смотрите', 'видите', 'вот здесь',
            'на экране', 'демонстрация', 'пример', 'давайте посмотрим',
            'обратите внимание', 'важно', 'главное', 'ключевой момент',
            'презентация', 'слайд', 'код', 'диаграмма', 'схема',
            'результат', 'итог', 'вывод', 'заключение'
        ]
        
        for segment in transcript_segments:
            # Проверяем, попадает ли сегмент в временное окно
            if (segment['start'] >= current_time - window and 
                segment['start'] <= current_time + window/2):
                relevant_segments.append(segment)
                
                # Ищем ключевые слова
                text_lower = segment['text'].lower()
                for keyword in important_keywords:
                    if keyword in text_lower:
                        keywords.append(keyword)
        
        text = ' '.join([s['text'] for s in relevant_segments])
        
        return {
            'text': text,
            'segments': relevant_segments,
            'keywords': list(set(keywords)),
            'has_demonstration_keywords': len(keywords) > 0
        }
    
    def calculate_importance_factors(self, visual_features: Dict,
                                   transcript_context: Dict,
                                   state: Dict, current_time: float,
                                   video_context: VideoContext) -> Dict:
        """Вычисляет различные факторы важности"""
        
        factors = {}
        
        # Визуальные факторы
        factors['visual_change'] = visual_features['scene_change']
        factors['content_relevance'] = 1.0 if visual_features['content_type'] in [
            'presentation_slide', 'code_editor', 'diagram'
        ] else 0.5
        
        # Транскрипт факторы
        factors['has_keywords'] = 1.0 if transcript_context['has_demonstration_keywords'] else 0.0
        factors['transcript_density'] = min(len(transcript_context['text']) / 500, 1.0)
        
        # Временные факторы
        if state['recent_screenshots']:
            time_since_last = current_time - state['recent_screenshots'][-1]
            factors['time_factor'] = min(time_since_last / 30, 1.0)  # нормализуем к 30 сек
        else:
            factors['time_factor'] = 1.0
        
        # Контекстные факторы
        factors['context_match'] = self.calculate_context_match(
            visual_features, video_context
        )
        
        # Факторы состояния
        factors['demonstration_mode'] = 1.0 if state.get('demonstration_mode') else 0.3
        
        return factors
    
    def calculate_context_match(self, visual_features: Dict,
                              video_context: VideoContext) -> float:
        """Вычисляет соответствие визуального контента ожиданиям"""
        
        content_type = visual_features['content_type']
        expected = video_context.expected_demonstrations
        
        match_score = 0.0
        
        if 'code' in expected and content_type == 'code_editor':
            match_score = 1.0
        elif 'slides' in expected and content_type == 'presentation_slide':
            match_score = 1.0
        elif 'documents' in expected and content_type == 'text_document':
            match_score = 0.9
        elif 'diagrams' in expected and content_type == 'diagram':
            match_score = 0.95
        elif 'ui' in expected and content_type in ['mixed_content', 'video_share']:
            match_score = 0.8
        else:
            match_score = 0.3
        
        return match_score
    
    def apply_weights(self, base_importance: float, factors: Dict) -> float:
        """Применяет веса к факторам важности"""
        
        weighted_sum = 0.0
        weight_sum = 0.0
        
        for factor_name, factor_value in factors.items():
            if factor_name in self.weights:
                weight = self.weights[factor_name]
                weighted_sum += factor_value * weight
                weight_sum += weight
        
        if weight_sum > 0:
            # Комбинируем базовую важность с взвешенными факторами
            weighted_importance = (base_importance * 0.6 + 
                                 (weighted_sum / weight_sum) * 0.4)
        else:
            weighted_importance = base_importance
        
        return min(weighted_importance, 1.0)
    
    def heuristic_decision(self, visual_features: Dict,
                         transcript_context: Dict,
                         state: Dict) -> ScreenshotDecision:
        """Эвристическое решение как fallback"""
        
        # Простые правила
        capture = False
        reason = "no significant change"
        importance = 0.3
        
        if visual_features['scene_change'] > 0.5:
            capture = True
            reason = "significant scene change"
            importance = 0.7
        elif transcript_context['has_demonstration_keywords']:
            capture = True
            reason = "demonstration keywords detected"
            importance = 0.8
        elif visual_features['content_type'] in ['presentation_slide', 'code_editor']:
            if visual_features['scene_change'] > 0.2:
                capture = True
                reason = f"new {visual_features['content_type']}"
                importance = 0.75
        
        return ScreenshotDecision(
            capture=capture,
            reason=reason,
            importance=importance,
            confidence=0.6,
            visual_features=visual_features,
            context_match=0.5
        )
    
    def update_transcript_state(self, state: Dict, transcript_segments: List[Dict],
                              current_time: float):
        """Обновляет состояние на основе транскрипта"""
        
        # Обновляем буфер транскрипта
        recent_segments = []
        for segment in transcript_segments:
            if segment['start'] >= current_time - 20 and segment['start'] <= current_time:
                recent_segments.append(segment['text'])
        
        state['transcript_buffer'] = deque(recent_segments[-10:], maxlen=10)
        
        # Проверяем режим демонстрации
        demo_keywords = ['покажу', 'демонстрация', 'пример', 'смотрите', 'на экране']
        recent_text = ' '.join(state['transcript_buffer']).lower()
        
        state['demonstration_mode'] = any(kw in recent_text for kw in demo_keywords)
        
        # Определяем важность секции
        importance_keywords = ['важно', 'ключевой', 'главное', 'основной', 'критично']
        state['important_section'] = any(kw in recent_text for kw in importance_keywords)
    
    def select_best_frame(self, decision: ScreenshotDecision) -> Tuple[np.ndarray, float]:
        """Выбирает лучший кадр из кэша"""
        
        if not self.frame_cache:
            return None, 0.0
        
        # Если есть только один кадр
        if len(self.frame_cache) == 1:
            return self.frame_cache[0]
        
        # Анализируем качество кадров
        best_score = -1
        best_frame = None
        best_time = 0
        
        for frame, timestamp in self.frame_cache:
            # Оцениваем качество кадра
            score = self.evaluate_frame_quality(frame)
            
            # Учитываем близость к моменту решения
            if decision.visual_features.get('scene_change', 0) > 0.3:
                # Для изменений сцены берем более поздний кадр
                time_weight = timestamp / self.frame_cache[-1][1]
            else:
                # Для стабильных сцен берем средний кадр
                time_weight = 1.0 - abs(0.5 - timestamp / self.frame_cache[-1][1])
            
            final_score = score * 0.7 + time_weight * 0.3
            
            if final_score > best_score:
                best_score = final_score
                best_frame = frame
                best_time = timestamp
        
        return best_frame, best_time
    
    def evaluate_frame_quality(self, frame: np.ndarray) -> float:
        """Оценивает качество кадра"""
        
        # Проверка на размытие
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = min(laplacian_var / 1000, 1.0)
        
        # Проверка на яркость
        brightness = np.mean(gray) / 255
        brightness_score = 1.0 - abs(brightness - 0.5) * 2
        
        # Проверка на контраст
        contrast = gray.std() / 128
        contrast_score = min(contrast, 1.0)
        
        # Комбинированная оценка
        quality = blur_score * 0.5 + brightness_score * 0.25 + contrast_score * 0.25
        
        return quality
    
    def save_screenshot(self, frame: np.ndarray, timestamp: float,
                       output_dir: str, decision: ScreenshotDecision) -> str:
        """Сохраняет скриншот с метаданными"""
        
        # Формируем имя файла
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)
        milliseconds = int((timestamp % 1) * 1000)
        
        # Добавляем информацию о причине в имя файла
        reason_short = decision.reason.replace(' ', '_')[:20]
        filename = f"screenshot_{minutes:02d}m{seconds:02d}s{milliseconds:03d}ms_{reason_short}.jpg"
        filepath = os.path.join(output_dir, filename)
        
        # Сохраняем с высоким качеством
        cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Сохраняем метаданные
        metadata = {
            'timestamp': timestamp,
            'decision': {
                'reason': decision.reason,
                'importance': decision.importance,
                'confidence': decision.confidence
            },
            'visual_features': decision.visual_features
        }
        
        metadata_path = filepath.replace('.jpg', '_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def generate_contextual_description(self, screenshot_path: str,
                                      timestamp: float,
                                      transcript_segments: List[Dict],
                                      video_context: VideoContext,
                                      decision: ScreenshotDecision) -> str:
        """Генерирует контекстное описание скриншота"""
        
        # Получаем контекст транскрипта
        transcript_context = self.get_transcript_context(
            transcript_segments, timestamp, window=15
        )
        
        # Кодируем изображение для GPT-4V
        with open(screenshot_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        prompt = f"""Опиши этот скриншот из {video_context.meeting_type} встречи.

Контекст:
- Время: {timestamp:.1f} сек ({int(timestamp//60)}:{int(timestamp%60):02d})
- Причина скриншота: {decision.reason}
- Тип контента: {decision.visual_features.get('content_type', 'unknown')}
- Основные темы встречи: {', '.join(video_context.main_topics[:3])}

Транскрипт около этого момента:
{transcript_context['text'][:800]}

Создай информативное описание, которое:
1. Описывает что видно на экране
2. Связывает визуальное содержимое с контекстом разговора
3. Выделяет ключевую информацию
4. Объясняет важность этого момента

Ответ должен быть кратким (2-4 предложения), но информативным."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "low"
                            }
                        }
                    ]
                }],
                max_tokens=300,
                temperature=0.3
            )
            
            description = response.choices[0].message.content.strip()
            
            # Добавляем контекстную информацию
            if decision.importance > 0.8:
                description = f"⭐ Важный момент: {description}"
            elif decision.visual_features.get('content_type') == 'code_editor':
                description = f"💻 Код: {description}"
            elif decision.visual_features.get('content_type') == 'presentation_slide':
                description = f"📊 Слайд: {description}"
            
            return description
            
        except Exception as e:
            logger.error(f"Ошибка при генерации описания: {e}")
            return f"Скриншот в {timestamp:.1f}с - {decision.reason}"
    
    def learn_from_decision(self, decision: ScreenshotDecision, was_captured: bool):
        """Обучается на основе принятого решения"""
        
        # Анализируем историю решений
        if len(self.decision_history) < 10:
            return
        
        # Считаем статистику по последним решениям
        recent_decisions = list(self.decision_history)[-20:]
        capture_rate = sum(1 for d in recent_decisions if d['decision'].capture) / len(recent_decisions)
        
        # Адаптируем пороги
        if capture_rate > 0.7:  # Слишком много скриншотов
            self.importance_threshold = min(self.importance_threshold * 1.05, 0.9)
            logger.debug(f"Повышаем порог важности до {self.importance_threshold:.2f}")
        elif capture_rate < 0.2:  # Слишком мало скриншотов
            self.importance_threshold = max(self.importance_threshold * 0.95, 0.4)
            logger.debug(f"Понижаем порог важности до {self.importance_threshold:.2f}")
        
        # Адаптируем веса на основе успешных решений
        if was_captured and decision.importance > 0.7:
            # Усиливаем факторы, которые привели к хорошему решению
            for factor_name in ['visual_change', 'transcript_relevance', 'content_type']:
                if factor_name in self.weights:
                    self.weights[factor_name] *= 1.02
                    self.weights[factor_name] = min(self.weights[factor_name], 2.0)
    
    def post_process_screenshots(self, screenshots: List[Dict],
                               video_context: VideoContext) -> List[Tuple]:
        """Постобработка и оптимизация результатов"""
        
        if not screenshots:
            return []
        
        logger.info("Постобработка скриншотов...")
        
        # Удаляем слишком близкие дубликаты
        filtered = []
        last_time = -self.min_interval
        
        for screenshot in screenshots:
            if screenshot['timestamp'] - last_time >= self.min_interval * 0.8:
                filtered.append(screenshot)
                last_time = screenshot['timestamp']
            else:
                # Проверяем, не важнее ли этот скриншот предыдущего
                if (screenshot['decision'].importance > 
                    filtered[-1]['decision'].importance * 1.2):
                    # Заменяем предыдущий
                    filtered[-1] = screenshot
                    logger.debug(f"Заменяем скриншот на более важный: "
                               f"{screenshot['timestamp']:.1f}с")
        
        # Анализируем распределение
        if len(filtered) > 5:
            # Проверяем равномерность распределения
            timestamps = [s['timestamp'] for s in filtered]
            intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
            avg_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            
            logger.info(f"Распределение скриншотов: avg={avg_interval:.1f}с, "
                       f"std={std_interval:.1f}с")
        
        # Преобразуем в нужный формат
        result = []
        for screenshot in filtered:
            result.append((
                screenshot['path'],
                screenshot['timestamp'],
                screenshot['description'],
                screenshot['decision'].reason
            ))
        
        logger.info(f"После постобработки: {len(result)} скриншотов")
        
        return result

# Вспомогательные функции для интеграции с основным скриптом

def integrate_adaptive_extractor(video_path: str, output_dir: str,
                               transcript_segments: List[Dict],
                               api_key: str) -> List[Tuple]:
    """Интегрирует адаптивный экстрактор в основной pipeline"""
    
    extractor = AdaptiveScreenshotExtractor(api_key)
    screenshots = extractor.extract_screenshots(
        video_path, output_dir, transcript_segments
    )
    
    return screenshots

# Пример использования
if __name__ == "__main__":
    # Тестовый запуск
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python adaptive_screenshot_extractor.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("Ошибка: Необходим OPENAI_API_KEY")
        sys.exit(1)
    
    # Заглушка для транскрипта
    dummy_transcript = [
        {"text": "Давайте посмотрим на презентацию", "start": 10.0, "duration": 3.0},
        {"text": "Вот здесь показан важный код", "start": 30.0, "duration": 4.0},
    ]
    
    extractor = AdaptiveScreenshotExtractor(api_key)
    screenshots = extractor.extract_screenshots(
        video_path, "output", dummy_transcript
    )
    
    print(f"\nИзвлечено {len(screenshots)} скриншотов")
    for i, (path, time, desc, reason) in enumerate(screenshots):
        print(f"{i+1}. {time:.1f}с - {reason}")
        print(f"   {desc}\n")
