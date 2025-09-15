# 🎬 YouTube Video Analysis & Meeting Transcript System

Комплексная система для анализа видео с YouTube и локальных файлов, включая интеллектуальную обработку транскриптов встреч и извлечение кода из видео.

## 🚀 Основные возможности

### 📊 **Новая интегрированная система** (рекомендуется)
- **Хронологический анализ встреч** с определением участников
- **Умное извлечение скриншотов** на основе содержания
- **Коррекция транскриптов** с учетом контекста
- **Интеграция визуальных элементов** с речью
- **Кэширование результатов** для быстрой повторной обработки

### 📹 **Классический анализ YouTube**
- Извлечение транскриптов с YouTube
- Загрузка аудио и видео контента
- Анализ с помощью OpenAI GPT
- Извлечение кода из кадров видео
- Сохранение в Markdown и JSON форматах

## 📁 Структура проекта

```
YouTube/
├── New/                          # 🆕 Новая интегрированная система
│   ├── updated-video-analyzer.py # Главный скрипт для анализа
│   ├── quick_analyze.py          # Быстрый анализ нескольких файлов
│   ├── chronological_transcript_processor.py # Хронологический процессор
│   ├── adaptive_screenshot_extractor.py      # Умные скриншоты
│   ├── smart_transcript_extractor.py         # Умная обработка транскриптов
│   ├── cache_manager.py          # Система кэширования
│   └── README_INTEGRATED.md      # Подробная документация новой системы
├── youtube.py                    # Классический анализ YouTube (устарел)
├── get_transcript.py            # Простое извлечение транскриптов (устарел)
├── download_youtube.sh          # Bash-скрипт для загрузки через yt-dlp
└── README.md                    # Этот файл
```

## 🛠️ Установка

### 1. Клонирование и настройка окружения
```bash
git clone https://github.com/petro1eum/YouTube.git
cd YouTube
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 2. Установка зависимостей
```bash
pip install openai whisper opencv-python numpy pillow requests python-dotenv
# Для работы с YouTube (если нужно):
pip install pytubefix youtube-transcript-api
# Для OCR (опционально):
pip install pytesseract
```

### 3. Настройка API ключа
Создайте файл `.env` в корневой папке:
```bash
OPENAI_API_KEY=your_api_key_here
```

### 4. Установка FFmpeg (для извлечения аудио)
- **macOS**: `brew install ffmpeg`
- **Ubuntu**: `sudo apt install ffmpeg`  
- **Windows**: Скачайте с [ffmpeg.org](https://ffmpeg.org/)

## 🎯 Использование

### 🆕 **Рекомендуется: Новая система**

#### Полный анализ встречи с хронологией:
```bash
cd New
python updated-video-analyzer.py meeting.mp4 --chronological
```

#### Быстрый анализ нескольких файлов:
```bash
cd New  
python quick_analyze.py *.mp4 --full
```

#### Основные параметры:
- `--chronological` - Создать интегрированный хронологический отчет ⭐
- `--whisper-model MODEL` - Модель Whisper (tiny/base/small/medium/large)
- `--screenshot-mode MODE` - Режим скриншотов (smart/periodic/both)
- `--output DIR` - Папка для результатов

### 📹 **Классические скрипты** (для YouTube)

#### Простое извлечение транскрипта:
```bash
python get_transcript.py https://www.youtube.com/watch?v=VIDEO_ID
```

#### Комплексный анализ с извлечением кода:
```bash
python youtube.py https://www.youtube.com/watch?v=VIDEO_ID --extract-code --whisper-model base
```

#### Загрузка видео в максимальном качестве:
```bash
./download_youtube.sh https://www.youtube.com/watch?v=VIDEO_ID
```

## 📊 Что получается на выходе

### 🆕 **Новая система**
```
results/
├── video_name_INTEGRATED_chronological.md    # 🎯 Главный отчет
├── video_name_integrated_data.json           # Структурированные данные  
└── video_name_screenshots/                   # Папка со скриншотами
    ├── screenshot_001_presentation_start.jpg
    └── screenshot_002_diagram_shown.jpg
```

### 📹 **Классическая система**
```
results/
├── VIDEO_ID_transcript.md     # Транскрипт и анализ
├── VIDEO_ID_analysis.json     # JSON с результатами
└── code_frames/               # Извлеченный код (если использовался --extract-code)
```

## 🔄 Миграция со старой системы

**Устаревшие файлы** (можно удалить):
- `youtube_improved.py` - заменен новой системой
- `analyze_text.py` - функциональность включена в новые скрипты

**Актуальные файлы**:
- `New/` - основная рабочая система ✅
- `youtube.py` - для работы с YouTube (при необходимости)
- `download_youtube.sh` - для ручной загрузки видео

## 🎯 Рекомендации по использованию

### Для анализа встреч и презентаций:
```bash
cd New
python updated-video-analyzer.py meeting.mp4 --chronological --whisper-model base
```

### Для извлечения кода из YouTube:
```bash
python youtube.py https://youtube.com/watch?v=ID --extract-code --interval 0.5
```

### Для быстрой обработки множества файлов:
```bash
cd New
python quick_analyze.py /path/to/videos/*.mp4 --standard
```

## 🛠️ Устранение неполадок

- **Ошибки PyTube**: Переустановите `pytubefix`
- **Проблемы с API**: Проверьте `.env` файл и ключ OpenAI
- **OCR не работает**: Установите Tesseract
- **Медленная работа**: Используйте `--whisper-model tiny` для быстрого анализа

## 📚 Дополнительная документация

- [Подробное руководство по новой системе](New/README_INTEGRATED.md)
- [Примеры использования](New/test_chronological_output.md)

## 📄 Лицензия

Проект доступен для личного и образовательного использования.

---

💡 **Совет**: Начните с новой системы в папке `New/` - она предоставляет более мощные возможности анализа!