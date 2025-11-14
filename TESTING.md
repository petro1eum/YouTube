# Тестирование проекта YouTube Video Analysis

## Обзор

Проект содержит комплексные тесты для проверки работоспособности всех модулей системы анализа видео.

## Структура тестов

```
tests/
├── __init__.py                      # Инициализация пакета тестов
├── test_cache_manager.py            # Unit тесты для CacheManager
├── test_chronological_processor.py  # Unit тесты для ChronologicalTranscriptProcessor
└── test_integration.py              # Integration тесты для всей системы
```

## Установка зависимостей для тестирования

```bash
pip install pytest pytest-cov pytest-mock
```

## Запуск тестов

### Все тесты
```bash
pytest tests/ -v
```

### С покрытием кода
```bash
pytest tests/ --cov=New --cov-report=html
```

### Только unit тесты
```bash
pytest tests/test_cache_manager.py -v
pytest tests/test_chronological_processor.py -v
```

### Только integration тесты
```bash
pytest tests/test_integration.py -v
```

### Базовые тесты без pytest
Если pytest не установлен, можно запустить базовые тесты:
```bash
python run_basic_tests.py
```

## Результаты тестирования

### CacheManager ✅
- ✅ Инициализация
- ✅ Генерация хэша видео
- ✅ Получение путей к кэшу
- ✅ Сохранение/загрузка метаданных
- ✅ Сохранение/загрузка транскрипта
- ✅ Сохранение/загрузка анализа
- ✅ Получение статуса кэша
- ✅ Очистка кэша
- ✅ Удаление старых файлов

### ChronologicalTranscriptProcessor (требует OpenAI API)
- ✅ Инициализация
- ✅ Получение контекста сегмента
- ✅ Создание временной линии
- ✅ Форматирование времени
- ✅ Определение говорящего
- ✅ Анализ темы блока
- ✅ Безопасный парсинг JSON
- ✅ Получение описания события
- ✅ Структуры данных (Speaker, TimelineEvent)

### Integration тесты
- ✅ Полный workflow кэширования
- ✅ Переиспользование кэша
- ✅ Инвалидация кэша
- ✅ Импорт модулей
- ✅ Совместимость форматов данных

## Известные ограничения

1. **Зависимости**: Некоторые тесты требуют установки всех зависимостей из `requirements.txt`
   - numpy
   - opencv-python (cv2)
   - openai
   - и другие

2. **API ключ**: Тесты, требующие OpenAI API, помечены маркером `@pytest.mark.requires_api`

3. **Медленные тесты**: Тесты с реальными API вызовами помечены маркером `@pytest.mark.slow`

## Запуск без зависимостей

Базовый тест `run_basic_tests.py` проверяет:
- ✅ CacheManager (не требует внешних зависимостей)
- ⚠️  Другие модули (требуют numpy, cv2, openai)

## Continuous Integration

Для CI/CD рекомендуется:

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: pytest tests/ -v --cov=New
```

## Добавление новых тестов

1. Создайте файл `test_<module_name>.py` в папке `tests/`
2. Импортируйте модуль для тестирования
3. Создайте класс `Test<ModuleName>`
4. Добавьте методы `test_<function_name>()`
5. Используйте fixtures для подготовки данных

Пример:
```python
import pytest

class TestMyModule:
    @pytest.fixture
    def sample_data(self):
        return {"key": "value"}

    def test_my_function(self, sample_data):
        result = my_function(sample_data)
        assert result == expected_value
```

## Отладка тестов

Для отладки используйте:
```bash
# Подробный вывод
pytest tests/ -vv

# Остановка на первой ошибке
pytest tests/ -x

# Запуск конкретного теста
pytest tests/test_cache_manager.py::TestCacheManager::test_init -v

# Показать print() в тестах
pytest tests/ -s
```

## Покрытие кода

После запуска тестов с `--cov-report=html` откройте:
```bash
open htmlcov/index.html
```

Цель: покрытие >80% для критичных модулей.
