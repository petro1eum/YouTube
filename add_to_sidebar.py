#!/usr/bin/env python3
"""
CLI утилита для добавления папок в Finder Sidebar на macOS
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def add_to_finder_sidebar(folder_path):
    """
    Добавляет папку в Finder Sidebar используя mysides утилиту
    """
    # Проверяем, что путь существует
    if not os.path.exists(folder_path):
        print(f"❌ Ошибка: Путь '{folder_path}' не существует")
        return False
    
    # Получаем абсолютный путь
    abs_path = os.path.abspath(folder_path)
    
    try:
        # Проверяем, установлена ли mysides
        result = subprocess.run(['which', 'mysides'], capture_output=True, text=True)
        if result.returncode == 0:
            # Добавляем папку в sidebar используя mysides
            result = subprocess.run(
                ['mysides', 'add', os.path.basename(abs_path), f'file://{abs_path}'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(f"✅ Папка '{abs_path}' успешно добавлена в Finder Sidebar")
                return True
            else:
                print(f"❌ Ошибка mysides: {result.stderr}")
                # Fallback к другим способам
                return add_via_python_api(abs_path)
        else:
            print("ℹ️  mysides не найден, используем альтернативные методы...")
            return add_via_python_api(abs_path)
            
    except FileNotFoundError:
        print("ℹ️  mysides не найден, используем Python API...")
        return add_via_python_api(abs_path)
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return add_via_python_api(abs_path)


def add_via_python_api(abs_path):
    """
    Fallback метод через Python API
    """
    try:
        import Foundation
        from Foundation import NSURL
        
        # Создаем URL из пути
        url = NSURL.fileURLWithPath_(abs_path)
        
        # Добавляем в SharedFileList для Sidebar
        from LaunchServices import LSSharedFileListCreate, kLSSharedFileListFavoriteItems, LSSharedFileListInsertItemURL
        
        # Получаем список избранного
        favorite_list = LSSharedFileListCreate(None, kLSSharedFileListFavoriteItems, None)
        
        if favorite_list:
            # Добавляем элемент
            LSSharedFileListInsertItemURL(favorite_list, None, None, None, url, None, None)
            print(f"✅ Папка '{abs_path}' добавлена в Sidebar через Python API")
            return True
        else:
            print("❌ Не удалось получить доступ к списку избранного")
            return False
            
    except ImportError:
        print("❌ PyObjC не установлен. Используем простой AppleScript...")
        return add_via_simple_applescript(abs_path)
    except Exception as e:
        print(f"❌ Ошибка Python API: {e}")
        return add_via_simple_applescript(abs_path)


def add_via_simple_applescript(abs_path):
    """
    Простой AppleScript без проверок
    """
    try:
        # Кодируем путь в base64 чтобы избежать проблем с кириллицей
        import base64
        encoded_path = base64.b64encode(abs_path.encode('utf-8')).decode('ascii')
        
        applescript = f'''
        set encodedPath to "{encoded_path}"
        set decodedPath to do shell script "echo " & quoted form of encodedPath & " | base64 -d"
        
        tell application "Finder"
            try
                set targetPath to POSIX file decodedPath
                make new favorite item at end of sidebar preferences with properties {{target:targetPath}}
                return "success"
            on error
                return "error"
            end try
        end tell
        '''
        
        result = subprocess.run(
            ['osascript', '-e', applescript],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0 and "success" in result.stdout:
            print(f"✅ Папка '{abs_path}' добавлена в Sidebar")
            return True
        else:
            print(f"❌ AppleScript ошибка: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False


def remove_from_finder_sidebar(folder_path):
    """
    Удаляет папку из Finder Sidebar используя AppleScript
    """
    abs_path = os.path.abspath(folder_path)
    
    applescript = f'''
    tell application "Finder"
        try
            set sidebarItems to every item of sidebar preferences
            repeat with anItem in sidebarItems
                if (target of anItem as string) contains "{abs_path}" then
                    delete anItem
                    return "success"
                end if
            end repeat
            return "not_found"
        on error errMsg
            return "error: " & errMsg
        end try
    end tell
    '''
    
    try:
        result = subprocess.run(
            ['osascript', '-e', applescript],
            capture_output=True,
            text=True,
            check=True
        )
        
        output = result.stdout.strip()
        
        if output == "success":
            print(f"✅ Папка '{abs_path}' удалена из Finder Sidebar")
            return True
        elif output == "not_found":
            print(f"ℹ️  Папка '{abs_path}' не найдена в Finder Sidebar")
            return True
        else:
            print(f"❌ Ошибка: {output}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка выполнения AppleScript: {e}")
        return False


def list_sidebar_items():
    """
    Показывает текущие элементы в Finder Sidebar
    """
    applescript = '''
    tell application "Finder"
        try
            set sidebarItems to every item of sidebar preferences
            set itemList to ""
            repeat with anItem in sidebarItems
                set itemList to itemList & (target of anItem as string) & "\\n"
            end repeat
            return itemList
        on error errMsg
            return "error: " & errMsg
        end try
    end tell
    '''
    
    try:
        result = subprocess.run(
            ['osascript', '-e', applescript],
            capture_output=True,
            text=True,
            check=True
        )
        
        output = result.stdout.strip()
        if output.startswith("error:"):
            print(f"❌ Ошибка: {output}")
            return False
        
        print("📂 Текущие элементы в Finder Sidebar:")
        for item in output.split('\n'):
            if item.strip():
                print(f"  • {item}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка выполнения AppleScript: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='CLI утилита для управления Finder Sidebar на macOS',
        epilog='''
Примеры использования:
  %(prog)s add "/Users/username/Documents/MyFolder"
  %(prog)s add "/Users/edcher/Applications/Chrome Apps.localized/Google Диск.app"
  %(prog)s remove "/Users/username/Documents/MyFolder"
  %(prog)s list
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Доступные команды')
    
    # Команда добавления
    add_parser = subparsers.add_parser('add', help='Добавить папку в Finder Sidebar')
    add_parser.add_argument('path', help='Путь к папке для добавления')
    
    # Команда удаления
    remove_parser = subparsers.add_parser('remove', help='Удалить папку из Finder Sidebar')
    remove_parser.add_argument('path', help='Путь к папке для удаления')
    
    # Команда просмотра
    list_parser = subparsers.add_parser('list', help='Показать текущие элементы в Sidebar')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Проверяем, что мы на macOS
    if sys.platform != 'darwin':
        print("❌ Эта утилита работает только на macOS")
        return 1
    
    if args.command == 'add':
        success = add_to_finder_sidebar(args.path)
        return 0 if success else 1
    elif args.command == 'remove':
        success = remove_from_finder_sidebar(args.path)
        return 0 if success else 1
    elif args.command == 'list':
        success = list_sidebar_items()
        return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
