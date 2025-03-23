# -*- coding: utf-8 -*-

import os
import sys

# Добавляем пути к директориям проекта
current_dir = os.path.dirname(os.path.abspath(__file__))
gpt_sovits_dir = os.path.dirname(current_dir)  # /ai/vits/GPT-SoVITS/GPT_SoVITS
project_root = os.path.dirname(gpt_sovits_dir)  # /ai/vits/GPT-SoVITS

# Меняем текущую директорию на GPT_SoVITS, чтобы импорты работали корректно
os.chdir(gpt_sovits_dir)

# Добавляем пути в PYTHONPATH
if gpt_sovits_dir not in sys.path:
    sys.path.insert(0, gpt_sovits_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Отладочный вывод
print("Debug: Python paths:")
for path in sys.path:
    print(f"  - {path}")

# Установка переменных окружения
os.environ["inp_text"] = "/ai/vits/GPT-SoVITS/dataset_raw/transcription.list"
os.environ["inp_wav_dir"] = "/ai/vits/GPT-SoVITS/dataset_raw/wavs"
os.environ["exp_name"] = "dima1"
os.environ["i_part"] = "0"
os.environ["all_parts"] = "1"
os.environ["opt_dir"] = "/ai/vits/GPT-SoVITS/dataset_raw/processed"
os.environ["bert_pretrained_dir"] = "/ai/vits/GPT-SoVITS/GPT_SoVITS/pretrained_models/bert-base-multilingual-cased"

print("\nDebug: Current directory:", os.getcwd())
print("Debug: Script directory:", current_dir)
print("Debug: Project root:", project_root)
print("Debug: GPT-SoVITS directory:", gpt_sovits_dir)

# Импорт и выполнение оригинального скрипта с полным путем
script_path = os.path.join(current_dir, "1-get-text.py")
print("Debug: Loading script:", script_path)
exec(open(script_path).read()) 