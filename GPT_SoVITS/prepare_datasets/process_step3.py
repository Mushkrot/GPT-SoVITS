import os
import sys

# Устанавливаем текущую директорию и добавляем пути в PYTHONPATH
current_dir = os.getcwd()
project_root = os.path.dirname(os.path.dirname(current_dir))
gpt_sovits_dir = os.path.join(project_root, "GPT_SoVITS")
parent_dir = os.path.dirname(project_root)

# Добавляем пути в PYTHONPATH
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if gpt_sovits_dir not in sys.path:
    sys.path.insert(0, gpt_sovits_dir)

# Устанавливаем переменные окружения
os.environ["inp_text"] = os.path.join(project_root, "dataset_raw", "processed", "2-name2text-0.txt")
os.environ["exp_name"] = "default"
os.environ["i_part"] = "0"
os.environ["all_parts"] = "1"
os.environ["opt_dir"] = os.path.join(project_root, "dataset_raw", "processed")
os.environ["pretrained_s2G"] = os.path.join(gpt_sovits_dir, "pretrained_models", "s2", "G_latest.pth")
os.environ["s2config_path"] = os.path.join(gpt_sovits_dir, "configs", "s2.json")
os.environ["is_half"] = "True"

# Выводим информацию о путях
print(f"Текущая директория: {current_dir}")
print(f"Корень проекта: {project_root}")
print(f"Директория GPT-SoVITS: {gpt_sovits_dir}")
print(f"Входной файл с текстом: {os.environ['inp_text']}")
print(f"Выходная директория: {os.environ['opt_dir']}")
print(f"Предобученная модель: {os.environ['pretrained_s2G']}")
print(f"Конфигурационный файл: {os.environ['s2config_path']}")
print(f"PYTHONPATH: {sys.path}")

# Запускаем скрипт третьего этапа
os.chdir(gpt_sovits_dir)  # Меняем директорию для правильной работы импортов
script_path = os.path.join(gpt_sovits_dir, "prepare_datasets", "3-get-semantic.py")
with open(script_path, 'r', encoding='utf-8') as f:
    exec(f.read()) 