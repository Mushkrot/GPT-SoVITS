import os
import sys

# Добавляем корневую директорию проекта в PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
gpt_sovits_dir = os.path.join(project_root, "GPT_SoVITS")

# Добавляем необходимые пути в PYTHONPATH
sys.path.insert(0, current_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, gpt_sovits_dir)
sys.path.insert(0, os.path.join(gpt_sovits_dir, "tools"))

# Меняем текущую директорию на GPT_SoVITS для правильной работы импортов
os.chdir(gpt_sovits_dir)

# Устанавливаем переменные окружения
os.environ["inp_text"] = os.path.join(project_root, "dataset_raw", "processed", "2-name2text-0.txt")
os.environ["inp_wav_dir"] = os.path.join(project_root, "dataset_raw", "wavs")  # Указываем корневую директорию с WAV файлами
os.environ["exp_name"] = "my_voice"
os.environ["i_part"] = "0"
os.environ["all_parts"] = "1"
os.environ["opt_dir"] = os.path.join(project_root, "dataset_raw", "processed")
os.environ["cnhubert_base_dir"] = os.path.join(gpt_sovits_dir, "pretrained_models", "hubert")

# Выводим информацию о путях для отладки
print("Current directory:", os.getcwd())
print("Project root:", project_root)
print("GPT-SoVITS directory:", gpt_sovits_dir)
print("Input text file:", os.environ["inp_text"])
print("Input WAV directory:", os.environ["inp_wav_dir"])
print("Output directory:", os.environ["opt_dir"])
print("CNHubert base directory:", os.environ["cnhubert_base_dir"])
print("Python path:", sys.path)

# Запускаем скрипт второго этапа
script_path = os.path.join(current_dir, "2-get-hubert-wav32k.py")
print("\nStarting second stage processing...")
exec(open(script_path).read()) 