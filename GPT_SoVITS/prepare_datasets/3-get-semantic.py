import os

inp_text = os.environ.get("inp_text")
exp_name = os.environ.get("exp_name")
i_part = os.environ.get("i_part")
all_parts = os.environ.get("all_parts")
if "_CUDA_VISIBLE_DEVICES" in os.environ:
     os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
opt_dir = os.environ.get("opt_dir")
pretrained_s2G = os.environ.get("pretrained_s2G")
s2config_path = os.environ.get("s2config_path")

if os.path.exists(pretrained_s2G):
    print(f"\nИспользуем предобученную модель: {pretrained_s2G}")
else:
    raise FileNotFoundError(f"Не найдена предобученная модель: {pretrained_s2G}")

# Определяем версию модели по размеру файла
size = os.path.getsize(pretrained_s2G)
if size < 82978 * 1024:
    version = "v1"
elif size < 100 * 1024 * 1024:
    version = "v2"
elif size < 103520 * 1024:
    version = "v1"
elif size < 700 * 1024 * 1024:
    version = "v2"
else:
    version = "v3"
print(f"Определена версия модели: {version}")

import torch
is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()
import math, traceback
import multiprocessing
import sys

now_dir = os.getcwd()
sys.path.append(now_dir)
from random import shuffle
import torch.multiprocessing as mp
from glob import glob
from tqdm import tqdm
import logging, librosa, utils
if version != "v3":
    from module.models import SynthesizerTrn
else:
    from module.models import SynthesizerTrnV3 as SynthesizerTrn
from tools.my_utils import clean_path
logging.getLogger("numba").setLevel(logging.WARNING)

hubert_dir = "%s/4-cnhubert" % (opt_dir)
semantic_path = "%s/6-name2semantic-%s.tsv" % (opt_dir, i_part)

print(f"\nПроверяем директории:")
print(f"HuBERT features: {hubert_dir}")
print(f"Выходной файл: {semantic_path}")

if not os.path.exists(semantic_path):
    os.makedirs(opt_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = "cuda"
        print("\nИспользуем GPU")
    else:
        device = "cpu"
        print("\nИспользуем CPU")

    print("\nЗагружаем конфигурацию модели...")
    hps = utils.get_hparams_from_file(s2config_path)
    
    print("Инициализируем модель...")
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        version=version,
        **hps.model
    )
    
    if is_half:
        vq_model = vq_model.half().to(device)
        print("Модель конвертирована в half precision")
    else:
        vq_model = vq_model.to(device)
        print("Модель использует full precision")
    
    vq_model.eval()
    
    print("Загружаем веса модели...")
    print(
        vq_model.load_state_dict(
            torch.load(pretrained_s2G, map_location="cpu")["weight"], strict=False
        )
    )

    def name2go(wav_name, lines):
        print(f"\nОбрабатываем файл: {wav_name}")
        hubert_path = "%s/%s.pt" % (hubert_dir, wav_name)
        
        if not os.path.exists(hubert_path):
            print(f"ПРОПУСК: Не найден файл HuBERT features: {hubert_path}")
            return
            
        print("Загружаем HuBERT features...")
        ssl_content = torch.load(hubert_path, map_location="cpu")
        
        if is_half:
            ssl_content = ssl_content.half().to(device)
        else:
            ssl_content = ssl_content.to(device)
            
        print("Извлекаем семантические признаки...")
        codes = vq_model.extract_latent(ssl_content)
        semantic = " ".join([str(i) for i in codes[0, 0, :].tolist()])
        lines.append("%s\t%s" % (wav_name, semantic))
        print("Файл успешно обработан")

    print("\nЧитаем входной файл с текстом...")
    with open(inp_text, "r", encoding="utf8") as f:
        lines = f.read().strip("\n").split("\n")
    print(f"Прочитано {len(lines)} строк")

    lines1 = []
    for line in lines[int(i_part) :: int(all_parts)]:
        try:
            parts = line.strip().split("\t")
            if len(parts) >= 1:
                wav_name = clean_path(parts[0].strip())
                # Предполагаем, что файлы находятся в поддиректории dima1
                wav_name = os.path.join("dima1", wav_name)
                name2go(wav_name, lines1)
            else:
                print(f"Пропускаем пустую строку: {line}")
        except Exception as e:
            print(f"Ошибка при обработке строки: {line}")
            print(traceback.format_exc())
            
    print(f"\nСохраняем результаты в {semantic_path}...")
    with open(semantic_path, "w", encoding="utf8") as f:
        f.write("\n".join(lines1))
    print("Готово!")
