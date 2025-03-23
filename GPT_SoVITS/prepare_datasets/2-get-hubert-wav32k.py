# -*- coding: utf-8 -*-

import sys,os
inp_text=                           os.environ.get("inp_text")
inp_wav_dir=                        os.environ.get("inp_wav_dir")
exp_name=                           os.environ.get("exp_name")
i_part=                             os.environ.get("i_part")
all_parts=                          os.environ.get("all_parts")
if "_CUDA_VISIBLE_DEVICES" in os.environ:
     os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
from feature_extractor import cnhubert
opt_dir=                            os.environ.get("opt_dir")
cnhubert.cnhubert_base_path=                os.environ.get("cnhubert_base_dir")
import torch
is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()

import pdb,traceback,numpy as np,logging
from scipy.io import wavfile
import librosa
now_dir = os.getcwd()
sys.path.append(now_dir)
from simple_utils import load_audio,clean_path

print("\nПроверяем настройки:")
print(f"Входной файл с текстом: {inp_text}")
print(f"Директория с WAV файлами: {inp_wav_dir}")
print(f"Директория для выходных данных: {opt_dir}")
print(f"Директория с моделью CNHubert: {cnhubert.cnhubert_base_path}")

from time import time as ttime
import shutil
def my_save(fea,path):#####fix issue: torch.save doesn't support chinese path
    dir=os.path.dirname(path)
    name=os.path.basename(path)
    # tmp_path="%s/%s%s.pth"%(dir,ttime(),i_part)
    tmp_path="%s%s.pth"%(ttime(),i_part)
    torch.save(fea,tmp_path)
    shutil.move(tmp_path,"%s/%s"%(dir,name))

hubert_dir="%s/4-cnhubert"%(opt_dir)
wav32dir="%s/5-wav32k"%(opt_dir)
os.makedirs(opt_dir,exist_ok=True)
os.makedirs(hubert_dir,exist_ok=True)
os.makedirs(wav32dir,exist_ok=True)

print(f"\nСоздали директории для выходных данных:")
print(f"HuBERT features: {hubert_dir}")
print(f"32kHz WAV files: {wav32dir}")

maxx=0.95
alpha=0.5
if torch.cuda.is_available():
    device = "cuda:0"
    print("\nИспользуем GPU")
else:
    device = "cpu"
    print("\nИспользуем CPU")

print("\nЗагружаем модель CNHubert...")
model=cnhubert.get_model()
if(is_half==True):
    model=model.half().to(device)
    print("Модель конвертирована в half precision")
else:
    model = model.to(device)
    print("Модель использует full precision")

nan_fails=[]
def name2go(wav_name,wav_path):
    print(f"\nОбрабатываем файл: {wav_name}")
    print(f"Полный путь: {wav_path}")
    
    hubert_path="%s/%s.pt"%(hubert_dir,wav_name)
    if(os.path.exists(hubert_path)):
        print("Файл уже обработан, пропускаем")
        return
        
    if not os.path.exists(wav_path):
        print(f"ОШИБКА: Файл не найден: {wav_path}")
        return
        
    print("Загружаем аудио...")
    tmp_audio = load_audio(wav_path, 32000)
    tmp_max = np.abs(tmp_audio).max()
    if tmp_max > 2.2:
        print("%s-filtered,%s" % (wav_name, tmp_max))
        return
    print("Нормализуем аудио...")
    tmp_audio32 = (tmp_audio / tmp_max * (maxx * alpha*32768)) + ((1 - alpha)*32768) * tmp_audio
    tmp_audio32b = (tmp_audio / tmp_max * (maxx * alpha*1145.14)) + ((1 - alpha)*1145.14) * tmp_audio
    print("Ресемплируем в 16kHz...")
    tmp_audio = librosa.resample(
        tmp_audio32b, orig_sr=32000, target_sr=16000
    )
    tensor_wav16 = torch.from_numpy(tmp_audio)
    if (is_half == True):
        tensor_wav16=tensor_wav16.half().to(device)
    else:
        tensor_wav16 = tensor_wav16.to(device)
    print("Извлекаем HuBERT features...")
    ssl=model.model(tensor_wav16.unsqueeze(0))["last_hidden_state"].transpose(1,2).cpu()
    if np.isnan(ssl.detach().numpy()).sum()!= 0:
        nan_fails.append((wav_name,wav_path))
        print("nan filtered:%s"%wav_name)
        return
    
    # Создаем поддиректории в выходных папках, если они не существуют
    speaker_dir = os.path.dirname(wav_name)
    if speaker_dir:
        os.makedirs(os.path.join(wav32dir, speaker_dir), exist_ok=True)
        os.makedirs(os.path.join(hubert_dir, speaker_dir), exist_ok=True)
        print(f"Создали поддиректории для спикера: {speaker_dir}")
    
    print("Сохраняем результаты...")
    wavfile.write(
        "%s/%s"%(wav32dir,wav_name),
        32000,
        tmp_audio32.astype("int16"),
    )
    my_save(ssl,hubert_path)
    print("Файл успешно обработан")

print("\nЧитаем входной файл с текстом...")
with open(inp_text,"r",encoding="utf8")as f:
    lines=f.read().strip("\n").split("\n")
print(f"Прочитано {len(lines)} строк")

for line in lines[int(i_part)::int(all_parts)]:
    try:
        parts = line.split("\t")
        if len(parts) >= 1:  # Нам нужно только имя WAV файла
            wav_name = clean_path(parts[0])
            if (inp_wav_dir != "" and inp_wav_dir != None):
                # Сохраняем полный путь с именем спикера
                speaker_name = "dima1"  # Используем имя спикера из вашей структуры
                wav_name = os.path.join(speaker_name, os.path.basename(wav_name))
                wav_path = os.path.join(inp_wav_dir, wav_name)
            else:
                wav_path=wav_name
                wav_name = os.path.basename(wav_name)
            name2go(wav_name,wav_path)
    except Exception as e:
        print(f"Error processing line: {line}")
        print(traceback.format_exc())

if(len(nan_fails)>0 and is_half==True):
    print("\nПовторная обработка файлов с ошибками NaN в режиме full precision...")
    is_half=False
    model=model.float()
    for wav in nan_fails:
        try:
            name2go(wav[0],wav[1])
        except:
            print(wav_name,traceback.format_exc())
