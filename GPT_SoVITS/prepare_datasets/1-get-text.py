# -*- coding: utf-8 -*-

import os

inp_text = os.environ.get("inp_text")
inp_wav_dir = os.environ.get("inp_wav_dir")
exp_name = os.environ.get("exp_name")
i_part = os.environ.get("i_part")
all_parts = os.environ.get("all_parts")
if "_CUDA_VISIBLE_DEVICES" in os.environ:
     os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
opt_dir = os.environ.get("opt_dir")
bert_pretrained_dir = os.environ.get("bert_pretrained_dir")

print("Debug: Input parameters:")
print(f"inp_text: {inp_text}")
print(f"inp_wav_dir: {inp_wav_dir}")
print(f"exp_name: {exp_name}")
print(f"i_part: {i_part}")
print(f"all_parts: {all_parts}")
print(f"opt_dir: {opt_dir}")
print(f"bert_pretrained_dir: {bert_pretrained_dir}")

print("Debug: Importing required modules...")
import torch
print("Debug: torch imported successfully")
is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()
print(f"Debug: is_half = {is_half}, cuda available: {torch.cuda.is_available()}")
version = os.environ.get('version', None)
print(f"Debug: version = {version}")
import sys, numpy as np, traceback, pdb
print("Debug: basic modules imported")
import os.path
from glob import glob
from tqdm import tqdm
try:
    print("Debug: Importing text.cleaner...")
    from text.cleaner import clean_text
    print("Debug: text.cleaner imported successfully")
    print("Debug: Importing transformers...")
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    print("Debug: transformers imported successfully")
except Exception as e:
    print(f"Error during imports: {str(e)}")
    print(traceback.format_exc())
    raise

print("Debug: Importing remaining modules...")
import numpy as np
from tools.simple_utils import clean_path
from time import time as ttime
import shutil

print("Debug: Setting up paths...")
txt_path = "%s/2-name2text-%s.txt" % (opt_dir, i_part)
print(f"Debug: txt_path = {txt_path}")

print("Debug: Checking if output file exists...")
if os.path.exists(txt_path) == False:
    print("Debug: Output file does not exist, proceeding with processing...")
    bert_dir = "%s/3-bert" % (opt_dir)
    print(f"Debug: Creating directories: {opt_dir} and {bert_dir}")
    os.makedirs(opt_dir, exist_ok=True)
    os.makedirs(bert_dir, exist_ok=True)
    print("Debug: Directories created successfully")
    
    print("Debug: Setting up device...")
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    print(f"Debug: Using device: {device}")
    
    print("Debug: Checking BERT model directory...")
    if os.path.exists(bert_pretrained_dir):
        print(f"Debug: BERT model directory exists: {bert_pretrained_dir}")
        print("Debug: Files in BERT directory:")
        print("\n".join(os.listdir(bert_pretrained_dir)))
    else:
        print(f"Error: BERT model directory not found: {bert_pretrained_dir}")
        raise FileNotFoundError(bert_pretrained_dir)
    
    print("Debug: Loading BERT model and tokenizer...")
    try:
        print("Debug: Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(bert_pretrained_dir)
        print("Debug: Tokenizer loaded successfully")
        print("Debug: Loading BERT model...")
        bert_model = AutoModelForMaskedLM.from_pretrained(bert_pretrained_dir)
        print("Debug: BERT model loaded successfully")
        if is_half == True:
            print("Debug: Converting model to half precision...")
            bert_model = bert_model.half().to(device)
        else:
            print("Debug: Moving model to device...")
            bert_model = bert_model.to(device)
        print("Debug: Model prepared successfully")
    except Exception as e:
        print(f"Error loading BERT model: {str(e)}")
        print(traceback.format_exc())
        raise

# inp_text=sys.argv[1]
# inp_wav_dir=sys.argv[2]
# exp_name=sys.argv[3]
# i_part=sys.argv[4]
# all_parts=sys.argv[5]
# os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[6]#i_gpu
# opt_dir="/data/docker/liujing04/gpt-vits/fine_tune_dataset/%s"%exp_name
# bert_pretrained_dir="/data/docker/liujing04/bert-vits2/Bert-VITS2-master20231106/bert/chinese-roberta-wwm-ext-large"

from time import time as ttime
import shutil


def my_save(fea,path):#####fix issue: torch.save doesn't support chinese path
    dir=os.path.dirname(path)
    name=os.path.basename(path)
    # tmp_path="%s/%s%s.pth"%(dir,ttime(),i_part)
    tmp_path="%s%s.pth"%(ttime(),i_part)
    torch.save(fea,tmp_path)
    shutil.move(tmp_path,"%s/%s"%(dir,name))


txt_path = "%s/2-name2text-%s.txt" % (opt_dir, i_part)
if os.path.exists(txt_path) == False:
    bert_dir = "%s/3-bert" % (opt_dir)
    os.makedirs(opt_dir, exist_ok=True)
    os.makedirs(bert_dir, exist_ok=True)
    if torch.cuda.is_available():
        device = "cuda:0"
    # elif torch.backends.mps.is_available():
    #     device = "mps"
    else:
        device = "cpu"
    if os.path.exists(bert_pretrained_dir):...
    else:raise FileNotFoundError(bert_pretrained_dir)
    tokenizer = AutoTokenizer.from_pretrained(bert_pretrained_dir)
    bert_model = AutoModelForMaskedLM.from_pretrained(bert_pretrained_dir)
    if is_half == True:
        bert_model = bert_model.half().to(device)
    else:
        bert_model = bert_model.to(device)

    def get_bert_feature(text, word2ph):
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(device)
            res = bert_model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]

        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)

        phone_level_feature = torch.cat(phone_level_feature, dim=0)

        return phone_level_feature.T

    def process(data, res):
        for name, text, lan in data:
            try:
                name=clean_path(name)
                name = os.path.basename(name)
                print(f"Processing: {name}, text: {text}, language: {lan}")
                phones, word2ph, norm_text = clean_text(
                    text.replace("%", "-").replace("￥", ","), lan, version
                )
                print(f"Cleaned text: {norm_text}")
                print(f"Phones: {phones}")
                print(f"Word2ph: {word2ph}")
                path_bert = "%s/%s.pt" % (bert_dir, name)
                if os.path.exists(path_bert) == False and lan == "zh":
                    bert_feature = get_bert_feature(norm_text, word2ph)
                    assert bert_feature.shape[-1] == len(phones)
                    my_save(bert_feature, path_bert)
                phones = " ".join(phones)
                res.append([name, phones, word2ph, norm_text])
            except Exception as e:
                print(f"Error processing {name}: {str(e)}")
                print(name, text, traceback.format_exc())

    todo = []
    res = []
    print("Reading input file...")
    with open(inp_text, "r", encoding="utf8") as f:
        lines = f.read().strip("\n").split("\n")
    print(f"Read {len(lines)} lines from input file")

    language_v1_to_language_v2 = {
        "ZH": "zh",
        "zh": "zh",
        "JP": "ja",
        "jp": "ja",
        "JA": "ja",
        "ja": "ja",
        "EN": "en",
        "en": "en",
        "En": "en",
        "KO": "ko",
        "Ko": "ko",
        "ko": "ko",
        "yue": "yue",
        "YUE": "yue",
        "Yue": "yue",
        "RU": "ru",
        "ru": "ru",
    }

    for line in lines[int(i_part)::int(all_parts)]:
        try:
            print(f"Processing line: {line}")
            wav_name, spk_name, language, text = line.split("|")
            print(f"Split into: wav_name={wav_name}, spk_name={spk_name}, language={language}, text={text}")
            if language in language_v1_to_language_v2.keys():
                mapped_language = language_v1_to_language_v2.get(language, language)
                print(f"Language {language} mapped to {mapped_language}")
                todo.append([wav_name, text, mapped_language])
            else:
                print(f"\033[33m[Warning] The {language = } of {wav_name} is not supported for training.\033[0m")
        except Exception as e:
            print(f"Error processing line: {line}")
            print(f"Error: {str(e)}")
            print(line, traceback.format_exc())

    print(f"Processing {len(todo)} items...")
    for item in todo:
        print(f"Processing item: {item}")
        try:
            process([item], res)
        except Exception as e:
            print(f"Error processing item {item}: {str(e)}")
            print(traceback.format_exc())
    print(f"Processed {len(res)} items")

    opt = []
    for name, phones, word2ph, norm_text in res:
        print(f"Adding to output: name={name}, phones={phones}, word2ph={word2ph}, norm_text={norm_text}")
        opt.append("%s\t%s\t%s\t%s" % (name, phones, word2ph, norm_text))
    print(f"Writing {len(opt)} items to output file...")

    with open(txt_path, "w", encoding="utf8") as f:
        f.write("\n".join(opt) + "\n")
    print("Done!")
