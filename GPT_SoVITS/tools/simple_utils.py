import librosa
import numpy as np

def clean_path(path):
    return path.strip().replace("\\", "/").replace("//", "/")

def load_audio(file, sr):
    try:
        x = librosa.load(file, sr=sr)[0]
    except:
        # Если не удалось загрузить файл через librosa, пробуем через scipy
        import soundfile as sf
        x, sr = sf.read(file)
        if len(x.shape) > 1:
            x = x.mean(1)
        x = librosa.resample(x, orig_sr=sr, target_sr=sr)
    return x
