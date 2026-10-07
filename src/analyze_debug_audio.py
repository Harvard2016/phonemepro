
import librosa
import numpy as np
import scipy.io.wavfile as wav

try:
    y, sr = librosa.load("debug_last_upload.wav", sr=16000)
    print(f"Sample Rate: {sr}")
    print(f"Duration: {librosa.get_duration(y=y, sr=sr)}s")
    print(f"Min: {np.min(y)}, Max: {np.max(y)}")
    print(f"RMS: {np.sqrt(np.mean(y**2))}")
    print(f"Silent? {np.max(np.abs(y)) < 0.01}")
    
    # Check for DC offset
    print(f"Mean (DC Offset): {np.mean(y)}")
except Exception as e:
    print(f"Error analyzing: {e}")
