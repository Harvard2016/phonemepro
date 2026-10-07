
import requests
import json
import numpy as np
import scipy.io.wavfile as wav

# Create a dummy WAV file (1 second of silence + 1 second of tone)
# Use a real file from the dataset
real_file = "data/speechocean762/WAVE/SPEAKER7001/070010008.WAV"
url = "http://localhost:8000/api/analyze"

files = {'audio': open(real_file, 'rb')}
data = {
    'word': 'test',
    'phonemes': json.dumps(["T", "EH1", "S", "T"])
}

try:
    print("Sending request...")
    res = requests.post(url, files=files, data=data)
    print(f"Status Code: {res.status_code}")
    print(f"Response: {res.json()}")
except Exception as e:
    print(f"Error: {e}")
