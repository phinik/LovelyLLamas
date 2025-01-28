import os

li_paths = [os.path.join(os.getcwd(), 'data', 'files_for_chatGPT', '2024-12-12', file) for file in os.listdir(os.path.join(os.getcwd(), 'data', 'files_for_chatGPT', '2024-12-12'))]

# pick 1% of the files randomly using numpy

import numpy as np
np.random.seed(0)
li_paths = np.random.choice(li_paths, int(len(li_paths)*0.01), replace=False)

# copy selected files to a new directory
import shutil
os.makedirs(os.path.join(os.getcwd(), 'demo', 'backend', 'dataset'), exist_ok=True)
for path in li_paths:
    shutil.copy(path, os.path.join(os.getcwd(), 'demo', 'backend', 'dataset'))

# create a json file with the list of files their city name and their ID
import json
li_files = os.listdir(os.path.join(os.getcwd(), 'demo', 'backend', 'dataset'))
d_files = {}

for i, file in enumerate(li_files):
    # read the ciry key from the file content
    with open(os.path.join(os.getcwd(), 'demo', 'backend', 'dataset', file), 'r', encoding='utf-8') as f:
        city = json.load(f)['city']
    d_files[i] = {'file': file, 'city': city, 'id': file.split('_')[0].split('-')[-1]}

with open(os.path.join(os.getcwd(), 'demo', 'backend', 'files.json'), 'w', encoding='utf-8') as f:
    json.dump(d_files, f)