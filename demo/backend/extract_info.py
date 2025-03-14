import os
import json

dset_path = os.path.join(os.getcwd(), 'demo', 'backend', 'assets', 'dset_test.json')
with open(dset_path, 'r', encoding = 'utf-8') as f:
    dset = [str(file).split('/')[-1] for file in json.load(f)]

# for every file in dset try to open it and extract the city name key
cities = {}
li_cities = os.listdir(os.path.join(os.getcwd(), 'demo', 'backend', 'dataset'))
if 'dset_test.json' in li_cities:
    li_cities.remove('dset_test.json')
for file in dset:
    file_path = os.path.join(os.getcwd(), 'demo', 'backend', 'dataset', file.split('/')[-1])
    if not os.path.exists(file_path):
        continue
    with open(file_path, 'r', encoding = 'utf-8') as f:
        data = json.load(f)
        cities[file.split('/')[-1].split('_')[0].split('-')[-1]] = [data['city'], file, li_cities.index(file.split('/')[-1])]

# save the cities dictionary to a json file
cities_path = os.path.join(os.getcwd(), 'demo', 'backend', 'assets', 'cities.json')
with open(cities_path, 'w', encoding = 'utf-8') as f:
    json.dump(cities, f, ensure_ascii = False, indent = 4)

new_dset = [f'dataset/{val[1]}' for val in cities.values()]
print(new_dset)
# save the new dset to a json file
new_dset_path = os.path.join(os.getcwd(), 'demo', 'backend', 'dataset', 'dset_test.json')
with open(new_dset_path, 'w', encoding = 'utf-8') as f:
    json.dump(new_dset, f, ensure_ascii = False, indent = 4)
