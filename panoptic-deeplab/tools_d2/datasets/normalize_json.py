import json
import pandas as pd
import glob

cnt = 0 
for json_path in glob.glob("./cityscapes_256_512/*/*/*/*.json"):
    cnt += 1
    print(cnt)
    #print(json_path)
    with open(json_path) as json_file:
        json_data = json.load(json_file)
        json_data['imgHeight'] = 256
        json_data['imgWidth'] = 512
        polygon = json_data['objects'][0]['polygon']
        polygon = [[x[0]//4, x[1]//4] for x in polygon]
        json_data['objects'][0]['polygon'] = polygon
    with open(json_path, "w") as json_file:
        json.dump(json_data, json_file, indent=4)