import json
import requests

def download_image(data):
    '''
    make a folder with name "images" in directory first
    or uncomment the filename line if you don't want to make a folder.
    '''
    for i in range(len(data["images"])):
        url = data["images"][i]["url"]
        filename = "./{}".format(data["images"][i]["file_name"])
        # filename = data["images"][0]['file_name'].split("/")[-1]
        r = requests.get(url, allow_redirects=True)
        with(open(filename, 'wb')) as f:
            f.write(r.content)
        if i % 100 == 0:
            print(i)

import shutil
def train_test_split(data):
    train_anno, test_anno = {"images":[], "categories":data["categories"], "annotations":[]}, {"images":[], "categories":data["categories"], "annotations":[]}
    annotations = data["annotations"]
    for i in range(len(data["images"])):
        src_path = "./{}".format(data["images"][i]["file_name"])
        print(i)
        
        if i < int(0.8 * len(data["images"])):
            dst_path = "./train" + src_path[1:]
            anno = data["images"][i]
            anno["file_name"] = dst_path[1:]
            train_anno['images'].append(anno)
            for j in range(len(annotations)):
                if annotations[j]["image_id"] == data["images"][i]["id"]:
                    train_anno['annotations'].append(data["annotations"][j])
        else:
            dst_path = "./test" + src_path[1:]
            anno = data["images"][i]
            anno["file_name"] = dst_path[1:]
            test_anno['images'].append(anno)
            
            for j in range(len(annotations)):
                if annotations[j]["image_id"] == data["images"][i]["id"]:
                    test_anno['annotations'].append(data["annotations"][j])
        shutil.copy(src_path, dst_path)

    with open("./train/train_anno.json", "w", encoding='utf-8') as outfile:
        json.dump(train_anno, outfile, ensure_ascii=False, indent = 4)

    with open("./test/test_anno.json", "w", encoding='utf-8') as outfile:
        json.dump(test_anno, outfile, ensure_ascii=False, indent = 4)

if __name__ == "__main__":
    data = json.load(open("./results_new.json", encoding="utf-8-sig"))
    #download_image(data=data)
    train_test_split(data)
