from torch.utils.data import Dataset
from PIL import Image
import json
import pandas as pd


class CapDataloader(Dataset):
    def __init__(self, annotation, processor):
        self.cats = {
                        "body":"بادی",
                        "top":"تاپ",
                        "t-shirt":"تیشرت",
                        "jumpsuit": "سرهمی",
                        "socks": "جوراب",
                        "skirt":"دامن",
                        "earrings":"گوشواره",
                        "bracelet":"دستبند",
                        "ring":"انگشتر",
                        "necklace":"گردنبند",
                        "anklet":"پابند",
                        "glove":"دستکش",
                        "wristlet":"مچبند",
                        "watch":"ساعت",
                        "luggage":"ساک",
                        "bra":"سوتین",
                        "sweatshirt":"ژاکت",
                        "scarf":"شال",
                        "pants":"شلوار",
                        "shorts":"شلوارک",
                        "underwear":"لباس زیر",
                        "glasses":"عینک",
                        "dress": "لباس مجلسی",
                        "manto":"مانتو",
                        "swimsuit":"مایو",
                        "hoodie":"هودی",
                        "pullover":"پلیور",
                        "vest":"وست",
                        "pancho":"پانچو",
                        "shirt":"پیراهن مردانه",
                        "chador":"چادر",
                        "jacket":"کاپشن",
                        "tie":"کروات",
                        "shoe":"کفش",
                        "boot":"بوت",
                        "hat":"کلاه",
                        "belt":"کمربند",
                        "bag":"کیف",
                        "backpack":"کوله پشتی",
                        "COAT":"کت",
                        "shomiz":"شومیز"
                    }
        with open(annotation, 'rt') as annotations:
            coco = json.load(annotations)
        self.anns_img = pd.json_normalize(coco['images'])
        self.anns_cat = pd.json_normalize(coco['categories'])
        self.anns = pd.json_normalize(coco['annotations'])
        self.dataset = self.preprocess_anns(self.anns)
        self.processor = processor
        

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset.iloc[idx]
        img = Image.open(item['file_name'])
        encoding = self.processor(images=img, text=item["tags"], padding="max_length", return_tensors="pt")
        # remove batch dimension
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        return encoding
    def change_file_names(self, x):
        x = x.split('/')[-1]
        return x
    def concat_tags(self, tags):
        concat_str = ''
        for idx, i in enumerate(tags):
          if idx != len(tags)-1:
            concat_str += i['value']+','
          elif idx == len(tags)-1:
            concat_str += i['value']
        return concat_str
    def preprocess_anns(self, anns):
        x = pd.merge(self.anns, self.anns_img[['id', 'file_name']], left_on='image_id', right_on='id')[['id_x', 'image_id', 'tags', 'file_name', 'category_id']]
        x = pd.merge(x, self.anns_cat, left_on='category_id', right_on='id')
        x['file_name'] = x['file_name'].apply(self.change_file_names)
        x['id_x'] = x['id_x'].map(str)
        x['file_name'] = x.agg(lambda x: 'masks/'+ x['name'] + '/' + 'id_'+ x['id_x'] +'_' + x['file_name'], axis=1)
        x = x.drop(x[x['id_x'] == '17003'].index)
        x = x.dropna()
        x['tags'] =x['tags'].apply(self.concat_tags)
        x['tags'] = x.agg(lambda x: self.cats[x['name']] + ',' + x['tags'], axis=1)
        return x