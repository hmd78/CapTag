from simple_retriever import SimpleRetriever
from transformers import TrainingArguments, AutoTokenizer, CLIPFeatureExtractor, AutoModel, CLIPVisionModel
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import json
import torchvision.transforms as transforms
import torch

# #load models

TEXT_MODEL = 'saved_models/clip/text'
IMAGE_MODEL = 'saved_models/clip/vision'
tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
vision_encoder = CLIPVisionModel.from_pretrained(IMAGE_MODEL)
text_encoder = AutoModel.from_pretrained(TEXT_MODEL)

model = SimpleRetriever(vision_encoder=vision_encoder,text_encoder=text_encoder,tokenizer=tokenizer)

IMAGE_SIZE = 200
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])
#data
class CapDataloader(Dataset):
    def __init__(self, annotation, mode: str = 'train'):
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

        if mode == 'train':
            self.augment = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD)
            ])
        elif mode == 'test':
            self.augment = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD)
            ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset.iloc[idx]
        tokens = tokenizer(item["tags"], padding='max_length',
                                max_length=MAX_LEN, truncation=True)
        # remove batch dimension
        # encoding = {k:v.squeeze() for k,v in encoding.items()}
        return {'input_ids': tokens.input_ids, 'attention_mask': tokens.attention_mask,
                'pixel_values': self.augment(Image.open(item['file_name']).convert('RGB'))}

    def change_file_names(self, x):
        x = x.split('/')[-1]
        return x

    def concat_tags(self, tags):    
        # TODO: concat category too
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
        x['file_name'] = x.agg(lambda x: './masks/'+ x['name'] + '/' + 'id_'+ x['id_x'] +'_' + x['file_name'], axis=1)
        # x = x.drop(x[x['id_x'] == '17003'].index)
        x = x.dropna()
        x['tags'] =x['tags'].apply(self.concat_tags)
        x['tags'] = x.agg(lambda x: self.cats[x['name']] + ',' + x['tags'], axis=1)
        return x

train_ds = CapDataloader('./dataset/train/train_anno.json')
df = train_ds.dataset

mydf = df.iloc[list(df['name'] == 'manto')]

texts = mydf['tags'].tolist()
images = mydf['file_name'].tolist()
print(len(images), len(texts))

# inference

model.compute_embeddings(images, texts)

model.image_to_image_search('/home/art/Code/CapTag/masks/manto/id_336_324719832_911835690260307_1627614184832298502_n53EG0VTTI8DNBTA90HJ3_ebd4.jpg')


model.text_to_image_search('جین کتی دخترانه عروسکی')
model.text_to_image_search(','.join(['کتی', 'کوتاه', 'یقه انگلیسی','زنانه', 'هنری']))