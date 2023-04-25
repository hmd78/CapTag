from transformers import default_data_collator
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from PIL import Image
import json
from config import (tokenizer, BATCH_SIZE, IMAGE_SIZE, MEAN,
                      STD, MAX_LEN, DATA_FILE, TEST_SIZE)
from utils import show_data
from pycocotools.coco import COCO

class CapDataloader(Dataset):
    def __init__(self, annotation, mode: str = 'train'):
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
        x['tags'] =x['tags'].apply(self.concat_tags)
        return x

class CLIPDataset(Dataset):

    def __init__(self, image_paths: list, text: list, mode: str = 'train'):
        self.image_paths = image_paths
        self.tokens = tokenizer(text, padding='max_length',
                                max_length=MAX_LEN, truncation=True)

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

    def __getitem__(self, idx):
        token = self.tokens[idx]
        return {'input_ids': token.ids, 'attention_mask': token.attention_mask,
                'pixel_values': self.augment(Image.open(self.image_paths[idx]).convert('RGB'))}

    def __len__(self):
        return len(self.image_paths)


# if __name__ == '__main__':
    dl = CapDataloader("./dataset/train/train_anno.json", mode='train')
    train_dl = DataLoader(dl, batch_size=2,
                          collate_fn=default_data_collator)
    train_ds = dl
    test_ds = dl 
 
    # df = pd.read_csv(DATA_FILE)
    # train_df, test_df = train_test_split(df, test_size=TEST_SIZE)
    # train_ds = CLIPDataset(image_paths=train_df.image.tolist(),
    #                        text=train_df.caption.tolist(), mode='train')
    # test_ds = CLIPDataset(image_paths=test_df.image.tolist(),
    #                       text=test_df.caption.tolist(), mode='test')

    # train_dl = DataLoader(train_ds, batch_size=2,
    #                       collate_fn=default_data_collator)
    # for it in train_dl:
    #     print(it['input_ids'].shape)
    #     print(it['pixel_values'].shape)
    #     show_data(it)
    #     break

    
