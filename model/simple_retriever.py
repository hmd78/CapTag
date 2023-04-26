
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import default_data_collator

from fe.clipfa.application.utils import VisionDataset, TextDataset


class SimpleRetriever:
    def __init__(self, vision_encoder, text_encoder, tokenizer,
                 batch_size: int = 32, max_len: int = 64, device='cuda'):

        self.vision_encoder = vision_encoder.eval().to(device)
        self.text_encoder = text_encoder.eval().to(device)
        self.batch_size = batch_size
        self.device = device
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.text_embeddings_ = None
        self.image_embeddings_ = None
        

    def compute_embeddings(self, image_paths: list, text: list):
        """ Compute image embeddings for a list of image paths and text embeddings for their specified text
        """

        embeddings = {'images':[], 'texts':[]}
        #image features
        self.image_paths = image_paths
        image_datalodear = DataLoader(VisionDataset(
            image_paths=image_paths), batch_size=self.batch_size)
        with torch.no_grad():
            for images in tqdm(image_datalodear, desc='computing image embeddings'):
                image_embedding = self.vision_encoder(
                    pixel_values=images.to(self.device)).pooler_output
                embeddings['images'].append(image_embedding)
        self.image_embeddings_ =  torch.cat(embeddings['images'])

        # test features
        self.text = text
        text_dataloader = DataLoader(TextDataset(text=text, tokenizer=self.tokenizer, max_len=self.max_len),
                                batch_size=self.batch_size, collate_fn=default_data_collator)
        
        with torch.no_grad():
            for tokens in tqdm(text_dataloader, desc='computing text embeddings'):
                text_embedding = self.text_encoder(input_ids=tokens["input_ids"].to(self.device),
                                                    attention_mask=tokens["attention_mask"].to(self.device)).pooler_output
                embeddings['texts'].append(text_embedding)
        self.text_embeddings_ = torch.cat(embeddings['texts'])


    def text_query_embedding(self, query: str = 'موز'):
        tokens = self.tokenizer(query, return_tensors='pt')
        with torch.no_grad():
            text_embedding = self.text_encoder(input_ids=tokens["input_ids"].to(self.device),
                                               attention_mask=tokens["attention_mask"].to(self.device)).pooler_output
        return text_embedding

    def image_query_embedding(self, image):
        image = VisionDataset.preprocess(image).unsqueeze(0)
        with torch.no_grad():
            image_embedding = self.vision_encoder(
                image.to(self.device)).pooler_output
        return image_embedding

    def most_similars(self, embeddings_1, embeddings_2):
        values, indices = torch.cosine_similarity(
            embeddings_1, embeddings_2).sort(descending=True)
        return values.cpu(), indices.cpu()
    
    def image_to_image_search(self, image_path: str, top_k=10):
        image = Image.open(image_path)
        image_embedding = self.image_query_embedding(image)
        ids, indices = self.most_similars(self.image_embeddings_, image_embedding)
        properties = []
        for ids in indices[0:3]:
            temp = self.text[ids].split(',')
            for tok in temp:
                if tok not in properties:
                    properties.append(tok)
        print(properties)

        matches = np.array(self.image_paths)[indices][:top_k]
        _, axes = plt.subplots(2, int(top_k/2), figsize=(15, 5))
        for match, ax in zip(matches, axes.flatten()):
            ax.imshow(Image.open(match).resize((224, 224)))
            ax.axis("off")
        plt.show()


    def text_to_image_search(self, query: str, top_k=10):

        query_embedding = self.text_query_embedding(query=query)
        _, indices = self.most_similars(self.text_embeddings_, query_embedding)
        properties = []
        for ids in indices[0:3]:
            print(self.text[ids])
            temp = self.text[ids].split(',')
            for tok in temp:
                if tok not in properties:
                    properties.append(tok)
        print(properties)
        matches = np.array(self.image_paths)[indices][:top_k]
        _, axes = plt.subplots(2, int(top_k/2), figsize=(15, 5))
        for match, ax in zip(matches, axes.flatten()):
            ax.imshow(Image.open(match).resize((224, 224)))
            ax.axis("off")
        plt.show()




    
