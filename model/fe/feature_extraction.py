import numpy as np
import logging
import json
from PIL import Image
import torch
import sys
import open_clip
import time
import os
from transformers import BertModel, BertConfig, BertTokenizerFast, AutoConfig, AutoTokenizer, AutoModel

from io import BytesIO
from typing import Optional, List
import asyncio
import copy
import pickle

from dimention_reduction import TwoLayerClassifier

import torchvision.transforms as T
transform = T.ToPILImage()

from transformers import CLIPVisionModel, RobertaModel, AutoTokenizer, CLIPFeatureExtractor

import torch



class ImageFeatureExtractor:
    # feature extraction from images
    def __init__(self):

        FEATURE_REDUCTION_MODEL_DIR = './saved_models/clip_resnet/RN50_twolayer_cpu.pt'
        print("Loading CLIP model...")

        self.device = "cuda"
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms('RN50', pretrained='openai', device=self.device, cache_dir='./saved_models/clip_resnet')
        self.clip_model.eval()
        self.clip_model.to(self.device)
        # TODO: train a dimention reduction model fit to our classes
        self.feature_reduction_model = TwoLayerClassifier(1024, 128, 10)
        self.feature_reduction_model.load_state_dict(torch.load(FEATURE_REDUCTION_MODEL_DIR))
        self.feature_reduction_model.eval()
        self.feature_reduction_model.to(self.device)

    def _read(self, img_or_path):
        """Read an image.
        Args:
            img_or_path (ndarray or str or Path)
        Returns:
            ndarray: Loaded image array.
        """

        if isinstance(img_or_path, np.ndarray):
            return img_or_path

        elif type(img_or_path) == str:
            img = Image.open(img_or_path)
            img = np.array(img)
            return img
        
        else:
            print("error reading image")

    def resize_flatten(self, images, size = (100,100)):
        #simple resizeing and flattening images as features
        features = []
        if type(images) != list:
            images = [images]
        else:
            pass

        for image in images:
            img = self._read(image)
            img = np.resize(img, size)
            feature = img.flatten()
            features.append(feature)
        return features

    def image_feature_extractor(self, payload):

        if type(payload) != list:
            payload = [payload]
        else:
            pass

        images = []
        for imge in payload:
            img = self._read(imge)
            img = transform(img)
            img = self.clip_preprocess(img).unsqueeze(0)
            images.append(img.cpu().numpy()[0])

        with torch.no_grad():
            image_features = self.clip_model.encode_image(torch.tensor(images).to(self.device))
            image_features /= image_features.norm(dim=-1, keepdim=True)
            image_features = self.feature_reduction_model.reduce_dim(image_features)
            image_features /= image_features.norm(dim=-1, keepdim=True)

        # output = json.dumps(pickle.dumps(image_features.cpu().numpy()).decode('latin-1'))

        return image_features.cpu().numpy()


class TextFeatureExtractor:

    def __init__(self):

        FEATURE_REDUCTION_MODEL_DIR = './saved_models/parsbert/bert_dim_128_cpu.pt'
        BERT_DIR = './saved_models/pars_bert/base/'
        TOKENIZER_DIR = './saved_models/pars_bert/tokenizer/'
        print("Loading BERT model...")
        self.device = "cuda"
        # bert_config = BertConfig.from_pretrained('HooshvareLab/bert-base-parsbert-uncased', output_hidden_states = True)
        # self.tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_DIR)
        # self.bert_model = BertModel.from_pretrained(BERT_DIR + 'pytorch_model.bin', config=bert_config)
        config = AutoConfig.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")
        self.tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")
        self.bert_model = AutoModel.from_pretrained("HooshvareLab/bert-base-parsbert-uncased",output_hidden_states = True)

        self.bert_model.eval()
        self.feature_reduction_model = TwoLayerClassifier(768, 128, 14)
        self.feature_reduction_model.load_state_dict(torch.load(FEATURE_REDUCTION_MODEL_DIR))
        self.feature_reduction_model.eval()

    def simple_text_feature_extractor(self, text):
        # takes strings as input
        texts = [text]
        Embedding = self._embedding( texts=texts,
                            tokenizer=self.tokenizer, 
                            bert_model=self.bert_model,
                            dim_red=self.feature_reduction_model,
                            device = self.device
                           )

        return Embedding.cpu().numpy()

    def _embedding(self, texts, tokenizer, bert_model, dim_red, device):

        tokenized_text = tokenizer(texts, truncation=True, padding='max_length', max_length=500, return_tensors='pt')

        text_ids = tokenized_text['input_ids'].to(device)
        text_attention = tokenized_text['attention_mask'].to(device)

        bert_model.to(device)
        dim_red.to(device)
        
        bert_model.eval()
        dim_red.eval()

        with torch.no_grad():
            outputs = bert_model(text_ids, text_attention)
            bert_text_embedding = outputs[1]

        bert_text_embedding /= bert_text_embedding.norm(dim=-1, keepdim=True)

        return bert_text_embedding


class CLIP:
    def __init__(self):
        # download pre-trained models
        # self.vision_encoder = CLIPVisionModel.from_pretrained('SajjadAyoubi/clip-fa-vision')
        # torch.save(self.vision_encoder,'saved_models/vision_encoder.pth')
        # self.vision_encoder.save_pretrained('saved_models/clip/vision')

        # self.preprocessor = CLIPFeatureExtractor.from_pretrained('SajjadAyoubi/clip-fa-vision')
        # torch.save(self.preprocessor,'saved_models/vision_preprocessor.pth')
        # self.preprocessor.save_pretrained('saved_models/clip/vision')

        # self.text_encoder = RobertaModel.from_pretrained('SajjadAyoubi/clip-fa-text')
        # torch.save(self.text_encoder,'saved_models/text_encoder.pth')
        # self.text_encoder.save_pretrained("./saved_models/clip/text")
        # self.tokenizer = AutoTokenizer.from_pretrained('SajjadAyoubi/clip-fa-text')
        # self.tokenizer.save_pretrained("./saved_models/clip/text")

        # load models locally with config
        # self.vision_encoder = torch.load('saved_models/vision_encoder.pth')
        # self.preprocessor = torch.load('saved_models/vision_preprocessor.pth')
        # self.text_encoder = torch.load('saved_models/text_encoder.pth')
        # self.tokenizer = AutoTokenizer.from_pretrained("./saved_models/tokenizer/")

        # load models locally 
        self.vision_encoder = CLIPVisionModel.from_pretrained('saved_models/clip/vision')
        self.preprocessor = CLIPFeatureExtractor.from_pretrained('saved_models/clip/vision')
        self.text_encoder = RobertaModel.from_pretrained('./saved_models/clip/text')
        self.tokenizer = AutoTokenizer.from_pretrained('./saved_models/clip/text')

        

    def _read(self, img_or_path):
        """Read an image.
        Args:
            img_or_path (ndarray or str or Path)
        Returns:
            ndarray: Loaded image array.
        """

        if isinstance(img_or_path, np.ndarray):
            return img_or_path

        elif type(img_or_path) == str:
            img = Image.open(img_or_path)
            img = np.array(img)
            return img
        
        else:
            print("error reading image")

    def image_fe(self, payload):

        if type(payload) != list:
            payload = [payload]
        else:
            pass

        features = []
        for imge in payload:
            img = self._read(imge)
            img = transform(img)

            image_embedding = self.vision_encoder(**self.preprocessor(img, return_tensors='pt')).pooler_output
            features.append(image_embedding)
        return features

    def text_fe(self, payload):

        if type(payload) != list:
            payload = [payload]
        else:
            pass

        features = []
        for text in payload:
            text_embedding = self.text_encoder(**self.tokenizer(text, return_tensors='pt')).pooler_output
            features.append(text_embedding)
        return features