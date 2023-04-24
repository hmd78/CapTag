import argparse

import numpy as np
import pandas as pd
import streamlit as st
import yaml
import torch
import sys
from transformers import BertModel, BertConfig, BertTokenizer
import time
import os
import open_clip

from st_aggrid import AgGrid, GridUpdateMode, ColumnsAutoSizeMode
from st_aggrid.grid_options_builder import GridOptionsBuilder

from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

from TextSimilarity.model import dim_reduction
# from TextSimilarity.embeddings_128 import embedding
from TextSimilarity.embeddings_128_v2 import embedding
from ImageSimilarity.model import TwoLayerClassifier




TEXT_DATA_DIR = 'data'
FEATURE_REDUCTION_MODEL_DIR = 'TextSimilarity/saved_models/bert_dim_128_cpu.pt'
K = 10


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

@st.cache
def load_models():
    # global clip_model, train_preprocess, val_preprocess, feature_reduction_model, yolo_model, yolo_opt

    print("Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # bert_config = BertConfig.from_json_file('bert-hooshvare/config.json')
    clip_model, clip_train_preprocess, clip_val_preprocess = open_clip.create_model_and_transforms('RN50', pretrained='openai', device=device, cache_dir='./ImageSimilarity/saved_models')
    clip_model.eval()

    feature_reduction_model = TwoLayerClassifier(1024, 128, 10)
    feature_reduction_model.load_state_dict(torch.load(FEATURE_REDUCTION_MODEL_DIR))
    feature_reduction_model.eval()


    connections.connect("default", host="205.134.224.157", port="19530")
    milvus_collection = Collection("captioning_similarity")

    
    
    st.session_state['clip_model'] = clip_model
    st.session_state['clip_train_preprocess'] = clip_train_preprocess
    st.session_state['clip_val_preprocess'] = clip_val_preprocess
    st.session_state['feature_reduction_model'] = feature_reduction_model
    st.session_state['milvus_collection'] = milvus_collection



def similarity_image():

    st.title('Similarity Demo')

    ############## get image feature ###########

    output = "embedding"
    #######################################

    if st.button('Find similars'):
        
        search_params = {
                "metric_type": "IP",
                "params": {"nprobe": 10},
            }
        st.session_state["milvus_collection"].load()

        t_query = time.time()
        result = st.session_state["milvus_collection"].search([output.cpu().numpy()], "embedding",search_params, limit=10)
        print("milvus query time: {:.4f}s".format(time.time() - t_query))
        values_result = result[0].ids
        st.success(values_result)
        for guid in values_result:
            out = st.session_state["milvus_collection"].query(expr="id == {}".format(guid), output_fields= ["texture"])
            st.success(out)




def main():
    st.set_page_config(layout="wide")
    load_models()
    new_title = '<p style="font-size: 42px;">Similarity Demo</p>'
    read_me_0 = st.markdown(new_title, unsafe_allow_html=True)

    read_me_0.empty()
    similarity_image()
        

if __name__ == '__main__':
	main()	
