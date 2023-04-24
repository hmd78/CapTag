import time
import numpy as np
import pandas as pd
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
from threading import Thread
import argparse
import pickle

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, choices=['create', 'insert', 'load', 'release', 'drop'], required=True, help="create collection, insert data into collection and re-index, load vector index into memory, release vector index from memory, drop the collection and delete all data")
    parser.add_argument('--col-name', type=str, default='captioning_similarity', help="collection name")
    parser.add_argument('--host', type=str, default='localhost', help="milvus host")
    parser.add_argument('--port', type=str, default='19530', help="milvus port")
    
    # Creation Parameters
    parser.add_argument('--insert-bsz', type=int, default=100000, help="batch size for insert operation on db")
    parser.add_argument('--data-dir', type=str, default='TextSimilarity/feature_bert_768_with_index.obj', help="path to input data for insert operation")
    parser.add_argument('--vector-dim', type=int, default=128, help="dimension of vector embeddings")

    # Index Parameters
    parser.add_argument('--index-method', type=str, default='IVF_SQ8', help="indexing method")
    parser.add_argument('--nlist', type=int, default=1000, help="number of clusters")
    args = parser.parse_args()

    connections.connect("default", host=args.host, port=args.port)


    if args.task == 'create':
        if utility.has_collection(args.col_name):
            print("Collection already exists, drop it first!")
            exit()
        

        # Creating Collection

        fields = [
            FieldSchema(name="idx", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="tags", dtype=DataType.LIST, max_length=700),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=700),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=args.vector_dim)
        ]
        schema = CollectionSchema(fields, "vector search database")
        collection = Collection(args.col_name, schema, consistency_level="Strong")
        print("Collection Schema: {}".format(collection.schema))

    
    elif args.task == 'insert':
        while True:
            inp = input("Are you sure you want to insert data from this file? (Y/n)\n\
!!  Note: Milvus doesn't check the uniqueness of primary keys. ")
            if inp == "Y":
                break
            elif inp == "n":
                exit()  

        collection = Collection(args.col_name)
        collection.release()

        with open(args.data_dir, 'rb') as f:
            data = pickle.load(f)
        # data_count = data['features'].shape[0]
        data_count = len(data)
        print("Data Count: {}".format(data_count))

        t_insert = time.time()
        bsz = args.insert_bsz
        for i in range((data_count + bsz - 1) // bsz):
            st_range = i * bsz
            end_range = min((i + 1) * bsz, data_count)
            tmp_embd = data['embeddings'].values[st_range:end_range]
            # tmp_embd /= np.linalg.norm(tmp_embd, axis=0, keepdims=True)

            entities = [
                [data['id'].values[j] for j in range(st_range, end_range)], # id
                [data['tags'].values[j].split() for j in range(st_range, end_range)], # values
                tmp_embd, # embedding
                [data['category'].values[j] for j in range(st_range, end_range)] # category
            ]
            insert_result = collection.insert(entities)
        del tmp_embd, entities
        collection.flush()
        print("inserting time: {:.4f}s".format(time.time() - t_insert))


        # Building Index

        index = {
            "index_type": args.index_method,
            "metric_type": "IP",
            # "max_segment_size" : 2048,
            "params": {"nlist": args.nlist} #, 'm': 10, "efConstruction": 64, 'M': 16},
        }
        t_index = time.time()
        collection.create_index("features", index)
        print("indexing time: {:.4f}s".format(time.time() - t_index))    
        t_load = time.time()
        collection.load()
        print("loading time: {:.4f}s".format(time.time() - t_load))

        print("Collection Size: {}".format(collection.num_entities))


    elif args.task == 'load':
        collection = Collection(args.col_name)
        t_load = time.time()
        try:
            collection.load()
            print("name: {}\n".format(collection.name))
        except Exception as e:
            print("ERROR {}".format(e))
        print("loading time: {:.4f}s".format(time.time() - t_load))

    elif args.task == 'release':
        collection = Collection(args.col_name)
        collection.release()
        
    elif args.task == 'drop':
        while True:
            inp = input("Are you sure you want to drop the collection and remove all old data?(Y/n) ")
            if inp == "Y":
                break
            elif inp == "n":
                exit()
        utility.drop_collection(args.col_name)