from model.fe.clipfa.application.main import CLIPDemo
# from model.fe.feature_extraction import CLIP
from transformers import TrainingArguments, AutoTokenizer, CLIPFeatureExtractor, AutoModel, CLIPVisionModel
import pathlib

# p = pathlib.Path('.')
# print(list(p.glob('**/*')))

DATA_DIR = pathlib.Path("./dataset/masks/shomiz")
# print(list(DATA_DIR.glob('*')))

# model = CLIP()
# simple_clip = CLIPDemo(vision_encoder=model.vision_encoder,text_encoder=model.text_encoder,tokenizer=model.tokenizer)

TEXT_MODEL = 'saved_models/clip/text'
IMAGE_MODEL = 'saved_models/clip/vision'

# vision_preprocessor = CLIPFeatureExtractor.from_pretrained(IMAGE_MODEL)
tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
vision_encoder = CLIPVisionModel.from_pretrained(IMAGE_MODEL)
text_encoder = AutoModel.from_pretrained(TEXT_MODEL)
simple_clip = CLIPDemo(vision_encoder=vision_encoder,text_encoder=text_encoder,tokenizer=tokenizer)


simple_clip.compute_image_embeddings(list(DATA_DIR.glob('*')))

simple_clip.image_to_image_search('./dataset/masks/shomiz/id2485_326149482_702373984804333_8142245044675541309_nV597GABABJA3N3F3RVQF_99de.jpg')