from model.fe.clipfa.application.main import CLIPDemo
from model.fe.feature_extraction import CLIP
import pathlib

# p = pathlib.Path('.')
# print(list(p.glob('**/*')))

DATA_DIR = pathlib.Path("./dataset/masks/shomiz")
# print(list(DATA_DIR.glob('*')))

model = CLIP()
simple_clip = CLIPDemo(vision_encoder=model.vision_encoder,text_encoder=model.text_encoder,tokenizer=model.tokenizer)

simple_clip.compute_image_embeddings(list(DATA_DIR.glob('*')))

simple_clip.image_to_image_search('./dataset/masks/shomiz/id2485_326149482_702373984804333_8142245044675541309_nV597GABABJA3N3F3RVQF_99de.jpg')