from pycocotools.coco import COCO
import os
from matplotlib import image
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm

def extract_mask(annFile="./batch3_utf8.json", img_dir="./images"):
  coco=COCO(annFile)

  catIds = coco.getCatIds()
  annsIds = coco.getAnnIds()

  for cat in catIds:
      Path(os.path.join("./masks",coco.loadCats(cat)[0]['name'])).mkdir(parents=True, exist_ok=True)

  for ann in tqdm(annsIds):
      mask = coco.annToMask(coco.loadAnns(ann)[0])
      gt_bbox = coco.loadAnns(ann)[0]['bbox']

      file_path = os.path.join("./masks",coco.loadCats(coco.loadAnns(ann)[0]['category_id'])[0]['name'],'id_'+str(ann)+ '_' + coco.loadImgs(coco.loadAnns(ann)[0]['image_id'])[0]['file_name'].split('/')[-1])
      # image.imsave(file_path, mask)

      org = Image.open(coco.loadImgs(coco.loadAnns(ann)[0]['image_id'])[0]['file_name'])
      # print(coco.loadImgs(coco.loadAnns(ann)[0]['image_id'])[0]['file_name'])
      # print(file_path)
      mask = np.expand_dims(mask, axis=-1)
      masked_image = np.multiply(org, mask)
      # cropped_image = masked_image[int(gt_bbox[1]):int(gt_bbox[1]+gt_bbox[3]), int(gt_bbox[0]):int(gt_bbox[0]+gt_bbox[2]),  :]
      im = Image.fromarray(masked_image)
      im.save(file_path)
      # if ann == 4:
      #   break