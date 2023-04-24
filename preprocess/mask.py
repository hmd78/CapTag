from pycocotools.coco import COCO
import os
from matplotlib import image
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='extract segmentation  masks from original images')
    parser.add_argument('--annotation', help='path to json coco format annotation file')
    parser.add_argument('--mask-dir', help='directory where masks are saved')
    parser.add_argument('--dataset-dir', help='directory where original images are saved')
    
    args = parser.parse_args()

    return args


def extract_mask(annFile, mask_dir, dataset_prefix):
  coco=COCO(annFile)

  catIds = coco.getCatIds()
  annsIds = coco.getAnnIds()

  for cat in catIds:
      Path(os.path.join(mask_dir,coco.loadCats(cat)[0]['name'])).mkdir(parents=True, exist_ok=True)

  for ann in tqdm(annsIds):
      mask = coco.annToMask(coco.loadAnns(ann)[0])
      gt_bbox = coco.loadAnns(ann)[0]['bbox']

      file_path = os.path.join(mask_dir,coco.loadCats(coco.loadAnns(ann)[0]['category_id'])[0]['name'],'id_'+str(ann)+ '_' + coco.loadImgs(coco.loadAnns(ann)[0]['image_id'])[0]['file_name'].split('/')[-1])
      # image.imsave(file_path, mask)

      org = Image.open(dataset_prefix+coco.loadImgs(coco.loadAnns(ann)[0]['image_id'])[0]['file_name'])
      # print(coco.loadImgs(coco.loadAnns(ann)[0]['image_id'])[0]['file_name'])
      # print(file_path)
      mask = np.expand_dims(mask, axis=-1)
      masked_image = np.multiply(org, mask)
      cropped_image = masked_image[max(int(gt_bbox[1]), 0):min(int(gt_bbox[1]+gt_bbox[3]), masked_image.shape[0]), max(int(gt_bbox[0]), 0):min(int(gt_bbox[0]+gt_bbox[2]), masked_image.shape[1]),  :]
      im = Image.fromarray(masked_image)
      im.save(file_path)

    
def main():
    args = parse_args()
    extract_mask(args.annotation, args.mask_dir)


if __name__ == '__main__':
    main()
