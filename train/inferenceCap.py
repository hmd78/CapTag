from PIL import Image
import argparse
import torch
from transformers import AutoProcessor, BlipForConditionalGeneration


def parse_args():
    parser = argparse.ArgumentParser(description='extract segmentation  masks from original images')
    parser.add_argument('--img', help='path to test image')
    parser.add_argument('--model', help='path to model weights')
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BlipForConditionalGeneration.from_pretrained(args.model).to(device)
    processor = AutoProcessor.from_pretrained(args.model)
    image = Image.open(args.img)
    inputs = processor(images=image, return_tensors="pt").to(device)
    pixel_values = inputs.pixel_values

    generated_ids = model.generate(pixel_values=pixel_values, max_length=70)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_caption)

if __name__ == '__main__':
    main()
