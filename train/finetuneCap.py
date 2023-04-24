from transformers import AutoProcessor, BlipForConditionalGeneration
import torch
import argparse
from dataloaderCap import CapDataloader
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description='Finetune blip captioning')
    parser.add_argument('annotation', help='path to json coco format annotation file')
    parser.add_argument('--work-dir', help='the dir to save models')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batchsize', type=int, default=3)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    train_dataset = CapDataloader(args.annotation, processor)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batchsize)


    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    model.train()

    for epoch in range(args.epochs):
        print("Epoch:", epoch)
        for idx, batch in enumerate(train_dataloader):
            input_ids = batch.pop("input_ids").to(device)
            pixel_values = batch.pop("pixel_values").to(device)

            outputs = model(input_ids=input_ids,
                            pixel_values=pixel_values,
                            labels=input_ids)
            
            loss = outputs.loss

            print("Loss:", loss.item())

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
    
    model. save_model(args.work_dir)





if __name__ == '__main__':
    main()
