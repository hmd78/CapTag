from feature_extraction import TextFeatureExtractor, ImageFeatureExtractor



# testing image feature extractor
image_fe = ImageFeatureExtractor()

img_path = './dataset/train/images/327607275_724883585840985_4106076554935472280_n4BB0DTEHWANQHUDJD9JQ_fa75.jpg'

print(image_fe.image_feature_extractor(img_path))
print(image_fe.resize_flatten(img_path, (100,100)))


# testing text feature extractor
text_fe = TextFeatureExtractor()