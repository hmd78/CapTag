from feature_extraction import TextFeatureExtractor, ImageFeatureExtractor, CLIP


img_path = './dataset/train/images/327607275_724883585840985_4106076554935472280_n4BB0DTEHWANQHUDJD9JQ_fa75.jpg'
text = "تیشرت نخی"
# clip feature extraction test
fe = CLIP()

print("image embedding = {}".format(fe.image_fe(img_path)))
print("text embedding = {}".format(fe.text_fe(text)))




# # testing image feature extractor
# image_fe = ImageFeatureExtractor()



# print(image_fe.image_feature_extractor(img_path))
# print(image_fe.resize_flatten(img_path, (100,100)))

# # testing text feature extractor
# text_fe = TextFeatureExtractor()