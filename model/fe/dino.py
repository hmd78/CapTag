import torch
from PIL import Image
import torchvision.transforms as T
import hubconf

dinov2_vits14 = hubconf.dinov2_vits14()

img = Image.open('meta_dog.png')

transform = T.Compose([
T.Resize(224),
T.CenterCrop(224),
T.ToTensor(),
T.Normalize(mean=[0.5], std=[0.5]),
])

img = transform(img)[:3].unsqueeze(0)

with torch.no_grad():
features = dinov2_vits14(img, return_patches=True)[0]

print(features.shape)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
pca.fit(features)

pca_features = pca.transform(features)
pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
pca_features = pca_features * 255

plt.imshow(pca_features.reshape(16, 16, 3).astype(np.uint8))
plt.savefig('meta_dog_features.png')

In dinov2/models/vision_transformer.py line 290 add

def forward(self, *args, is_training=False, return_patches=False, **kwargs):
ret = self.forward_features(*args, **kwargs)
if is_training:
return ret
elif return_patches:
return ret["x_norm_patchtokens"]
else:
return self.head(ret["x_norm_clstoken"])