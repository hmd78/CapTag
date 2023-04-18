from sklearn.decomposition import PCA
import pickle as pk
from torch import nn

class TwoLayerClassifier(nn.Module):
    def __init__(self, input_dim, result_dim, class_num):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, result_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(result_dim, class_num)

    def reduce_dim(self, x):
        return self.linear1(x)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class pca():
    def __init__(self, n_components):
        self.model = PCA(n_components)

    def train(self,X):
        self.model.fit(X)

        pk.dump(self.model, open("./saved_models/pca.pkl","wb"))
        print("model is saved as saved_models/pca.pkl")
        print("explained variance ratio  : {pca.explained_variance_ratio_}")
        print("singular values :{pca.singular_values_}")
    
    def load(self,dir):
        self.model = pk.load(open(dir,'rb'))

    def reduce_dim(self,x):
        # TODO : batch input
        return self.model.transform(x)