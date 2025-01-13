import torch
from torch.utils.data import Dataset
import numpy as np
from ..preprocessing.augmentation.elastic_transform import ElasticTransform
from ..preprocessing.enhancement.clahe import CLAHE

class SyntheticDataset(Dataset):
    def __init__(self, real_dataset, gan_model, num_synthetic=1000):
        self.real_dataset = real_dataset
        self.gan_model = gan_model
        self.num_synthetic = num_synthetic
        self.elastic_transform = ElasticTransform()
        self.clahe = CLAHE()
        
        self.synthetic_images = []
        self.synthetic_labels = []
        self.generate_synthetic_data()

    def generate_synthetic_data(self):
        self.gan_model.eval()
        with torch.no_grad():
            for i in range(self.num_synthetic):
                z = torch.randn(1, 100).cuda()
                labels = torch.randint(0, 5, (1,)).cuda()
                synthetic_image = self.gan_model(z, labels)
                
                # Apply realistic transformations
                synthetic_image = synthetic_image.cpu().numpy()[0]
                synthetic_image = self.elastic_transform(synthetic_image)
                synthetic_image = self.clahe(synthetic_image)
                
                self.synthetic_images.append(synthetic_image)
                self.synthetic_labels.append(labels.cpu())

    def __len__(self):
        return self.num_synthetic

    def __getitem__(self, idx):
        return self.synthetic_images[idx], self.synthetic_labels[idx]
