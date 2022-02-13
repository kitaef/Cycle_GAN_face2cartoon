import os
import torchvision.transforms as tt
from torch.utils.data import Dataset
from config import DATASET_SIZE
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split


class ImageDS(Dataset):

    def __init__(self, path_a, path_b, image_size):
        super().__init__()
        self.image_size = image_size
        self.images_a = self.process_folder(path_a, crop='face')
        self.images_b = self.process_folder(path_b, crop='cartoon')
        self.size_a = len(self.images_a)
        self.size_b = len(self.images_b)
        self.bigger = True if self.size_a >= self.size_b else False

    def process_folder(self, path, crop):
        images = []
        progress_bar = tqdm(total=DATASET_SIZE, desc='Images processed:', position=0)
        for dirname, _, filenames in os.walk(path):
            if len(images) >= DATASET_SIZE:
                break
            for filename in filenames:
                if len(images) >= DATASET_SIZE:
                    break
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    image = Image.open(os.path.join(dirname, filename))
                    progress_bar.update(1)
                    scale = 1.4 if crop == 'cartoon' else 1.0
                    transform = tt.Compose([
                        tt.Resize(int(self.image_size * scale)),
                        tt.CenterCrop(self.image_size),
                        tt.RandomHorizontalFlip(),
                        tt.ToTensor(),
                        tt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                    image = transform(image)
                    images.append(image)
                    if len(images) >= DATASET_SIZE:
                        break
        return images

    def __len__(self):
        return self.size_a if self.bigger else self.size_b

    def __getitem__(self, index):
        if self.bigger:
            image_a = self.images_a[index]
            image_b = self.images_b[int(index * (self.size_b / self.size_a))]
        else:
            image_a = self.images_a[int(index * (self.size_a / self.size_b))]
            image_b = self.images_b[index]
        return image_a, image_b