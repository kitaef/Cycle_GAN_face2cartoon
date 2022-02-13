import warnings
warnings.filterwarnings("ignore")
import torch
from sklearn.model_selection import train_test_split
from config import *
from utils import denorm, show, weights_init
from utils import ReplayBuffer
from models import Generator, Discriminator
from datasets import ImageDS

faces_ds = ImageDS(PATH_A, PATH_B, IMAGE_SIZE)
train_ds, test_ds = train_test_split(faces_ds, test_size=0.2, random_state=1)
print(f'train batches: {len(train_ds)}, test batches: {len(test_ds)}')

G_A2B = Generator()
G_B2A = Generator()
D_A = Discriminator()
D_B = Discriminator()

G_A2B.apply(weights_init)
G_B2A.apply(weights_init)
D_A.apply(weights_init)
D_B.apply(weights_init)

G_A2B.load_state_dict(torch.load('/content/dls_final_project/src/pretrained_models/new_120_400_G_A2B'))
G_B2A.load_state_dict(torch.load('/content/dls_final_project/src/pretrained_models/new_120_400_G_B2A'))
D_A.load_state_dict(torch.load('/content/dls_final_project/src/pretrained_models/new_120_400_D_A'))
D_B.load_state_dict(torch.load('/content/dls_final_project/src/pretrained_models/new_120_400_D_B'))
