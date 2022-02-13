import torch.cuda

BATCH_SIZE = 5
IMAGE_SIZE = 64
DATASET_SIZE = 25000
PATH_A = '/content/real'
PATH_B = '/content/cartoon'
LR = 0.008
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 130