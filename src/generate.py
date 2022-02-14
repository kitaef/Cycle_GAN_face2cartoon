import torch
import numpy as np
import warnings
import torchvision.transforms as T
from torchvision import models
from facenet_pytorch import MTCNN
from google.colab import files
import matplotlib.pyplot as plt
from PIL import Image
from models import Generator, Discriminator


def denorm(img_tensors):
  return img_tensors * 0.5 + 0.5

# helper function to get segmentation mask of a person
def decode_segmap(image, nc=21):
  
  label_colors = np.array([(0, 0, 0),  # 0=background
               # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
               # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (255, 255, 255),
               # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)
  
  for l in range(0, nc):
    idx = image == l
    r[idx] = label_colors[l, 0]
    g[idx] = label_colors[l, 1]
    b[idx] = label_colors[l, 2]
    
  rgb = np.stack([r, g, b], axis=2)
  return rgb

def transform_face(fname=None):
  if not fname:
    uploaded = files.upload()
    fname = list(uploaded.keys())[0]
  img = Image.open(fname)
  face = mtcnn(img)
  trf = T.Compose([
                  T.Resize(64),
                  T.ToTensor(), 
                  T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  to_pil = T.ToPILImage()
  inp = face.unsqueeze(dim=0)
  out = fcn(inp)['out']
  om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
  mask = decode_segmap(om)
  face = to_pil(0.5 * face + 0.5)
  mask = to_pil(mask).convert('L')
  bg = Image.new('RGB', (64,64), (255,255,255))
  face_segmented = bg.paste(face, (0,0), mask=mask)
 
  G_A2B.eval()
  G_B2A.eval()

  cartoon = G_A2B(trf(bg).unsqueeze(dim=0))
  cartoon = to_pil(denorm(cartoon.squeeze().detach()))

  # show image, crop and mask
  plt.figure(figsize=(4, 10))
  plt.subplot(1,3,1)
  plt.imshow(img)
  plt.axis('off')
  plt.subplot(1,3,2)
  plt.imshow(bg)
  plt.axis('off')
  plt.subplot(1,3,3)
  plt.imshow(cartoon)
  plt.axis('off')
  plt.show()


warnings.filterwarnings('ignore')
mtcnn = MTCNN(image_size=64, margin=25)
fcn = models.segmentation.fcn_resnet101(pretrained=True).eval()
G_A2B = Generator()
G_B2A = Generator()
G_A2B.load_state_dict(torch.load('/content/Cycle_GAN_face2cartoon/src/pretrained_models/new_130_400_G_A2B'))
G_B2A.load_state_dict(torch.load('/content/Cycle_GAN_face2cartoon/src/pretrained_models/new_130_400_G_B2A'))