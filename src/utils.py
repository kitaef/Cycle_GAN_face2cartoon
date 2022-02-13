import random
import torch
from matplotlib import pyplot as plt


def denorm(img_tensors):
    return img_tensors * 0.5 + 0.5

def show(a_real, a_trans, a_recov,
         b_real, b_trans, b_recov,
         grid,
         rows=4):
    plt.subplot(grid[0:rows, 0:1])

    plt.imshow(denorm(make_grid(torch.cat((a_real[:rows].detach().cpu(),
                                           a_trans[:rows].detach().cpu(),
                                           a_recov[:rows].detach().cpu()),
                                          dim=0),
                                nrow=rows).permute(1, 2, 0)
                      )
               )
    plt.axis('off')
    plt.subplot(grid[0:rows, 1:2])
    plt.imshow(denorm(make_grid(torch.cat((b_real[:rows].detach().cpu(),
                                           b_trans[:rows].detach().cpu(),
                                           b_recov[:rows].detach().cpu()), dim=0), nrow=rows
                                ).permute(1, 2, 0)
                      )
               )
    plt.axis('off')


# image buffers to help Generators
class ReplayBuffer:
    def __init__(self, max_size=50):
        assert (max_size > 0), "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)