import numpy as np
import config
import os
from os.path import join
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import torchvision.transforms as transforms
import random
import torch
import torch.utils.data as data

class PETDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.input_path = join(root_dir, "input")
        self.target_path = join(root_dir, "target")
        self.list_files = os.listdir(self.input_path)

    def __len__(self):
        return len(self.input_path)

    def __getitem__(self, index):
        # img_file = self.list_files[index]
        # img_path = os.path.join(self.root_dir, img_file)
        # image = np.array(Image.open(img_path))
        # input_image = image[:, :432, :]
        # target_image = image[:, 432:, :]
        #
        # # augmentations = config.both_transform(image=input_image, image0=target_image)
        # # input_image = augmentations["image"]
        # # target_image = augmentations["image0"]
        #
        # # input_image = config.transform_only_input(image=input_image)["image"]
        # # target_image = config.transform_only_mask(image=target_image)["image"]
        #
        # return input_image, target_image
        
        input_image = Image.open(join(self.input_path, self.list_files[index])).convert('RGB')
        target_image = Image.open(join(self.target_path, self.list_files[index])).convert('RGB')
        input_image = input_image.resize((286, 286), Image.BICUBIC)
        target_image = target_image.resize((286, 286), Image.BICUBIC)
        input_image = transforms.ToTensor()(input_image)
        target_image = transforms.ToTensor()(target_image)
        w_offset = random.randint(0, max(0, 286 - 256 - 1))
        h_offset = random.randint(0, max(0, 286 - 256 - 1))

        input_image = input_image[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
        target_image = target_image[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
                
        input_image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(input_image)
        target_image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(target_image)

        if random.random() < 0.5:
            idx = [i for i in range(input_image.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            input_image = input_image.index_select(2, idx)
            target_image = target_image.index_select(2, idx)

        return input_image, target_image
       
        # if self.direction == "a2b":
        #     return input_image, target_image
        # else:
        #     return target_image, input_image


if __name__ == "__main__":
    dataset = PETDataset("data/train/")
    loader = DataLoader(dataset, batch_size=5)
    for x, y in loader:
        print(x.shape)
        save_image(x, "x.png")
        save_image(y, "y.png")
        import sys

        sys.exit()
