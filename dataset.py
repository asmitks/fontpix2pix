import numpy as np
import config
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

class fontDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.A_folder = root_dir+'/A/'
        self.B_folder = root_dir+'/B/'
        self.list_files_A = os.listdir(self.A_folder)
        self.list_files_B = os.listdir(self.B_folder)




    def __len__(self):
        return len(self.list_files_A)

    def __getitem__(self, index):
        img_file_A = self.list_files_A[index]
        img_file_B = self.list_files_B[index]

        img_path_A = os.path.join(self.A_folder, img_file_A)
        img_path_B = os.path.join(self.B_folder, img_file_B)

        input_image = np.array(Image.open(img_path_A))
        target_image = np.array(Image.open(img_path_B))

        augmentations = config.both_transform(image=input_image, image0=target_image)

        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        input_image = config.transform_only_input(image=input_image)["image"]
        target_image = config.transform_only_mask(image=target_image)["image"]

        return input_image, target_image
    
class MapDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))
        input_image = image[:, :600, :]
        target_image = image[:, 600:, :]

        augmentations = config.both_transform(image=input_image, image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        input_image = config.transform_only_input(image=input_image)["image"]
        target_image = config.transform_only_mask(image=target_image)["image"]

        return input_image, target_image


if __name__ == "__main__":
    dataset = fontDataset("data/train/")
    loader = DataLoader(dataset, batch_size=1)
    for x, y in loader:
        print(x.shape)
        save_image(x, "x.png")
        save_image(y, "y.png")
        import sys

        sys.exit()