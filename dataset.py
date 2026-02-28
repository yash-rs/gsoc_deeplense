import os
import numpy as np
import torch
from torch.utils.data import Dataset


class LensDataset(Dataset):
    """
    Dataset for loading .npy lensing images.
    """

    def __init__(self, root_dir, split="train", transform=None):
        """
        Args:
            root_dir: path to dataset directory
            split: "train" or "val"
            transform: optional transform
        """
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform

        self.classes = sorted(os.listdir(self.root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        self.samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(self.root_dir, cls_name)
            for fname in os.listdir(cls_dir):
                if fname.endswith(".npy"):
                    path = os.path.join(cls_dir, fname)
                    self.samples.append((path, self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        image = np.load(path)  # shape: (1, 150, 150)

        image = image.astype(np.float32)
        image = torch.from_numpy(image)

        return image, label