import torch
import numpy as np
import random

class DummySegDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        num_samples=100,
        image_size=(256, 256),
        num_classes=5,
    ):
        self.num_samples = num_samples
        self.H, self.W = image_size
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def generate_sample(self):
        # Random image
        image = np.random.rand(3, self.H, self.W).astype(np.float32)

        # Empty mask
        mask = np.zeros((self.H, self.W), dtype=np.int64)

        # Draw random rectangles (simple segmentation structure)
        num_objects = random.randint(1, 5)

        for _ in range(num_objects):
            cls = random.randint(1, self.num_classes - 1)

            x1, y1 = random.randint(0, self.W // 2), random.randint(0, self.H // 2)
            x2, y2 = random.randint(x1 + 10, self.W), random.randint(y1 + 10, self.H)

            mask[y1:y2, x1:x2] = cls

        return image, mask

    def __getitem__(self, idx):
        image, mask = self.generate_sample()

        return {
            "im": [torch.tensor(image, dtype=torch.float32)],
            "segmentation": torch.tensor(mask, dtype=torch.long),
            "im_metas": [{
                "ori_shape": torch.tensor([self.H, self.W]),
                "ori_filename": [f"dummy_{idx}.png"]
            }]
        }

    def get_gt_seg_maps(self):
        dataset = self.dataset
        gt = {}
        for i in range(len(dataset)):
            sample = dataset[i]
            filename = sample["im_metas"][0]["ori_filename"][0]
            gt[filename] = sample["segmentation"].numpy()
        return gt