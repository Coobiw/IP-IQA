import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
import torch

"""
If you don't want to use default collate_fn in pytorch
please set the collator in the dataset.
"""

class std_collator:
    def __call__(self, batch):
        images, prompt, mos = zip(*batch)
        images = torch.stack(images)
        mos = torch.stack(mos)

        return {"images": images, "text": prompt, "mos": mos}

class AGIQA1k(Dataset):
    def __init__(self, data_info, transform, vis_root):
        super().__init__()
        self.vis_root = Path(vis_root)
        self.transform = transform

        self.data_info = data_info  # pd.DataFrame
        self.collator = std_collator()

    def __len__(self):
        return self.data_info.shape[0]

    def __getitem__(self, index):
        image_name = self.data_info.iloc[index, 0]
        prompt = self.data_info.iloc[index, 1]
        mos = self.data_info.iloc[index, -1]

        image = Image.open(str(self.vis_root / image_name)).convert("RGB")

        return self.transform(image), prompt, torch.tensor(mos,dtype=torch.float32)

class doublescore_collator:
    def __call__(self, batch):
        images, prompt, mos, align = zip(*batch)
        images = torch.stack(images)
        mos = torch.stack(mos).view(-1,1)
        align = torch.stack(align).view(-1,1)

        score = torch.cat([mos,align],dim=-1)

        return {"images": images, "text": prompt, "score": score}

class AGIQA3k(Dataset):
    def __init__(self, data_info, transform, vis_root):
        super().__init__()
        self.vis_root = Path(vis_root)
        self.transform = transform

        self.data_info = data_info  # pd.DataFrame
        self.collator = doublescore_collator()

    def __len__(self):
        return self.data_info.shape[0]

    def __getitem__(self, index):
        image_name = self.data_info.iloc[index, 0]
        prompt = self.data_info.iloc[index, 1]
        mos = self.data_info.iloc[index, 2]
        align = self.data_info.iloc[index, 3]

        image = Image.open(str(self.vis_root / image_name)).convert("RGB")

        return self.transform(image), prompt, torch.tensor(mos,dtype=torch.float32), torch.tensor(align,dtype=torch.float32)