import torch
import torch.utils.data as data


class Dataset(data.Dataset):
    def __init__(self, targets, conditions):
        self.targets = targets
        self.conditions = conditions

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        x = self.targets[idx]
        c = self.conditions[idx]
        return torch.Tensor(x).float(), torch.Tensor(c).float()
