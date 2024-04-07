import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler
import os


class HandGestureDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.labels = []
        self.transform = transform
        gesture_folders = [f.path for f in os.scandir(root_dir) if f.is_dir()]

        for idx, folder in enumerate(gesture_folders):
            for file in os.listdir(folder):
                data = np.load(os.path.join(folder, file))
                if len(data) < 63:
                    print(f'data len is not 63, but {len(data)}')
                    continue
                # Убедитесь, что data уже в правильном формате (seq_len, input_size)
                if self.transform:
                    data = self.transform(data)
                self.samples.append(data)
                self.labels.append(idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = self.labels[idx]
        return torch.tensor(sample, dtype=torch.float), torch.tensor(label, dtype=torch.long)


# Загрузка датасета
dataset = HandGestureDataset('datasets')
batch_size = 4

# Разделение на обучающую и тестовую выборки
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)