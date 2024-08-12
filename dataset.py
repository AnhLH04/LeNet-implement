from torch.utils.data import DataLoader,random_split
from torchvision import datasets
from utils import transform

training_data = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transform = transform,
)
train_size = int(0.8* len(training_data))
val_size = len(training_data)- train_size
training_data , val_data = random_split(training_data, [train_size, val_size])

test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform = transform,
)
train_dataloader = DataLoader(training_data, batch_size = 64, shuffle = True)
val_dataloader = DataLoader(val_data, batch_size=64, shuffle = True)
test_dataloader  = DataLoader(test_data, batch_size=64, shuffle = False)