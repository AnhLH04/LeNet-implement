import matplotlib.pyplot as plt
import torch
from torchvision import transforms

torch.manual_seed(42)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.1307, 0.3081),
])
def plot_loss(train_loss, val_loss):
  epochs = range(1, len(train_loss) +1)
  plt.plot(epochs, train_loss, "b", label = "Training loss")
  plt.plot(epochs, val_loss, "r", label = "Validation loss")
  plt.grid()
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.legend()
  plt.show()

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])