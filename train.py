"""LeNet.ipynb
"""

import torch
import torch.nn as nn
from torchvision import datasets,transforms
from model import LeNet
from dataset import train_dataloader, val_dataloader, test_dataloader

device = (
    "cuda" if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(device)
torch.manual_seed(42)
model = LeNet().to(device)

loss_fn = nn.CrossEntropyLoss()
# Consider using higher learning rate (Adam default: 1e-3) and using a learning rate scheduler
optimizer = torch.optim.Adam(params = model.parameters()) #weight_decay=1e-4) # Consider using weight_decay for l2 regularization

train_loss = []
val_loss = []
def train(train_dataloader, val_dataloader, model, loss_fn, optimizer):
  model.train()
  loss, acc = 0.0, 0.0
  train_size, val_size = len(train_dataloader.dataset), len(val_dataloader.dataset)
  num_train_batches, num_val_batches = len(train_dataloader), len(val_dataloader)

  for X, y in train_dataloader:
    X, y = X.to(device), y.to(device)

    # Forward pass
    pred = model(X)
    batch_loss = loss_fn(pred, y)

    # Backprop
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Compute loss, acc
    loss += batch_loss
    acc += (pred.argmax(1)==y).type(torch.float).sum().item() # Consider using torchmetrics for more accurate metrics than regular accuracy
  loss /= num_train_batches
  acc /= train_size

  print(f"Epoch: {epoch+1} | Training loss: {loss:.4f} | Training acc: {acc*100:.2f}%", end =" ")

  model.eval()
  test_loss, correct =  0.0 , 0.0
  with torch.inference_mode():
    for X,y in val_dataloader:
      X,y = X.to(device), y.to(device)
      pred = model(X)
      test_loss += loss_fn(pred, y).item()
      correct += (pred.argmax(1)==y).type(torch.float).sum().item()
  correct = correct/val_size # Using torchmetrics instead
  test_loss = test_loss/num_val_batches
  print(f"| Validation loss: {test_loss:.4f} | Validation acc: {correct*100:.2f}%")
  val_loss.append(test_loss)
  train_loss.append(loss.item())
  return train_loss, val_loss

epochs = 5 # Training the model for many epochs since added regularization will make the model generalize better,
            # hence it takes longer to overfit -> more epochs.
for epoch in range(epochs):
  train_loss, val_loss = train(train_dataloader, val_dataloader, model, loss_fn, optimizer)


# Tweak hyperparameters: number of layers, neurons per layer, adding regularization, dropout, etc. based on this curve.

"""- This curve shows that the model starts to overfit at epoch 25, so it's common to retrain a brand new (final (?)) model for 25 epochs.
- Consider plotting a accuracy curve as well.
- Using another metric (F1, precision, recall, AUC,...) is recommended.
"""

# After tweaking all the hyperparams and satisfied with the result, move on to this step.

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = model.parameters())

# Using all data for final training (no validation split)
# It is recommended to wrap this in a function instead
'''final_train = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transform = transformer,
)

final_dataloader = DataLoader(final_train, batch_size=64,shuffle=True)

# Train the model for the final time
model.train()
loss, acc = 0.0, 0.0
train_size = len(final_dataloader.dataset)
num_train_batches = len(final_dataloader)

epochs = 5

for epoch in range(epochs):
    for X, y in final_dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        batch_loss = loss_fn(pred, y)
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss += batch_loss
        acc += (pred.argmax(1)==y).type(torch.float).sum().item()
    loss /= num_train_batches
    acc /= train_size
    print(f"Epoch: {epoch+1} | Training loss: {loss:.4f} | Training acc: {acc*100:.2f}%")'''

# Measure performance
model.eval()
acc = 0.0
for X,y in test_dataloader:
  X, y = X.to(device), y.to(device)
  y_pred = model(X)
  acc += (y_pred.argmax(1)==y).type(torch.float).sum().item() # Consider using torchmetrics.
acc = acc/len(test_dataloader.dataset)*100
print(f"Accuracy final: {acc:.2f}%")