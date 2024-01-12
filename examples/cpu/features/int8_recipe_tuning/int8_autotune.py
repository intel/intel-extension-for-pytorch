import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import intel_extension_for_pytorch as ipex

########################################################################  # noqa F401
# Reference for training portion:
# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)
batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=1)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}    [{current:>5d}/{size:>5d}]")


model, optimizer = ipex.optimize(model, optimizer=optimizer)

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
print("Done!")

################################ QUANTIZE ##############################  # noqa F401
model.eval()

def evaluate(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    accuracy = 0
    with torch.no_grad():
        for X, y in dataloader:
            # X, y = X.to('cpu'), y.to('cpu')
            pred = model(X)
            accuracy += (pred.argmax(1) == y).type(torch.float).sum().item()
    accuracy /= size
    return accuracy

######################## recipe tuning with INC ########################  # noqa F401
def eval(prepared_model):
    accu = evaluate(test_dataloader, prepared_model)
    return float(accu)

tuned_model = ipex.quantization.autotune(model, test_dataloader, eval_func=eval, sampling_sizes=[100],
                                         accuracy_criterion={'relative': .01}, tuning_time=0)
########################################################################  # noqa F401

# run tuned model
data = torch.randn(1, 1, 28, 28)
convert_model = ipex.quantization.convert(tuned_model)
with torch.no_grad():
    traced_model = torch.jit.trace(convert_model, data)
    traced_model = torch.jit.freeze(traced_model)
    traced_model(data)

# save tuned qconfig file
tuned_model.save_qconf_summary(qconf_summary="tuned_conf.json")

print("Execution finished")
