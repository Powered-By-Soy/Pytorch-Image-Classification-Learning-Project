import torch
from torch import nn


import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt

'''
torchvision. modules
datasets -get datasets and data loading functions for computer vision
models - get pre trained computer vision models
transform - functions for manipulating vision data to be suitable with an ML model

torch.utils.data. modules
Dataset - base dataset class for pytorch
DataLoader- creates python iterable over dataset

'''

train_data = datasets.FashionMNIST(
    root = "data", #where to download data to
    train = True, #do we want test or training data
    download = True, #do we want to download
    transform=torchvision.transforms.ToTensor(),#how do we want to transform data
    target_transform=None #how do we want to transform the labels
)

test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform=ToTensor(),
    target_transform=None
)

print(len(train_data), len(test_data))
image, label = train_data [0]
print(image, label)


class_names = train_data.classes
print(class_names)
print(image.shape, label)

#c H W
# colour channel is size 1 because grey scale
# plt.imshow(image.squeeze(), cmap = "gray")
# plt.show()


# torch.manual_seed(42)
# fig = plt.figure(figsize=(9,9))
# rows, cols = 4, 4
# for i in range(1, rows*cols+1):
#     random_idx = torch.randint(0,len(train_data), size=[1]).item()
#     # print(random_idx)
#     img, label = train_data[random_idx]
#     fig.add_subplot(rows, cols, i)
#     plt.imshow(img.squeeze(), cmap = "gray")
#     plt.title(class_names[label])
#     plt.axis(False)

# plt.show()


from torch.utils.data import DataLoader

BATCH_SIZE = 32

train_dataloader = DataLoader(dataset=train_data, batch_size = BATCH_SIZE, shuffle = True)

test_dataloader = DataLoader(dataset=test_data, batch_size = BATCH_SIZE, shuffle = False)

print(train_dataloader, test_dataloader)
print(len(train_dataloader), len(test_dataloader))


train_features_batch, train_labels_batch = next(iter(train_dataloader))

print(train_features_batch.shape, train_labels_batch.shape)

# torch.manual_seed(42)
# random_idx = torch.randint(0, len(train_features_batch), size=[1]).item()
# img, label = train_features_batch[random_idx], train_labels_batch[random_idx]
# plt.imshow(img.squeeze(), cmap = "gray")
# plt.title(class_names[label])
# plt.axis(False)
# plt.show()

#create a flatten layer
flatten_model = nn.Flatten()

x = train_features_batch[0]
print(x.shape)

output = flatten_model(x)

print(output.shape)
#color, h , w -> color, h*w

class FashionMNISTModelV0(nn.Module):
    def __init__(self,
                input_shape:int,
                hidden_units: int,
                output_shape:int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = input_shape, out_features = hidden_units),
            nn.Linear(in_features = hidden_units, out_features = output_shape)
        )

    def forward(self, x):
        return self.layer_stack(x)

torch.manual_seed(42)

model_0 = FashionMNISTModelV0(input_shape = 784,#784 = 28*28
                                hidden_units= 10,
                                output_shape = len(class_names)# one for every class name
                                ).to("cpu")


from helper_functions import accuracy_fn, eval_model

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(params = model_0.parameters(), lr = 0.1)

from timeit import default_timer as timer

def print_train_time(start: float, end: float, device: torch.device = None):
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time

# start_time = timer()

# end_time = timer()
# print_train_time(start = start_time, end=end_time, device="cpu")


from tqdm.auto import tqdm

torch.manual_seed(42)

train_time_start_on_cpu = timer()

epochs = 3

for epoch in tqdm(range(epochs)):
    
    print(f"Epoch {epoch}")

    #calculate training loss per batch
    train_loss = 0

    #image, label
    for batch, (X,y) in enumerate(train_dataloader):
        model_0.train()

        y_pred = model_0(X)

        loss = loss_fn(y_pred, y)

        train_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #update once per batch

        if batch %400 == 0:
            print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples")

    train_loss /= len(train_dataloader)

    #testing

    test_loss, test_acc = 0, 0

    model_0.eval()

    with torch.inference_mode():
        for X_test, y_test in test_dataloader:

            test_pred = model_0(X_test)

            test_loss += loss_fn(test_pred, y_test)

            test_acc += accuracy_fn(y_true = y_test, y_pred=test_pred.argmax(dim=1))

        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)
    print(f"\nTrain loss: {train_loss:.4f} | Train loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")

    train_time_end_on_cpu = timer()
    total_train_time_model_0 = print_train_time(start = train_time_start_on_cpu, 
                                                end = train_time_end_on_cpu, 
                                                device = str(next(model_0.parameters())))

    print(next(model_0.parameters()).device)

torch.manual_seed(42)

model_0_results = eval_model(model=model_0, 
                            data_loader= test_dataloader,
                            loss_fn = loss_fn,
                            accuracy_fn = accuracy_fn,
                            device = "cpu")
print(model_0_results)




'''
epoch time
12.016 seconds
20.731 seconds
30.060 seconds
{'model_name': 'FashionMNISTModelV0', 'model_loss': 0.4766388535499573, 'model_acc': 83.42651757188499}
'''





