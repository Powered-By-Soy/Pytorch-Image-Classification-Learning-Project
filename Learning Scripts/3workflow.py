import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np


# import nn
# neural network
#pytorch building blocks

#linear regression

#Y = a + bX
weight = 0.7    # b
bias = 0.3      # a

start = 0
end = 1
step = 0.02
#tensor / matrix hence capital
X = torch.arange(start, end, step).unsqueeze(dim=1)

y = weight * X + bias
print(X[:10], y[:10])


#create a train/test split
train_split = int(0.8 * len(X))

X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

print(len(X_train), len(y_train), len(X_test), len(y_test))


def plot_predictions(train_data = X_train,
                        train_labels = y_train,
                        test_data = X_test,
                        test_labels = y_test,
                        predictions = None):

    plt.figure(figsize=(10,7))

    plt.scatter(train_data, train_labels, c="b", s=4, label = "Training Data")

    plt.scatter(test_data, test_labels, c="g", s=4, label = "Testing Data")


    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    plt.legend(prop={"size": 14})
    plt.show()

# plot_predictions()


#model building
#linear regression model class

#almost everything in pytorch inherits from nn.Module
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

        #forward method to define computation in model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias
        #start with random values and look at training data to adjust
        #uses gradient descent(requires_grad=true) and back propagation

#model building essentials
'''
torch.nn - all buildings for computational graph
torch.nn.Parameter - what parameters model tries and learn
torch.nn.Module -base class of all neural network modules
torch.optim - where the optimizers in pytorch live, helps with gradient descent
def forward() - all nn.Module subclassses requires overwriting forward method
'''

#torchvision.models pre built model

#checking pytorch model
torch.manual_seed(42)

model_0 = LinearRegressionModel()

print(model_0)
print(list(model_0.parameters()))
# print(model_0.parameters)


#list named parameters
print(model_0.state_dict())

#making prediction using torch.inference_mode()
#inference disables training, better for computation
with torch.inference_mode():
    y_preds = model_0(X_test)


# print(y_preds)

plot_predictions(predictions = y_preds)

# measuring how wrong predicrtions are, use loss function

#loss

loss_fn = nn.L1Loss()

#optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01) 
#stochastic gradient descent
#lr learning rate

#training and testing loops
#loop
#forward pass
#calculate loss
#optimizer zero grade
# loss backward
# optimizer step

#epoch is one loop through data

epochs = 200

#track different values
epoch_count = []
loss_values = []
test_loss_values = []


for epoch in range (epochs):
    #set model to training mode
    model_0.train()
    #enables gradients tracking
    
    #forward pass
    y_pred = model_0(X_train)

    #loss
    loss = loss_fn(y_pred, y_train)
    # print(f"Loss: {loss}")


    #optimize zero grad
    optimizer.zero_grad()

    #perform backpropagation
    loss.backward()

    #step optimizer
    optimizer.step()


    model_0.eval()# disables gradients tracking(exit training)

    #testing

    with torch.inference_mode():
        #forward pass
        test_pred = model_0(X_test)

        #calculate loss
        test_loss = loss_fn(test_pred, y_test)

    if epoch % 10 == 0:
        epoch_count.append(epoch)
        loss_values.append(loss)
        test_loss_values.append(test_loss)
        print(f"Epoch : {epoch} | Loss: {loss} | Test loss: {test_loss}")
        print(model_0.state_dict())

print(model_0.state_dict())



with torch.inference_mode():
    y_preds = model_0(X_test)

plot_predictions(predictions = y_preds)

# np.array(torch.tensor(loss_values).numpy()), test_loss_values

plt.plot(epoch_count, np.array(torch.tensor(loss_values).numpy()), label ="Train loss")
plt.plot(epoch_count, test_loss_values, label ="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()

# print(epoch_count)
# print(loss_values)
# print(test_loss_values)




#saving model
'''
1. torch.save() save pytorch object in pickle format
2. torch.load() load saved object
3. torch.nn.Module.load_state_dict() allows to load a models saved state dictionary

recommended save state dict

other method save whole model
'''
from pathlib import Path

#create model directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents = True, exist_ok = True)

#create model save path
#.pt or .pth extension for models

MODEL_NAME = "01_pytorch_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

print(MODEL_SAVE_PATH)

#save model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(), f=MODEL_SAVE_PATH)

#loading state dict
loaded_model_0 = LinearRegressionModel()
print(loaded_model_0.state_dict())
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
print(loaded_model_0.state_dict())

loaded_model_0.eval()
with torch.inference_mode():
        loaded_model_preds = loaded_model_0(X_test)

model_0.eval()
with torch.inference_mode():
    y_preds = model_0(X_test)

print(y_preds == loaded_model_preds)
