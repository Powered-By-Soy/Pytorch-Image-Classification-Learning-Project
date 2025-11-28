import torch
import matplotlib.pyplot as plt
from torch import nn

#device agnostic code
#use GPU if have GPU

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

#create data
weight = 0.7
bias = 0.3

#create range values
start = 0
end = 1
step = 0.02

X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias
print(X[:10], y[:10])


train_split = int(0.8*len(X))
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
# plot_predictions(X_train, y_train, X_test, y_test)


#building linear model
class LinearRegressionModel1V2(nn.Module):
    def __init__(self):
        super().__init__()
        #use nn.Linear() to create model params
        self.linear_layer = nn.Linear(in_features=1, out_features=1)#input of size 1, output of size 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)
#set manual seed
torch.manual_seed(42)
model_1 = LinearRegressionModel1V2()
print(model_1,model_1.state_dict())

#check model device

print(next(model_1.parameters()).device)
model_1.to(device)
print(next(model_1.parameters()).device)

#training
loss_fn = nn.L1Loss()

optimizer = torch.optim.SGD(params=model_1.parameters(),lr=0.01)

#training loo
torch.manual_seed(42)

epochs = 200

#put data on target device
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

for epoch in range(epochs):
    
    model_1.train()
    #forward pass
    y_pred = model_1(X_train)

    #calc loss
    loss = loss_fn(y_pred, y_train)

    #optimizer 0 grad
    optimizer.zero_grad()

    #back propagation
    loss.backward()

    #optimizer step
    optimizer.step()



    #TESTING
    model_1.eval()
    with torch.inference_mode():
        test_pred = model_1(X_test)

        test_loss = loss_fn(test_pred, y_test)
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}")


#predictions
model_1.eval()

with torch.inference_mode():
    y_preds = model_1(X_test)



plot_predictions(predictions=y_preds.cpu())







