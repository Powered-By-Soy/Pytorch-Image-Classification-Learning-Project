import torch
import numpy as np

#common errors
#1 tensors not right data type
#2 tensors not right shape
#3 tensors not right device

float_32_tensor = torch.tensor([3.0, 6.0, 9.0], 
                                dtype = None, #datatype
                                device = None, #device "cuda" tensors can live on cpu or gpu, causing error if mismatch
                                requires_grad = False) #whether to track gradient with this tensor

#dtype 'tensor.dtype'
#shape 'tensor.shape'
#device 'tensor.device'


print(float_32_tensor.dtype)

#cast
float_16_tensor = float_32_tensor.type(torch.float16)
print(float_16_tensor.dtype)

print(float_32_tensor)


print(float_32_tensor*float_16_tensor)


#addition

tensor = torch.tensor([1,2,3])
# tensor = tensor +10
print(tensor+10)
print(tensor*10)
print(tensor-10)

print(torch.mul(tensor,10))

print(torch.matmul(tensor, tensor))
#tensor @ tensor
# @ is matrix mult
#torch.mm is torch.matmul
# tensor.T transpose

#torch.min(x), x.min(), torch.max(), x.max(), torch.mean(x), x.mean()
#position min max x.argmin, x.argmax


#torch.stack
#torch.vstack torch.hstack

#squeeze to remove x dimensions
#unsqueeze to add x dimensions
#view shares same memory

x = torch.arange(1,10)

z = x.view(1,9)


z[:,0] = 5
print(z)
print(x)

x_stacked = torch.stack([x,x,x,x], dim=0)

print(x_stacked)



x_stacked = torch.stack([x,x,x,x], dim=1)

print(x_stacked)

print(x_stacked.unsqueeze(dim=1))


print(x_stacked.size())

print(x_stacked.unsqueeze(dim=1).size())


#torch.permute
#rearranges dimensions (as a view)




#numpy converison
print("numpy convert \n\n")
array = np.arange(1.0,8.0)
tensor = torch.from_numpy(array) #make new tensor using data from array(not copy)
#print(" ")
#Pytorch reflects numpy default dtype of float 64
print(array, tensor)


#tensor to numpy
tensor = torch.ones(7)

numpy_tensor = tensor.numpy()

print(tensor, numpy_tensor, numpy_tensor.dtype)
#default of torch dthpe is float 32


#reproducibility
#RANDOM_SEED = 42
#torch.manual_seed(RANDOM_SEED)
#only lasts for the next random call

#setup device agnostic code

device = "cuda" if torch.cuda.is_available else "cpu"

print(device)


#count devices
print(torch.cuda.device_count())

#https://docs.pytorch.org/docs/stable/notes/cuda.html

# import argparse
# import torch

# parser = argparse.ArgumentParser(description='PyTorch Example')
# parser.add_argument('--disable-cuda', action='store_true',
#                     help='Disable CUDA')
# args = parser.parse_args()
# args.device = None
# if not args.disable_cuda and torch.cuda.is_available():
#     args.device = torch.device('cuda')
# else:
#     args.device = torch.device('cpu')


tensor = torch.tensor([1,2,3], device = "cpu")
print(tensor, tensor.device)

#default cpu
tensor1 = torch.tensor([1,2,3])
print(tensor1, tensor1.device)


#move tesnor to gpu if available

tensor_on_gpu = tensor.to(device)
print(tensor_on_gpu)
print("index 0 because it is first gpu")
#numpy only works on cpu

# tensor_on_gpu.numpy()
tensor_back_cpu = tensor_on_gpu.cpu().numpy()

print(tensor_back_cpu)


















