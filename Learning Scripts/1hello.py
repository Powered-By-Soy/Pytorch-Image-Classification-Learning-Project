import torch
# x = torch.rand(5, 3)
# print(x)
# print(torch.cuda.is_available())
# print(torch.__version__)
# scalar = torch.tensor(7)
# print(scalar)
# print(scalar.item())
# print(scalar.ndim)
# vector = torch.tensor([7, 7])
# print(vector)
# print(vector.ndim)
# print(vector.shape)
# MATRIX = torch.tensor([[7, 8], [9, 10]])
# print (MATRIX)
# print(MATRIX.ndim)
# print(MATRIX.shape)


# TENSOR = torch.tensor([[[1,2,3],[3,6,9],[2,4,5]]])

# print(TENSOR)

# print(TENSOR.ndim)
# print(TENSOR.shape)
# print(TENSOR[0])

#random tensor

'''
random tensors good because neural networks start with tensors with random numbers and callibrate to better represent data
'''

random_tensor = torch.rand(3,4)

print(random_tensor)


random_tensor2 = torch.rand(2,3,4)

print(random_tensor2)


random_img = torch.rand(224,224,3) #height width color

print(random_img.shape, random_img.ndim)

zeros = torch.zeros(size=(3,4))
#.ones
print(zeros)

#range of tensors

one_to_ten = torch.arange(0,11)

print("arange \n\n")

print(torch.arange(start=0,end =1000, step =77))
#finish at end - step

#creating tensors like
ten_zeros = torch.zeros_like(input = one_to_ten)

print(ten_zeros)










