import torch

x=torch.randn(20)
print(x)
l1=sum(x)/len(x)
l1=l1.item()
print(l1)