# from torch.utils.data import  DataLoader
import torch
#
# class MyDataset():
#     def __init__(self, data1,data2):
#         self.data1 = data1
#         self.data2 = data2
#
#     def __len__(self):
#         return len(self.data1)
#
#     def __getitem__(self, idx):
#         ret={'data1':self.data1[idx],'data2':self.data2[idx]}
#         return ret
#
#
# # 创建自定义数据集实例
# my_data1 = torch.eye(8).reshape(4,4,4)
# my_data2 = torch.eye(4).reshape(4,1,4)
# my_dataset = MyDataset(my_data1,my_data2)
#
# # 使用DataLoader加载自定义数据集my_dataset
# dataloader = DataLoader(dataset=my_dataset,batch_size=2)
#
# for iteration,data in enumerate(dataloader):
#     print(data.shape)

R=torch.tensor([[-0.99,0.09,-0.07],[0.08,0.01,-0.996],[-0.096,-0.995,-0.02]])
center=torch.tensor([-32.,5.,-4.55])
T=- R @ center
print(T)