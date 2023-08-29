import torch
state_dict = torch.load("/home/zhao/DataSets/s3dis/Area_1/conferenceRoom_1.pth")
print(type(state_dict))

for i in state_dict:
    print(i)
    print(type(state_dict[i]))