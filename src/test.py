import torch

torch.cuda.set_device("cuda:"+str(0))

d = {
    "a": torch.Tensor([1,2,3]),
    "b": torch.Tensor([3,4,5]),
    "c": torch.Tensor([4,5,6])
}

print("before")
for x in d.values():
    x = x.cuda()

for x in d.values():
    print(x.get_device())

print("after")
for x in d.keys():
    d[x] = d[x].cuda()

for x in d.values():
    print(x.get_device())

print("back to cpu")
for x in d.values():
    x = x.cpu()

for x in d.values():
    print(x.get_device())

print("back to cpu way2")
for x in d.keys():
    d[x] = d[x].cpu()

for x in d.values():
    print(x.get_device())
