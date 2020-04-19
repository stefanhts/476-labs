import torch

class Model(torch.nn.Module):


    def __init__(self, D_in, H, D_out):
        super(Model,self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H,D_out)


    def forward(self, x):
        x = self.linear1(x).clamp(0)
        x = self.linear2(x)
        return x

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

model = Model(D_in, H, D_out)

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

for t in range(500):
    y_pred = model(x)

    loss = criterion(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



    #
    #
    # def __init__(self, D_in, H, D_out):
    #     super(Model,self).__init__()
    #     self.numclasses=numclasses
    #     self.conv1=torch.nn.Conv2d(28,14,(3,3))
    #     self.conv2=torch.nn.Conv2d(28,14(3,3))
    #
    #
    # def forward(self, x):
    #     x = torch.nn.funcitonal.relu(self.conv1(x))
    #     x = torch.nn.functional.relu(self.conv2(x))
    #     return x
    #






