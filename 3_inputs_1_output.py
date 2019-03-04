import torch
from torch.autograd import Variable
import pandas as pd
#import matplotlib.pyplot as plt
#import numpy as np
from sklearn.model_selection import train_test_split


#####Preparing the data ###
fileNameandLoc= r'''C:\...\data\housing-data.csv''';
tab = pd.read_csv(fileNameandLoc)
traindf, testdf = train_test_split(tab, test_size=0.2)
xdata = Variable(torch.Tensor(traindf.iloc[:,0:3].values)) #Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
ydata = Variable(torch.Tensor(traindf.iloc[:,3:].values)) #Variable(torch.Tensor([[2.0], [4.0], [6.0]]))
xtdata = Variable(torch.Tensor(testdf.iloc[:,0:3].values)) #test input data
ytdata = Variable(torch.Tensor(testdf.iloc[:,3:].values)) #test output data
#### Normalizing Data , otherwise the vlaues are too big and the optimisation will fail (nan nan inf inf)
x_data=xdata #initialization
xt_data=xtdata #initialization
for cnt in range(3):
    x_data[:, cnt] = (xdata[:, cnt] - xdata[:, cnt].min()) / (xdata[:, cnt].max() - xdata[:, cnt].min())
    xt_data[:, cnt] = (xtdata[:, cnt] - xtdata[:, cnt].min()) / (xtdata[:, cnt].max() - xtdata[:, cnt].min())
'''
x_data[:,0]=(xdata[:,0]-xdata[:,0].min())/(xdata[:,0].max()-xdata[:,0].min())
x_data[:,1]=(xdata[:,1]-xdata[:,1].min())/(xdata[:,1].max()-xdata[:,1].min())
x_data[:,2]=(xdata[:,2]-xdata[:,2].min())/(xdata[:,2].max()-xdata[:,2].min())
'''
y_data=(ydata-ydata.min())/(ydata.max()-ydata.min())
yt_data=(ytdata-ytdata.min())/(ytdata.max()-ytdata.min())
####

##### ML part : ###########
''' Create a Class - Declare your Forward Pass - Tune the HyperParameters'''
class Model(torch.nn.Module):

    def __init__(self): #initializing
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(3, 1)  # one input/feature , one output
        # here where other NN layers are added

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        y_pred = self.linear(x)
        return y_pred

# save our model
model = Model()


# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Training loop
for epoch in range(500):
        # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data)

    # Compute and print loss
    loss = criterion(y_pred, y_data)
    print('epoch {}, loss {}',epoch, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# After training : testing with one value #########
[w, b] = model.parameters()  #weight and bias
#print(w)
#print(b)

testval= torch.Tensor(x_data)

hour_var = Variable(testval)
y_pred = model(hour_var)

errorT=torch.sqrt(torch.mean((y_data-y_pred)**2)) #RMSE of error for the training dataset
print('RMSE of error for the training dataset',errorT.data)
hour_var = Variable(torch.Tensor(xt_data))
y_pred = model(hour_var)
errort= torch.sqrt(torch.mean((yt_data-y_pred)**2)) #RMSE of error for the test dataset
print('RMSE of error for the test dataset',errort.data)
