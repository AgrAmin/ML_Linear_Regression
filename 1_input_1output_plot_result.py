import torch
from torch.autograd import Variable
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#####Preparing the data ###
fileNameandLoc= r'''C:\Users\Asus\Desktop\CH\colormix\seedr\course dl\practicex\zero_to_deep_learning_video\data\housing-data.csv''';
tab = pd.read_csv(fileNameandLoc)
xdata = Variable(torch.Tensor(tab.iloc[:,0:1].values)) #Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
ydata = Variable(torch.Tensor(tab.iloc[:,3:].values)) #Variable(torch.Tensor([[2.0], [4.0], [6.0]]))
#### Normalizing Data , otherwise the vlaues are too big and the optimisation will fail (nan nan inf inf)
x_data=abs(xdata-xdata.min())/(xdata.max()-xdata.min())
y_data= abs(ydata-ydata.min())/(ydata.max()-ydata.min())
x_data=torch.flip(x_data,[0])
y_data=torch.flip(y_data,[0])
####

##### ML part : ###########
''' Create a Class - Declare your Forward Pass - Tune the HyperParameters'''
class Model(torch.nn.Module):

    def __init__(self): #initializing
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # one input/feature , one output
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
    print(epoch, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# After training : testing with one value #########
[w, b] = model.parameters()  #weight and bias

testval= torch.Tensor([[1600.0]])
testval=(testval-x_data.max())/(testval-x_data.max()) #normalizing the the test value variable
hour_var = Variable(testval)
y_pred = model(hour_var)
print("predict (after training)", 4, model(hour_var).data[0][0]) #the result give a normalized value of the price
print(((y_data.max()-y_data.min())*model(hour_var).data[0][0])+y_data.min())
plt.plot(x_data.numpy(), y_data.numpy(), 'ro')
x = np.linspace(0, 1, 2)
plt.plot(x, x*w.item() + b.item(), linestyle='solid')
plt.show()
