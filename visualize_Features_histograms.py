from torch.autograd import Variable
import matplotlib.pyplot as plt

fileNameandLoc= r'''C:\...\housing-data.csv''';
tab = pd.read_csv(fileNameandLoc)

tab.info()
print(tab.head(10))

plt.subplot(2, 2, 1)
plt.hist(tab.sqft, bins=50, color='b', alpha=0.5)
plt.ylabel('sqft')
plt.subplot(2, 2, 2)
plt.hist(tab.bdrms, bins=50, color='r', alpha=0.9)
plt.ylabel('rooms')
plt.subplot(2, 2, 3)
plt.hist(tab.age, bins=50, color='g', alpha=0.7)
plt.ylabel('age')
plt.subplot(2, 2, 4)
plt.hist(tab.price, bins=50, color='pink', alpha=0.9)
plt.ylabel('Moneh')
plt.show()
