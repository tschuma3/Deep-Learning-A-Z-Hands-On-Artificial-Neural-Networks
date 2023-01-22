import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from pylab import bone, pcolor, colorbar, plot, show

#Importing the dataset
dataset = pd.read_csv(r'Unsupervised Deep Learning Models\Self Organizing Maps (SOMs)\Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Feature Scaling
sc = MinMaxScaler(feature_range=(0, 1))
X = sc.fit_transform(X)

#Training the SOM
som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)

#Visualizing the results
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5, 
        w[1] + 0.5,
        markers[y[i]],
        markeredgecolor=colors[y[i]],
        markerfacecolor='None',
        markersize=10,
        markeredgewidth=2)
show()

#Finding the frauds
mapping = som.win_map(X)
#Change the coordinates in the mapping dictionary '1, 1' and '4, 1' to the outlined winning nodes (show as a white square on the graph) 
frauds = np.concatenate((mapping[(1, 1)], mapping[(4, 1)]), axis=0)
frauds = sc.inverse_transform(frauds)

#Printing the Fraund Clients
print('Fraud Customer IDs')
for i in frauds[:, 0]:
    print(int(i))