import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
credit_card_df = pd.read_csv('Credit_Card_Applications.csv')

# Split the data into two sets
X_set = credit_card_df.iloc[:, :-1].values
y_set = credit_card_df.iloc[:, -1].values

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
X_set = sc.fit_transform(X_set)

# Training the SOM
from minisom import MiniSom
som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)
som.random_weights_init(X_set)
som.train_random(data=X_set, num_iteration=100)

# Visualizing the result
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']

for i, x in enumerate(X_set):
    winning_node = som.win_map(x)
    plot(winning_node[0]+0.5, winning_node[1]+0.5, markers[y[i]], markeredgecolor = colors[y[i]])