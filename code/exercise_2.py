from mse_vanilla import mean_squared_error as vanilla_mse
from mse_numpy import mean_squared_error as numpy_mse
from sklearn.metrics import mean_squared_error as sk_mse
import timeit
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2, l1

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


### TASK 1 ###
##############
observed = [2, 4, 6, 8]
predicted = [2.5, 3.5, 5.5, 7.5]

# Ensure correct argument names for different implementations
karg_vanilla_numpy = {'observed': observed, 'predicted': predicted}
karg_sk = {'y_true': observed, 'y_pred': predicted}

# Function mappings
factory = {
    'mse_vanilla': lambda: vanilla_mse(**karg_vanilla_numpy),
    'mse_numpy': lambda: numpy_mse(**karg_vanilla_numpy),
    'mse_sk': lambda: sk_mse(**karg_sk)
}

# Compute and time MSE for each method
results = {}

for name, func in factory.items():
    # Measure execution time for 100 runs
    exec_time = timeit.timeit(func, number=100) / 100
    mse = func()

    results[name] = mse
    print(f"Mean Squared Error ({name}): {mse:.6f}, "
          f"Average execution time: {exec_time:.6f} seconds")

# Validate that all MSE results match
assert len(set(results.values())) == 1, "MSE values do not match!"
print("Test is successfull!")

#### TASK 2 ####
###############

#function that makes a 1d oscillatory function with and without noise

#Ocillatory function (sine wave) with or without noise
def oscillatory_func(n_points, range, noise=False):
    x = np.linspace(0, range, n_points)
    y = np.sin(x)
    if noise:
        y += np.random.normal(0, 0.5, n_points)
    return x, y

n_points = 100
x_range = 10
x, y = oscillatory_func(n_points, x_range, noise=False)
plt.scatter(x, y)
plt.title("Oscillatory function without noise")
plt.show()

x, y = oscillatory_func(n_points, x_range, noise=True)
plt.scatter(x, y)
plt.title("Oscillatory function with noise")
plt.show()

#saving noisy data
np.save('oscillatory_data.npy', [x, y])

# Printing infoabout the data:
print(f"Data generated: {n_points} points, range: {x_range}, noise: np.random.normal(0, 0.5, n_points)")


#### TASK 3 ####
###############

# Use clustering to group data and print the variance as a function of the number of clusters


# Load noisy data correctly
x, y = np.load('oscillatory_data.npy', allow_pickle=True)

# Reshape data properly for clustering
data = np.column_stack((x, y))  # Now it's a (n_points, 2) shape

# Define the K-Means clustering model
k = 3  # Number of clusters
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)

# Fit the model to the 2D data (x, y)
kmeans.fit(data)

# Get the cluster assignments and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Plot the clustered data
plt.figure(figsize=(8, 6))
colors = ['r', 'g', 'b']

for i in range(k):
    cluster_points = data[labels == i]  # Select points in cluster i
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i], label=f'Cluster {i+1}')

    # Plot the centroid of the current cluster
    plt.scatter(centroids[i, 0], centroids[i, 1], c='black', marker='x', s=100)

plt.title('K-Means Clustering')
plt.xlabel('Feature 1 (x)')
plt.ylabel('Feature 2 (y)')
plt.legend()
plt.show()

#he variance as a function of the number of clusters:
variances = [] # Store the variance for each number of clusters
for k in range(1, 11): # Try different number of clusters
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(data)
    variances.append(kmeans.inertia_)
    print(f"Variance for {k} clusters: {kmeans.inertia_}")

#Print the info about your clustering method and its parameters
print(f"Clustering method: K-Means")
print(f"Parameters: n_clusters, random_state, n_init")
print(f"Variance: {variances}")


# Plot the variance as a function of the number of clusters
plt.plot(range(1, 11), variances, marker='o')
plt.title('Variance vs. Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Variance (Inertia)')
plt.show()


#### TASK 4 ####
###############

# Use LinearRegression, NeuralNetwork and PINNS to make a regression of such data

#Linear Regression#
###################


# Load noisy data correctly
x, y = np.load('oscillatory_data.npy', allow_pickle=True)

# Reshape data properly for Linear Regression
X = x.reshape(-1, 1)  # Reshape to (n_points, 1) for single feature
y = y.reshape(-1, 1)  # Reshape to (n_points, 1)

# Create and fit the Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X, y)

# Predict the output
y_pred_lr = lr_model.predict(X)

# Plot the Linear Regression model
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred_lr, color='red', label='Linear Regression')
plt.title('Linear Regression')
plt.xlabel('Feature (x)')
plt.ylabel('Output (y)')
plt.legend()
plt.show()


print("Task completed: Linear Regression")

#PINN and NN regression#
########################

# Load noisy oscillatory data
x, y = np.load('oscillatory_data.npy', allow_pickle=True)
x = x.reshape(-1, 1)  # Reshape for models
y = y.reshape(-1, 1)

# Convert to PyTorch tensors
x_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Normalize the data
scaler_x = StandardScaler()
scaler_y = StandardScaler()
x_scaled = torch.tensor(scaler_x.fit_transform(x_tensor), dtype=torch.float32)
y_scaled = torch.tensor(scaler_y.fit_transform(y_tensor), dtype=torch.float32)

# PINN model
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

pinn_model = PINN()
pinn_optimizer = optim.Adam(pinn_model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Define Neural Network model
nn_model = Sequential([
    Dense(64, input_dim=1, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1)  # Output layer
])
nn_model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')

# Train models & track progress
num_epochs = 200
loss_history_nn = []
loss_history_pinn = []

for epoch in range(1, num_epochs + 1):
    # NN Training
    history = nn_model.fit(x, y, epochs=1, verbose=0, batch_size=32)
    loss_history_nn.append(history.history['loss'][0])
    
    # PINN Training
    pinn_optimizer.zero_grad()
    y_pred_pinn = pinn_model(x_scaled)
    loss = criterion(y_pred_pinn, y_scaled)
    loss.backward()
    pinn_optimizer.step()
    loss_history_pinn.append(loss.item())
    
    # Plot progress at specific epochs
    if epoch % 50 == 0:
        y_pred_nn = nn_model.predict(x)
        plt.figure(figsize=(8, 5))
        plt.scatter(x, y, color='blue', label='Actual Data', alpha=0.3)
        plt.plot(x, y_pred_nn, label=f'NN Prediction (Epoch {epoch})', linestyle='solid', color='red')
        plt.plot(x, scaler_y.inverse_transform(y_pred_pinn.detach().numpy()), label=f'PINN Prediction (Epoch {epoch})', linestyle='dotted', color='purple')
        plt.title(f'NN & PINN Regression at Epoch {epoch}')
        plt.xlabel('Feature (x)')
        plt.ylabel('Output (y)')
        plt.legend()
        plt.show()

print("Task completed: NN and PINN Regression")
# Plot training loss over epochs
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs + 1), loss_history_nn, label='NN Training Loss')
plt.plot(range(1, num_epochs + 1), loss_history_pinn, label='PINN Training Loss', linestyle='dotted')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()




