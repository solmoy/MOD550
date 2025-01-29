
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os

class DatasetGenerator():
    ''' Class that generates noisy data around a "truth" function(sine)
    *Input:
        n_random_points ( int) - number of random points to generate
        noise (float) - amount of noise to add to the truth function
    
    *Methods:
        generate_data - generates 2D data


    '''
    def __init__(self, n_random_points, noise):
        self.n_random_points = n_random_points
        self.noise = noise 

    def generate_data(self,color='red'):
        
        def sine(x):
            return np.sin(x)

        # Truth: datapoints of sine functions 
        x = np.linspace(-2*np.pi, 2*np.pi, 100)
        y = sine(x)

        #noise around sine function
        y_noise = y + self.noise * np.random.normal(size=x.shape)

     
     
    
        #plotting the sine function and the noise
        plt.figure()
        plt.plot(x, y, label='Sine function')
        plt.scatter(x, y_noise, label='Noisy sine function', color=color)
        plt.legend()
        plt.savefig('MOD550/data/noisytruth.png')
        plt.show()
        

        dataset = pd.DataFrame({'x': x, 'y': y, 'y_noise': y_noise})
        dataset.to_csv('MOD550/data/noisytruth.csv', index=False)
        

#using the class to generate 2 datasets
dataset_1 = DatasetGenerator(100, 1)
dataset_1.generate_data(color='blue')
dataset1= pd.read_csv('MOD550/data/noisytruth.csv')

dataset_2 = DatasetGenerator(100, 1)
dataset_2.generate_data(color='green')
dataset2= pd.read_csv('MOD550/data/noisytruth.csv')

#append the datasets
combined_dataset = pd.concat([dataset1, dataset2], axis=0)
combined_dataset.to_csv('MOD550/data/combined_dataset.csv', index=False)

#plot and save the combined dataset
plt.scatter(combined_dataset['x'], combined_dataset['y_noise'], color='purple')
plt.savefig('MOD550/data/combined_dataset.png')
plt.show()

