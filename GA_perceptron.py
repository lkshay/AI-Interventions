# -*- coding: utf-8 -*-

def perceptron(x1,x2,w1,w2,b):
  if(0<(x1*w1+x2*w2+b)):
    return(1)
  else:
    return(0)

#Import the dataset

from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot
from pandas import DataFrame
Data_set_size=200
X, Y = make_blobs(n_samples=Data_set_size, centers=2, n_features=2,cluster_std=1.0, center_box=(-4.0, 4.0),random_state=1)
df = DataFrame(dict(x=X[:,0], y=X[:,1], label=Y))
colors = {0:'red', 1:'blue'}
fig, ax = pyplot.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
pyplot.show()

import numpy as np
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot
from pandas import DataFrame
Data_set_size=200
X, Y = make_blobs(n_samples=Data_set_size, centers=2, n_features=2,cluster_std=1.0, center_box=(-4.0, 4.0),random_state=1)
df = DataFrame(dict(x=X[:,0], y=X[:,1], label=Y))
colors = {0:'red', 1:'blue'}
fig, ax = pyplot.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
pyplot.show()

np.random.seed()



population_size=35                            
population=np.random.rand(population_size,3)*8-4  # random values being assigned for w1,w2 and b    
                                                  # here we define a numpy array for the population, and randomly initilize it. 
                                                   # notice that we are initializing it within the solution space we are looking at, [-4 4]
                                                   # there are population_size of individuals, so we have the number of rows is population_size.
                                                   # each individual has two variables, x1 and x2, which relate to coordinates in the solution spcae. 
                                                   #therefore we have two columns.
new_population=np.zeros((population_size,3))      #----------------------------similarly for w1,w2 and b 
                                                  # this is a temporary place to hold the new generation while we are creating them from the current gen (population)
                                                   # when we are done creating the new generation, we copy the individuals from new_population to population.
tournament_size= 9                                 # we use tournment selection to select who would reproduce. tournament_size is the size of the tournament.                  
select_potential_parents=np.zeros((tournament_size,3))  # ---------------------------------similarly for w1,w2 and b 
                                                        # this is where all potential parents selected to attend the tournament are hold.
max_generation_num=15                              # this says how many generation we should simulate. 
mutation_frac=0.4                                  # this says what fraction of new generation should be mutated.
mutation_scale=1                                 # this is the standard deviation of the noise that is added to 
                                                   #mutation_frac of the new generation that are randomly chosen to be mutated. 


def perceptron(x1,x2,w1,w2,b):

  if(0<(x1*w1+x2*w2+b)):
    return(1)
  else:
    return(0)

def accuracy_eval(Parent,X,Y):
  
  score = 0
  for i in range(len(Y)):
    
    if(perceptron(X[i,0],X[i,1],Parent[0], Parent[1], Parent[2]) == Y[i]):
      
      score = score + 1
      
    else:
      
      score = score
      
  return(score)


def crossover(a,b):              # this function implements the corssover operation, it recives parents a and b, and produces the child c! 
  c=np.random.rand(3)
  beta=np.random.rand(1)
  c[0]=beta*a[0]+(1-beta)*b[0]
  beta=np.random.rand(1)
  c[1]=beta*a[1]+(1-beta)*b[1]
  beta=np.random.rand(1)
  c[2] = beta*a[2] + (1-beta)*b[2]
  return(c)


def mutation(new_population):    
  num_of_mutation=math.ceil(len(new_population)*mutation_frac)
  mutation_index=np.random.choice(len(new_population),num_of_mutation, replace=False, p=None)
  new_population[mutation_index,:]=new_population[mutation_index,:]+np.random.normal(0,mutation_scale,(num_of_mutation,3))    
  return(new_population)
  
  

for i in range(0, max_generation_num):      # This is your generation loop... by looping this you are going through generation after generation.
  
  for j in range(0,population_size):     # This is your new population loop. At each loop you create a new instance for the next population. Therefore this loops population_size times.
    
    select_potential_parents=population[np.random.choice(len(population), size=tournament_size, replace=False)] # this is where we select some potential parents randomly
                                                                                                                # and let them compete against each other in a tournament.
                                                                                                                # the winner is simply the one who is the most fitted!
                                                                                                              
                                                                                                                
                                                          
    w1=select_potential_parents[:,0] # this is just a hack that I used to manage to send a vector to f instead of looping. Can you combine this line with the 
                                  #next line into the third line altogether? There should be a way...  
    w2=select_potential_parents[:,1]
    
    b = select_potential_parents[:,2]

    score_list = np.zeros(len(select_potential_parents))
    
    for i in range(len(select_potential_parents)):
      score_list[i] = accuracy_eval(select_potential_parents[i],X,Y)
    
    
    parent_1 = select_potential_parents[np.argmax(score_list)]
    
    w1=select_potential_parents[:,0]   
    w2=select_potential_parents[:,1]
    
    b = select_potential_parents[:,2]
    
    for i in range(len(select_potential_parents)):
      score_list[i] = accuracy_eval(select_potential_parents[i],X,Y)
    
    parent_2 = select_potential_parents[np.argmax(score_list)]

    
    new_population[j,:]=crossover(parent_1,parent_2) 
  
  
  new_population=mutation(new_population)  

  
  population=new_population.copy()                         #we replace the current generation with this new generation that we just created. 
  print(population[np.argmax(score_list)])
  eligible = population[np.argmax(score_list)]
  for m in range(len(Y)):
    plt.plot(X[m,0],X[m,1],"bo" if (Y[m] == 1) else "ro")
  for k in (np.linspace(-5,5,num=100)):
    y = -(eligible[2]/eligible[1])/(eligible[2]/eligible[0])*k + (-eligible[2]/eligible[1])
    plt.plot(k, y,'ko')

    score_list = np.zeros(population_size)
  plt.show()

for i in range(population_size):
  score_list[i] = (accuracy_eval(population[i],X,Y)/len(Y)) * 100

print(score_list)

print(population[np.argmax(score_list)])

eligible = population[np.argmax(score_list)]
for m in range(len(Y)):
    plt.plot(X[m,0],X[m,1],"bo" if (Y[m] == 1) else "ro")
for i in (np.linspace(-5,5,num=100)):
  y = -(eligible[2]/eligible[1])/(eligible[2]/eligible[0])*i + (-eligible[2]/eligible[1])
  plt.plot(i, y,'ko')

  score_list = np.zeros(population_size)
plt.show()

