import six
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
from activation import *

# import all four local random search algorithms 
from algorithms import *
from decay import *
from fitness import *
from neural import *

# import all optimization problems 
from opt_probs import *


# example 1. queens using predefined fitness function
# Initialize fitness function object using pre-defined class
fitness = Queens()
# Define optimization problem object
problem = DiscreteOpt(length = 8, fitness_fn = fitness, maximize=False, max_val=8)
# Define decay schedule
schedule = ExpDecay()
# Solve using simulated annealing - attempt 1         
init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])
best_state, best_fitness = simulated_annealing(problem, schedule = schedule, max_attempts = 10, 
                                                      max_iters = 1000, init_state = init_state,
                                                      random_state = 1)
print('The best state found is: ', best_state)
print('The fitness at the best state is: ', best_fitness)
# Solve using simulated annealing - attempt 2
best_state, best_fitness = simulated_annealing(problem, schedule = schedule, max_attempts = 100, 
                                                      max_iters = 1000, init_state = init_state,
                                                      random_state = 1)
print(best_state)
print(best_fitness)


# example 2. queens using custom fitness function
# Define alternative N-Queens fitness function for maximization problem
def queens_max(state):
    
    # Initialize counter
    fitness = 0
    
    # For all pairs of queens
    for i in range(len(state) - 1):
        for j in range(i + 1, len(state)):
            
            # Check for horizontal, diagonal-up and diagonal-down attacks
            if (state[j] != state[i]) \
                and (state[j] != state[i] + (j - i)) \
                and (state[j] != state[i] - (j - i)):
                
                # If no attacks, then increment counter
                fitness += 1

    return fitness

# Check function is working correctly
state = np.array([1, 4, 1, 3, 5, 5, 2, 7])

# The fitness of this state should be 22
queens_max(state)
# Initialize custom fitness function object
fitness_cust = CustomFitness(queens_max)
# Define optimization problem object
problem_cust = DiscreteOpt(length = 8, fitness_fn = fitness_cust, maximize = True, max_val = 8)
# Solve using simulated annealing - attempt 1
best_state, best_fitness = simulated_annealing(problem_cust, schedule = schedule, 
                                                      max_attempts = 10, max_iters = 1000, 
                                                      init_state = init_state, random_state = 1)
print(best_state)
print(best_fitness)

# Travelling Salesperson Using coordinate-Defined Fitness function
# Create list of city coordinates
coords_list = [(1, 1), (4, 2), (5, 2), (6, 4), (4, 4), (3, 6), (1, 5), (2, 3)]

# Initialize fitness function object using coords_list
fitness_coords = TravellingSales(coords = coords_list)
# Define optimization problem object
problem_fit = TSPOpt(length = 8, fitness_fn = fitness_coords, maximize = False)
# Solve using genetic algorithm - attempt 1
best_state, best_fitness = genetic_alg(problem_fit, random_state = 2)
print(best_state)
print(best_fitness)

# fitting a neural network to the iris Dataset
# Load the Iris dataset
data = load_iris()
# Get feature values of first observation
print(data.data[0])
# Get feature names
print(data.feature_names)
# Get target value of first observation
print(data.target[0])
# Get target name of first observation
print(data.target_names[data.target[0]])
# Get target name of first observation
print(data.target_names[data.target[0]])
# Get maximum feature values
print(np.max(data.data, axis = 0))
# Get unique target values
print(np.unique(data.target))
# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size = 0.2, 
                                                    random_state = 3)

# Normalize feature data
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# One hot encode target values
one_hot = OneHotEncoder()

y_train_hot = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).todense()

# Initialize neural network object and fit object - attempt 1
nn_model1 = NeuralNetwork(hidden_nodes = [2], activation ='relu', 
                                 algorithm ='random_hill_climb', 
                                 max_iters = 1000, bias = True, is_classifier = True, 
                                 learning_rate = 0.0001, early_stopping = True, 
                                 clip_max = 5, max_attempts = 100, random_state = 3)

nn_model1.fit(X_train_scaled, y_train_hot)
# Predict labels for train set and assess accuracy
y_train_pred = nn_model1.predict(X_train_scaled)

y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)

print(y_train_accuracy)
# Predict labels for test set and assess accuracy
y_test_pred = nn_model1.predict(X_test_scaled)

y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)

print(y_test_accuracy)
# Initialize neural network object and fit object - attempt 2
nn_model2 = NeuralNetwork(hidden_nodes = [2], activation = 'relu', 
                                 algorithm = 'gradient_descent', 
                                 max_iters = 1000, bias = True, is_classifier = True, 
                                 learning_rate = 0.0001, early_stopping = True, 
                                 clip_max = 5, max_attempts = 100, random_state = 3)

nn_model2.fit(X_train_scaled, y_train_hot)

# Predict labels for train set and assess accuracy
y_train_pred = nn_model2.predict(X_train_scaled)

y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)

print(y_train_accuracy)

# Predict labels for test set and assess accuracy
y_test_pred = nn_model2.predict(X_test_scaled)

y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)

print(y_test_accuracy)




