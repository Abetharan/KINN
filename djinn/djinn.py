import tensorflow as tf 
import numpy as np 
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from collections import Counter


# The tree structure can be traversed to compute various properties such
# as the depth of each node and whether or not it is a leaf.
def printUsefulTree(n_nodes, node_depth, is_leaves):
  #Print Check
  for i in range(n_nodes):
    if is_leaves[i]:
      print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
    else:
      print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
            "node %s."
            % (node_depth[i] * "\t",
                i,
                children_left[i],
                feature[i],
                threshold[i],
                children_right[i],
                ))
  print()

def getTreeInfo(children_left, children_right):
  node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
  is_leaves = np.zeros(shape=n_nodes, dtype=bool)
  stack = [(0, -1)]  # seed is the root node id and its parent depth
  while len(stack) > 0:
      node_id, parent_depth = stack.pop()
      node_depth[node_id] = parent_depth + 1

      # If we have a test node
      if (children_left[node_id] != children_right[node_id]):
          stack.append((children_left[node_id], parent_depth + 1))
          stack.append((children_right[node_id], parent_depth + 1))
      else:
          is_leaves[node_id] = True

  return(is_leaves, node_depth)

def returnBranchCount(node_depth):
  ###NUMBER OF BRANCHES PER LEVELS:
  ### Number of UNIQUES of branch depth
  #Uniques levels
  counts = dict(Counter(node_depth))
  print(counts)
  print("The number of uniques levels are : ")
  print(counts.keys()) # equals to list(set(words))
  print("The number of entries per levels are : ")
  values = counts.values() # counts the elements' frequency
  print(values)
  for key, value in counts:
    value *=0.5 #Individuals branch counted as one thus half.
  print(counts)

  if counts["0"] == 0.5:
    return(counts[1:])
  else:
    return counts

def returnFeatureDepth(n_nodes, is_leaves, feature, node_depth):
## Feature propagation .. FIND max level at which feature is represented.
  feature_depth = {}
  for i in range(n_nodes):
    if is_leaves[i]:
      pass
    else:
      if str(feature[i]) in feature_depth:
        if(feature_depth[str(feature[i])] > node_depth[i]):
          pass
        else:
          feature_depth[str(feature[i])] = node_depth[i]
      else:
        feature_depth[str(feature[i])] = node_depth[i]
    return feature_depth

#Normal Distribution with custom variance and zero mean
def djinn_norm_dist(n_prev, n_curr):
  mean = 0
  var = 3 / (n_prev, n_curr)
  return(gauss(mean, math.sqrt(var)))

def initNN(no_inputs, no_outputs, no_branches, all_weights, all_biases):
  nn_input = keras.Input(shape = (no_inputs), name ="Latent_space_input")
  no_dense = no_inputs
  for i, nb in enumerate(no_branches):
    no_dense = no_dense + nb
    weights = tf.constant_initializer(all_weights[i])
    bias = tf.constant_initializer(all_bias[i])
    if i == 0:
      x = layers.Dense(no_dense, activation= "relu", kernel_initializer= all_weights, 
                       bias_initializer=all_bias)(nn_input)
    else:
      x = layers.Dense(no_dense, activation= "relu")(x)
  nn_output = layers.Dense(no_outputs)(x)
  nn = keras.Model(nn_input, nn_output, name ="decoder")
  nn.summary()
  return nn

##Weight algorithm
#Create network i.e. just how many nodes where 
#Initialise the weights following the algorihmn in the paper 
#Recall shape is (n_in, n_curr) for tensorflow
def initWeights(no_inputs, no_outputs, no_branches):
  #Create Empty array
  all_weights = np.array([])  
  no_dense = no_inputs
  for i, nb in enumerate(no_branches):
    pre_dense = no_dense
    curr_dense = no_dense + nb
    all_weights = np.append(all_weights, np.zeros((pre_dense,curr_dense)))
  
  all_weights = np.append(all_weights, np.zeros((curr_dense, no_outputs)))
  return all_weights

def inputUpdateWeights(all_weights, feature_dict):
  #Fill with 1s
  for i in range(no_inputs):
    for key in feature_dict:
      for l in feature_dict[key]:
        all_weights[l, i, i] = 1
  
  return all_weights
  
  
def fillWeights(all_weights, max_depth, is leaves, n_nodes):

  ##Add biases 
  for l in range(max_depth):
    for c in range(n_nodes):
      if is_leaves[i]:

      else:


##Test Tree
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

estimator = DecisionTreeClassifier(max_depth=4,  random_state=0)
estimator.fit(X_train, y_train)

n_nodes = estimator.tree_.node_count
children_left = estimator.tree_.children_left
children_right = estimator.tree_.children_right
feature = estimator.tree_.feature
threshold = estimator.tree_.threshold


is_leaves, node_depth = getTreeInfo(children_left, children_right)
counts = returnBranchCount(node_depth)
feature_depth = returnFeatureDepth(n_nodes, is_leaves, feature, node_depth)