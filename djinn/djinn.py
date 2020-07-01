import tensorflow as tf 
import numpy as np 
import math
from random import gauss
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
    stack = [(0, -1)]    # seed is the root node id and its parent depth
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
    del counts[0]
    return counts

def returnLayerInfo(n_nodes, node_depth, feature):
    layer_feature = {}
    layer_parent = {}
    for i in range(n_nodes):
        layer_feature.setdefault(node_depth[i], [])
        layer_feature[node_depth[i]].append(feature[i])
    
    for key in layer_feature.keys():
        for value in layer_feature[key]:
            if value < 0:
                continue 
            else:
                parent = value
            layer_parent.setdefault(key + 1, [])
            for _ in range(2):
                layer_parent[key + 1].append(parent)
    return layer_parent, layer_feature

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


##Weight algorithm
#Normal Distribution with custom variance and zero mean
def djinn_norm_dist(n_prev, n_curr):
    mean = 0
    var = 3 / (n_prev + n_curr)
    return(gauss(mean, var))

#Create network i.e. just how many nodes where 
#Initialise the weights following the algorihmn in the paper 
#Recall shape is (n_in, n_curr) for tensorflow
#Step 2 on the algorithm
def initWeights(no_inputs, no_outputs, max_depth, no_branches):
    #Create Empty array
    all_weights = []    
    no_dense = no_inputs
    layer_info = []
    for i in no_branches.keys():
        pre_dense = no_dense
        layer_info.append(pre_dense)
        if i == max_depth:
            curr_dense = no_outputs 
        else:
            curr_dense = int(no_dense + no_branches[i]*0.5)
        all_weights.append(np.zeros((pre_dense, curr_dense)))
        no_dense = curr_dense   
    # all_weights = np.append(all_weights, np.zeros((curr_dense, no_outputs)))
    layer_info.append(no_outputs)
    return layer_info, all_weights

#Step 3 on the algorithm
def inputUpdateWeights(no_inputs, all_weights, feature_dict):
    #Fill with 1s
    for i in range(no_inputs):
        if str(i) not in feature_dict:
            continue
        for l in range(feature_dict[str(i)] - 1):
            all_weights[l][i, i] = 1
    
    return all_weights
    
def fillWeights(all_weights, no_outputs, max_depth, n_layer_nodes, 
                layer_feature_info, parent_info, nn_layer_info, regression = True):
    ##NOTE Add biases 
    parent_tracking = {}
    biases = []
    #REF the slight odd indexing, l = 0 corresponds to l=0 connections to l = 1
    #However, in written form l = 0 is just a single node which 
    #is the convention used in the dictionaries 
    #thus, l+1 are scattered about. 
    #Algorithm also says to start at level 1
    #This computationally refers to the first weight array
    #which si indexed at 0.
    for l in range(0, max_depth):
        pre_dense = nn_layer_info[l]
        curr_dense = nn_layer_info[l + 1]
        biases.append(makeBiases(pre_dense, curr_dense))
        new_neuron_index = pre_dense  #remember indexing is -1 pre_dense contains the full number#Note the slight odd indexing, l = 0 corresponds to l=0 connections to l = 1
        for c in range(n_layer_nodes[l + 1]):
            feature_index = layer_feature_info[l + 1][c]
            direct_parent = parent_info[l + 1][c]
            if l == 0:
                parent_index = direct_parent
            else:
                parent_index = parent_tracking[(l - 1, direct_parent)]['neuronID']
            #Leaf
            if feature_index < 0:
                for l_leaf in range(l, max_depth - 1): #was l+1 as in the algorithm
                    leaf_pre_dense = nn_layer_info[l]
                    leaf_curr_dense = nn_layer_info[l + 1]
                    all_weights[l_leaf][parent_index, parent_index] = djinn_norm_dist(leaf_pre_dense, leaf_curr_dense)
                if regression:
                    for i in range(no_outputs):
                        pre_dense = nn_layer_info[-2]
                        curr_dense = nn_layer_info[-1]
                        all_weights[-1][parent_index, i] = djinn_norm_dist(pre_dense, curr_dense)
                #Classification DJINN currently not working ... needs some modification i.e. info 
                #on what the leaft corresponds to in classification value.
                else:
                    all_weights[-1][i, c] = djinn_norm_dist(pre_dense, curr_dense)
            #Branch
            else:
                if new_neuron_index >= np.shape(all_weights[l])[1]:
                    continue
                all_weights[l][parent_index, new_neuron_index] = djinn_norm_dist(pre_dense, curr_dense)
                all_weights[l][feature_index, new_neuron_index] = djinn_norm_dist(pre_dense, curr_dense)
                parent_tracking[(l, feature_index)] = {'pID' : parent_index,
                                        'fID':feature_index,
                                        'neuronID': new_neuron_index}
                new_neuron_index += 1 #if another branch is present, add to additional neuron

    return all_weights, biases

def makeBiases(pre_dense, curr_dense):
    biases = np.zeros(curr_dense)
    for i in range(curr_dense):
        biases[i] = djinn_norm_dist(pre_dense, curr_dense)
    return biases

#Neural Network
def initNN(nn_layer_info, all_weights, all_biases):
    for i, neurons in enumerate(nn_layer_info):
        if i == 0:
            nn_input = keras.Input(shape = (neurons), name ="Latent_space_input")
            continue
        weights = tf.constant_initializer(all_weights[i - 1])
        bias = tf.constant_initializer(all_biases[i - 1])
        if i == 1:
            x = layers.Dense(neurons, activation= "relu", kernel_initializer= weights, 
                                             bias_initializer=bias)(nn_input)
        elif i == len(neuron_layer_info) - 1:
            nn_output = layers.Dense(neurons, kernel_initializer = weights, bias_initializer = bias)(x)
        else:
            x = layers.Dense(neurons, activation= "relu", kernel_initializer= weights, 
                                             bias_initializer=bias)(x)
    
    nn = keras.Model(nn_input, nn_output, name ="DJINN")
    nn.summary()
    return nn

##Test Tree
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

estimator = DecisionTreeClassifier(max_depth=4, random_state=0)
estimator.fit(X_train, y_train)

n_nodes = estimator.tree_.node_count
children_left = estimator.tree_.children_left
children_right = estimator.tree_.children_right
feature = estimator.tree_.feature
threshold = estimator.tree_.threshold

no_inputs = 4
no_outputs = 3
is_leaves, node_depth = getTreeInfo(children_left, children_right)
max_depth = np.max(node_depth)
branch_counts = returnBranchCount(node_depth)
feature_depth = returnFeatureDepth(n_nodes, is_leaves, feature, node_depth)
printUsefulTree(n_nodes, node_depth, is_leaves)
neuron_layer_info, weights = initWeights(no_inputs, no_outputs, max_depth, branch_counts)
layer_parents, layer_feature = returnLayerInfo(n_nodes, node_depth, feature)
weights = inputUpdateWeights(no_inputs, weights, feature_depth)
weights, biases = fillWeights(weights, no_outputs, max_depth,branch_counts, 
                    layer_feature, layer_parents, neuron_layer_info)
initNN(neuron_layer_info, weights, biases)
