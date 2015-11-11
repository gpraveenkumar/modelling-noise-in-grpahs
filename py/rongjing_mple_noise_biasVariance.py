import sys, math
from sets import Set
from collections import Counter
import random
import numpy
from multiprocessing import Pool
import gc
import os


import statsmodels.api as sm

basePath = '/homes/pgurumur/jen/noise/py/'
#school = "school074"
#school = "polblogs"
#school = "cora"
school = "facebook"
schoolLabel = "label0"


# graph
edges = {}
label = {}

#f_in = open(basePath + '../data/polblogs-nodes.txt')
f_in = open(basePath + '../data/' + school + '_' + schoolLabel +'-nodes.txt')

# no need for first line...Skipping the header
junk_ = f_in.readline()

for line in f_in:
	fields = line.strip().split()
	label[int(float(fields[0]))] = int(float(fields[1]))
	edges[int(float(fields[0]))] = set([])

f_in.close()

f_in = open(basePath + '../data/' + school + '_' + schoolLabel +'-edges.txt')

# no need for first line...Skipping the header
junk_ = f_in.readline()

for line in f_in:
	fields = line.strip().split()
	edges[ int(float(fields[0])) ].add( int(float(fields[1])) )
	edges[ int(float(fields[1])) ].add( int(float(fields[0])) )

f_in.close()



# remove those node with no neighbours
edges1 = dict(edges)
for id, neighbors in edges1.iteritems():
	if len(neighbors) == 0:
		del edges[id]
		del label[id]
del edges1



#Map AID to integers
AID_nodeId_map = {}
originalGraph = {}
originalLabels = {}
nodeIdCounter = 0
nodeAttributes = {}



# Map AID to nodeId and also transfrom labels to nodeId
for i in label:
	AID_nodeId_map[i] = nodeIdCounter
	originalLabels[nodeIdCounter] = label[i]
	originalGraph[ AID_nodeId_map[i] ] = Set([])
	nodeIdCounter += 1

# Transform graph based on AID to nodeID
for i in edges:
	neighbors = edges[i]
	for neighbor in neighbors:
		originalGraph[ AID_nodeId_map[i] ].add( AID_nodeId_map[neighbor] )


f_in = open(basePath + '../data/' + school + '.attr')

for line in f_in:
	fields = line.strip().split("::")
	index = int(float(fields[0]))
	if index in AID_nodeId_map:
		nodeAttributes[ AID_nodeId_map[ index ]] = []
		nodeAttributes[ AID_nodeId_map[ index ]].append( int(fields[1]) )
		nodeAttributes[ AID_nodeId_map[ index ]].append( int(fields[2]) )

f_in.close()




# Function to compute the sigmoid
# Input : Z
# Output : sigmoid(Z)
def sigmoid(Z):
	g = 1.0 / (1.0 + numpy.exp(-Z))
	return g



# Function to generate the class label based on a cutoff value by comparing it with a uniformly generated random number
# Input : Cutoff i.e. P(label = 1)
# Output : Class Label (0 or 1)
def generateClassLabelBasedOnCutoff(cutoff):

	# The input passed in this case is P(label = 1) but the if condition works on P(label=0)
	cutoff = 1 - cutoff

	x = random.uniform(0,1)
	if x < cutoff:
		t = 0
	else:
		t = 1
	return t



# Independent Model uses only the node attributes to predict the out of lables.
def independentModel(originalLabels,nodeAttributes,trainingLabels):
	trainLabels = []
	trainFeatures = []
	testFeatures = []
	testLabels = []

	for i in originalLabels:
		if i in trainingLabels:
			trainLabels.append( originalLabels[i] )
			l = [1] + nodeAttributes[i]
			trainFeatures.append( l )
		else:
			testLabels.append( originalLabels[i] )
			l = [1] + nodeAttributes[i]
			testFeatures.append( l )

	logit = sm.Logit(trainLabels, trainFeatures)
	 
	# fit the model
	result = logit.fit()

	#print result.summary()
	print result.params

	predicted = result.predict(testFeatures)
	resultingLabels = (predicted >= threshold).astype(int)
	accuracy,precision,recall = computeAccuracy1(testLabels,resultingLabels)
	print accuracy

	return result.params,accuracy



# Function to calculate the phi features i.e. no. of labels that match with the neighbours and no. of labels that don't match.
# Input : Node, its neighbours and set of all labels
# Output : n1 and n2 corresponing to the phi values mentioned above.
def compute_phiFeatures(node,neighbors,label):

	noOfZeroLabeledNeighbours = 0
	for neighbor in neighbors:
		if label[neighbor] == 0:
			noOfZeroLabeledNeighbours += 1

	n1 = 0
	n2 = 0

	if label[node] == 0:
		n1 = noOfZeroLabeledNeighbours
		n2 = len(neighbors) - noOfZeroLabeledNeighbours
	else:
		n1 = len(neighbors) - noOfZeroLabeledNeighbours
		n2 = noOfZeroLabeledNeighbours

	return n1,n2



# This function is used to compute the pseudo-likelyhood model and include both the node attributes and no. of label match features from the markov blanket.
def MPLE(G,label,testLabels,nodeAttributes,mleParameters,testLables_onlyTrain):

	# Construct a graph with only those nodes in the training Labels. This is requried for the training phase to learn the model parametes.
	onlyTrainingG = {}

	for id, neighbors in G.iteritems():
		if id in testLables_onlyTrain:
			continue

		newNeighbor = []
		# cycle through the neighbors
		for neighbor in neighbors:
			if neighbor in testLables_onlyTrain:
				continue
			else:
				newNeighbor.append( neighbor )
		
		# Doing this may or may not be a good idea. If you dont do this, phi1 and phi2 for 
		# the MPLE model would be zero - which is still fine as it may help learn a better set of parameter.	
		if len(newNeighbor) > 0:
			onlyTrainingG[ id ] = newNeighbor


	trainLabels = []
	trainFeatures = []

	for node,neighbors in onlyTrainingG.iteritems():

		n1,n2 = compute_phiFeatures(node,neighbors,label)

		l = []
		l.append(1)
		for i in nodeAttributes[node]:
			l.append( i )

		l.append( n1 )
		l.append( n2 )

		trainFeatures.append( l )
		trainLabels.append( label[node] )

	
	logit = sm.Logit(trainLabels, trainFeatures)
	# fit the model
	result = logit.fit()
	print result.params


	##############################
	# One Label Calculation - Just copied code from below and modified. 

	# Compute Initial Estimate i.e. compute the initial labels of all the nodes in the 
	# testLabel or unlabeled set before starting the gibbs sampling. The label of the testLabels 
	# are estimated based on just the neighbours in the training data that they are connected to.
	onlyTestG_withAllNeighours = {}

	for id, neighbors in G.iteritems():
		if id not in testLabels:
			continue
		
		# Again not sure if this is a good idea!	
		if len(neighbors) > 0:
			onlyTestG_withAllNeighours[ id ] = neighbors


	testFeatures = []
	toTestLabels = []
	nodeOrder = []

	for node,neighbors in onlyTestG_withAllNeighours.iteritems():

		n1,n2 = compute_phiFeatures(node,neighbors,label)	

		l = []
		l.append(1)
		for i in nodeAttributes[node]:
			l.append( i )

		l.append( n1 )
		l.append( n2 )

		testFeatures.append( l )
		toTestLabels.append( label[node] )
		nodeOrder.append(node)
		
	predictedProbabilities = result.predict(testFeatures)
	resultingLabels = (predictedProbabilities >= threshold).astype(int)
	initialAccuracy,precision,recall = computeAccuracy1(toTestLabels,resultingLabels)
	print initialAccuracy
	# Compute Initial Estimate -- Completed

	"""
	print toTestLabels
	print resultingLabels
	print testLabels
	print nodeOrder
	"""

	currentLabelEstimates_oneLabel = {}
	for i in range(len(nodeOrder)):
		currentLabelEstimates_oneLabel[ nodeOrder[i] ] = predictedProbabilities[i]
	##############################



	#Testing

	# Compute Initial Estimate i.e. compute the initial labels of all the nodes in the 
	# testLabel or unlabeled set before starting the gibbs sampling. The label of the testLabels 
	# are estimated based on just the neighbours in the training data that they are connected to.
	onlyTestG_withTrainingNeighours = {}

	for id, neighbors in G.iteritems():
		if id not in testLabels:
			continue

		newNeighbor = []
		# cycle through the neighbors - only get the neighbours in the trainingLabels
		for neighbor in neighbors:
			if neighbor in testLabels:
				continue
			else:
				newNeighbor.append( neighbor )
		
		# Again not sure if this is a good idea!	
		if len(newNeighbor) > 0:
			onlyTestG_withTrainingNeighours[ id ] = newNeighbor


	testFeatures = []
	toTestLabels = []
	nodeOrder = []

	for node,neighbors in onlyTestG_withTrainingNeighours.iteritems():

		n1,n2 = compute_phiFeatures(node,neighbors,label)	

		l = []
		l.append(1)
		for i in nodeAttributes[node]:
			l.append( i )

		l.append( n1 )
		l.append( n2 )

		testFeatures.append( l )
		toTestLabels.append( label[node] )
		nodeOrder.append(node)
		
	predictedProbabilities = result.predict(testFeatures)
	resultingLabels = (predictedProbabilities >= threshold).astype(int)
	initialAccuracy,precision,recall = computeAccuracy1(toTestLabels,resultingLabels)
	print initialAccuracy
	# Compute Initial Estimate -- Completed

	"""
	print toTestLabels
	print resultingLabels
	print testLabels
	print nodeOrder
	"""

	currentLabelEstimates = {}
	for i in range(len(nodeOrder)):
		currentLabelEstimates[ nodeOrder[i] ] = predictedProbabilities[i]

	
	#Gibbs Sampling part
	mpleParameters = result.params
	#meanAccuracy, precision, recall, squaredLoss = gibbsSampling_MPLE(G,label,testLabels,nodeAttributes,currentLabelEstimates,mpleParameters)
	arg_t = [G,label,testLabels,nodeAttributes,currentLabelEstimates,mpleParameters]
	
	arguments = []
	for i in range(noOfTimeToRunGibbsSampling):
		arguments.append(list(arg_t))

	pool = Pool(processes=noofProcesses)
	y = pool.map(func_star1, arguments)
	pool.close()
	pool.join()

	resultingLabelEstimates_differentRuns,squaredLoss_differentRuns = zip(*y)
	

	#Freeup space
	del arguments[:]
	gc.collect()

	return (currentLabelEstimates_oneLabel,resultingLabelEstimates_differentRuns,squaredLoss_differentRuns)



# Function to calculate the labels with doing gibbs sampling for MPLE model

def compute_p_mple(node,neighbors,label,nodeAttributes,mpleParameters):
	
	n3,n4 = compute_phiFeatures(node,neighbors,label)

	# Logistic Regression by default calculates the probability of y=1

	p_mple = mpleParameters[0] + mpleParameters[1]*nodeAttributes[0] + mpleParameters[2]*nodeAttributes[1] + mpleParameters[3]*n3 + mpleParameters[4]*n4
	p_mple = sigmoid(p_mple)

	return p_mple



# Gibbs Sampling for MPLE
def gibbsSampling_MPLE(G,label,testLabels,nodeAttributes,currentLabelEstimates,mpleParameters):	

	# Note the currentLabelEstimates are probability values instead of 0 or 1 labels. The reason they are assigned 0 or 1 inside
	# this function is because, I wanted to the randomness in the label the get assigned because of using random to be a part of
	# the different runs. This is done so as ensure different starting points essentailly to have better convergence.
	for node in currentLabelEstimates:
		currentLabelEstimates[node] = generateClassLabelBasedOnCutoff( currentLabelEstimates[node] )

	## Step 2 of algo

	nodeTraversalOrder = testLabels
	random.shuffle(nodeTraversalOrder)

	burnin = 100
	#iteration = 500

	# Although the resulting labels has the training Labels also set to zero, they are not used anywhere, so it doesnt matter what value they have.
	resultingLabels = {}
	for i in label:
		resultingLabels[i] = 0

	LabelDifferenceBetweenIterationsCounter = 0
	previousLabelDifferenceBetweenIterations = 0

	"""
	LabelDifferenceBetweenIterations = 0
	for j in currentLabelEstimates:
		if currentLabelEstimates[j] != label[j]:
				LabelDifferenceBetweenIterations += 1
	print "L Diff:","-1",LabelDifferenceBetweenIterations
	"""

	## Step 3 of algo
	#print "\nStart of Gibbs....\n"
	
	for i in range(iteration):
		
		LabelDifferenceBetweenIterations = 0

		for node in nodeTraversalOrder:
			#print "\nNode ",node
			neighbors = G[node]
			
			p_mple = compute_p_mple(node,neighbors,label,nodeAttributes[node],mpleParameters)

			"""
			t = 0
			if p_mple >= 0.5:
				t = 1
			currentLabelEstimates[node] = t
			"""

			currentLabelEstimates[node] = generateClassLabelBasedOnCutoff( p_mple )
			#print p_mple,currentLabelEstimates[node]
		
		if i > burnin:
			for j in currentLabelEstimates:
				if currentLabelEstimates[j] == 1:
					resultingLabels[j] += 1

				temp = (resultingLabels[j] + 0.0)/(i - burnin) 
				temp = int(temp >= 0.5)
				if temp != label[j]:
					LabelDifferenceBetweenIterations += 1
		#print "L Diff:",i,LabelDifferenceBetweenIterations

		
		if i >= burnin:
			# Check if the numbers of labels estimated differ from the previous interation
			if LabelDifferenceBetweenIterations == previousLabelDifferenceBetweenIterations:
				LabelDifferenceBetweenIterationsCounter += 1
			else:
				LabelDifferenceBetweenIterationsCounter = 0
				previousLabelDifferenceBetweenIterations = LabelDifferenceBetweenIterations
			
			#If the estimates don't change for 100 interations, we can exit considering it has converged
			#if LabelDifferenceBetweenIterationsCounter >= 100:
			#	print "Interations ended at " + str(i) + " as estimates have not changed!"

				# In the absence on this line
				#iteration = i
				#break
		
	
	# Will be used for computing Squared loss. Is a dictionary because of the 
	# second line in the for loop and for it to be consistent to resultingLabels
	resultingLabelsForSquaredLoss = {}

	for i in resultingLabels:
		resultingLabels[i] = (resultingLabels[i] + 0.0)/(iteration - burnin) 
		resultingLabelsForSquaredLoss[i] = resultingLabels[i]

	# Compute Squared Loss with the averages of Gibbs sampling before assigning them a single value
	squaredLoss = computeSquaredLoss(label,testLabels,resultingLabelsForSquaredLoss)
	
	print "No. of interation in which labels have not changed:",LabelDifferenceBetweenIterationsCounter

	return (resultingLabelsForSquaredLoss,squaredLoss)



# Function to compute the Accuracy, Precision, Recall
# Input : Original Labels, test labels and the predicted labels
# Output : Accuracy, Precision, Recall
def computeAccuracy(label,testLabels,resultingLabels):
	counts = numpy.zeros([2,2])
	for i in testLabels:
		counts[ label[i], resultingLabels[i] ] += 1

	#print counts
	accuracy = (counts[0,0]+counts[1,1] + 0.0)/sum(sum(counts))

	precision = 0.0
	recall = 0.0
	if (counts[0,1]+counts[1,1]) != 0:
		precision = counts[1,1] /(counts[0,1]+counts[1,1])
	if (counts[1,0]+counts[1,1]) != 0:
		recall = counts[1,1]  / (counts[1,0]+counts[1,1])
	return accuracy,precision,recall



# Function to compute the Accuracy, Precision, Recall. Difference between this and 
# the previous function is that in this function the testLabels are acutal prediction 
# values whereas in the previous function the testLabels are the nodeIds.
# Input : test labels and the predicted labels
# Output : Accuracy, Precision, Recall
def computeAccuracy1(testLabels,resultingLabels):
	counts = numpy.zeros([2,2])
	for i in range(len(testLabels)):
		counts[ testLabels[i], resultingLabels[i] ] += 1

	#print counts
	accuracy = (counts[0,0]+counts[1,1] + 0.0)/sum(sum(counts))

	precision = 0.0
	recall = 0.0
	if (counts[0,1]+counts[1,1]) != 0:
		precision = counts[1,1] /(counts[0,1]+counts[1,1])
	if (counts[1,0]+counts[1,1]) != 0:
		recall = counts[1,1]  / (counts[1,0]+counts[1,1])
	return accuracy,precision,recall



# Function to compute the squared loss
# Input : Original Labels, test labels and the predicted labels
# Output : squared loss, precision = 0,recall = 0 .... just to match the setting of computeAccuracy
def computeSquaredLoss(label,testLabels,resultingLabels):
	squaredLoss = 0
	for i in testLabels:
		squaredLoss += math.pow((label[i] - resultingLabels[i]),2)

	squaredLoss /= len(testLabels)

	return squaredLoss



# ListOfObject can be a list of numbers or a list of vectors or a list of matrices
def computeMeanAndStandardError(listOfObjects):
	mean = numpy.mean(listOfObjects,0)
	sd = numpy.std(listOfObjects,0)
	se = sd / math.sqrt(len(listOfObjects))
	median = numpy.median(listOfObjects,0)
	return (mean,sd,se,median)


def func_star_FlipLabels(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return makeGraph_FlipLabels(*a_b)

def makeGraph_FlipLabels(onlyTrainLabels_keys,onlyTrainLabels,onlyTrainGraph,nodeIdCounter,edgeList,onlyTrainNodeAttributes):
	# Map to store the relationships between nodes/labels in the training set and the new 
	# node formed due to flips. The is needed in order to replace old edges with new edges.
	map_onlyTrainLabelId_newLabelId = {}
	newGraph = {}
	newLabels = {}
	newNodeAttributes = {}

	# Randomly sample a percentage of original label and flip it
	#onlyTrainLabels_keys = onlyTrainLabels.keys()
	noOfLabelsToFlip = int(percentageOfGraph*len(onlyTrainLabels_keys))
	labelsToFlip = random.sample(onlyTrainLabels_keys,noOfLabelsToFlip)

	# Add new nodes to the graph from the old training Data with label flips
	for i in onlyTrainLabels_keys:
		t = onlyTrainLabels[i]
		
		# Flip the labels. The XORing with 1 reverses the labels
		# 0^1 = 1
		# 1^1 = 0
		if i in labelsToFlip:
			t = t^1
		
		# Add the modified labels to newLabels
		newLabels[nodeIdCounter] = t
		map_onlyTrainLabelId_newLabelId[ i ] = nodeIdCounter

		nodeIdCounter += 1


	# Modify the edge connection based on the new nodes id and add them to the newGraph
	for node,neighbors in onlyTrainGraph.iteritems():

		mappedId = map_onlyTrainLabelId_newLabelId[ node ]

		newNeighbors = set()
		for neighbor in neighbors:
			newNeighbors.add( map_onlyTrainLabelId_newLabelId[ neighbor ] )

		newGraph[ mappedId ] = newNeighbors
		newNodeAttributes[ mappedId ] = nodeAttributes[ node ]

	return (newGraph,newLabels,newNodeAttributes)		



def func_star_DropLabels(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return makeGraph_DropLabels(*a_b)

def makeGraph_DropLabels(onlyTrainLabels_keys,onlyTrainLabels,onlyTrainGraph,nodeIdCounter,edgeList):
	# Map to store the relationships between nodes/labels in the training set and the new 
	# node formed due to drops. The is needed in order to replace old edges with new edges.
	map_onlyTrainLabelId_newLabelId = {}
	newGraph = {}
	newLabels = {}

	# Randomly sample a percentage of original label and flip it
	noOfLabelsToDrop = int(percentageOfGraph*len(onlyTrainLabels_keys))
	labelsToDrop = random.sample(onlyTrainLabels_keys,noOfLabelsToDrop)

	# Add new nodes to the graph from the old training Data after dropping nodes
	for i in onlyTrainLabels_keys:
		
		# If i is one of the labels to be dropped, just continue
		if i in labelsToDrop:
			continue

		newLabels[nodeIdCounter] = onlyTrainLabels[i]
		map_onlyTrainLabelId_newLabelId[ i ] = nodeIdCounter

		nodeIdCounter += 1


	# Modify the edge connection based on the new nodes id and add them to the newGraph
	for node,neighbors in onlyTrainGraph.iteritems():

		# If node is one of the labels to be dropped, just continue
		if node in labelsToDrop:
			continue

		mappedId = map_onlyTrainLabelId_newLabelId[ node ]

		newNeighbors = set()
		for neighbor in neighbors:
			# If neighbor is one of the labels to be dropped, just continue
			if neighbor in labelsToDrop:
				continue
			# If not add the neighbours	
			newNeighbors.add( map_onlyTrainLabelId_newLabelId[ neighbor ] )

		newGraph[ mappedId ] = newNeighbors

	return (newGraph,newLabels)		



def func_star_DropEdges(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return makeGraph_DropEdges(*a_b)

def makeGraph_DropEdges(onlyTrainLabels_keys,onlyTrainLabels,onlyTrainGraph,nodeIdCounter,edgeList):
	# Map to store the relationships between nodes/labels in the training set and the new 
	# node formed. The is needed in order to replace old edges with new edges.
	map_onlyTrainLabelId_newLabelId = {}
	newGraph = {}
	newLabels = {}

	# Randomly sample a percentage of original label and flip it
	noOfEdgesToDrop = int(percentageOfGraph*len(edgeList))
	edgesToDrop = random.sample(edgeList,noOfEdgesToDrop)

	# Add new nodes to the graph from the old training Data after dropping nodes
	for i in onlyTrainLabels_keys:
		
		newLabels[nodeIdCounter] = onlyTrainLabels[i]
		map_onlyTrainLabelId_newLabelId[ i ] = nodeIdCounter
		nodeIdCounter += 1


	# Modify the edge connection based on the new nodes id and add them to the newGraph
	for node,neighbors in onlyTrainGraph.iteritems():

		mappedId = map_onlyTrainLabelId_newLabelId[ node ]

		newNeighbors = set()
		for neighbor in neighbors:
			# If [node,neighbor] is one of the edges to be dropped, just continue
			if (node,neighbor) in edgesToDrop or (neighbor,node) in edgesToDrop:
				continue
			# If not add the neighbours	
			newNeighbors.add( map_onlyTrainLabelId_newLabelId[ neighbor ] )

		newGraph[ mappedId ] = newNeighbors

	return (newGraph,newLabels)	



def func_star_RewireEdges(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return makeGraph_RewireEdges(*a_b)

def makeGraph_RewireEdges(onlyTrainLabels_keys,onlyTrainLabels,onlyTrainGraph,nodeIdCounter,edgeList):
	# Similar to drop Edges, first selection a fraction of edges and drop them.
	# Next, randomly select one end point of the end. Select an node from the graph excluding this point.
	# Rewire that edge starting from that one end point to the selected node.

	# Map to store the relationships between nodes/labels in the training set and the new 
	# node formed. The is needed in order to replace old edges with new edges.
	map_onlyTrainLabelId_newLabelId = {}
	newGraph = {}
	newLabels = {}

	# Randomly sample a percentage of original label and flip it
	noOfEdgesToDrop = int(percentageOfGraph*len(edgeList))
	edgesToDrop = random.sample(edgeList,noOfEdgesToDrop)

	# Add new nodes to the graph from the old training Data after dropping nodes
	for i in onlyTrainLabels_keys:
		
		newLabels[nodeIdCounter] = onlyTrainLabels[i]
		map_onlyTrainLabelId_newLabelId[ i ] = nodeIdCounter
		nodeIdCounter += 1

	# Modify the edge connection based on the new nodes id and add them to the newGraph
	for node,neighbors in onlyTrainGraph.iteritems():

		mappedId = map_onlyTrainLabelId_newLabelId[ node ]

		newNeighbors = set()
		for neighbor in neighbors:
			# If [node,neighbor] is one of the edges to be dropped, just continue
			if (node,neighbor) in edgesToDrop or (neighbor,node) in edgesToDrop:
				continue
			# If not add the neighbours	
			newNeighbors.add( map_onlyTrainLabelId_newLabelId[ neighbor ] )

		newGraph[ mappedId ] = newNeighbors

	
	# Do reviwing of edgesToDrop.
	# For each tuple in edgeToDrop select and fix either the start or the end point - call it start. 
	# Next randomly sample a node from the rest of the graph leaving this selected node out - call it end.
	# This start node and end node together form the new rewired edge.

	rewireEdgeList = Set()

	for edge in edgesToDrop:
		# Pick an index to keep it fixed. Let us call it the start vertex of the edge
		index = random.sample([0,1],1)
		start = edge[ index[0] ] # index is a list of 1 element.

		# Compute leaveIndexOut_onlyTrainLabels_keys by excluding the original start and end points from 
		# onlyTrainLabels_keys and find the new end point by randomly sampling an node
		leaveIndexOut_onlyTrainLabels_keys = [i for i in onlyTrainLabels_keys if i != edge]
		end = random.sample(leaveIndexOut_onlyTrainLabels_keys,1)
		end = end[0] # end is a list of one element. Hence, list not needed.

		# Make a tuple of the new edge and add it to rewireEdgeList
		rewireEdgeList.add( (start,end) )


	# Add the rewired edges back to the graph
	for edge in rewireEdgeList:
		start = map_onlyTrainLabelId_newLabelId[ edge[0] ]
		end = map_onlyTrainLabelId_newLabelId[ edge[1] ]

		newGraph[ start ].add( end )
		newGraph[ end ].add( start )


	return (newGraph,newLabels)	





def func_star_FlipLabelDropEdges(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return makeGraph_func_star_FlipLabelDropEdges(*a_b)

def makeGraph_func_star_FlipLabelDropEdges(onlyTrainLabels_keys,onlyTrainLabels,onlyTrainGraph,nodeIdCounter,edgeList):
	# Map to store the relationships between nodes/labels in the training set and the new 
	# node formed. The is needed in order to replace old edges with new edges.
	map_onlyTrainLabelId_newLabelId = {}
	newGraph = {}
	newLabels = {}

	# Randomly sample a percentage of original label and flip it
	noOfEdgesToDrop = int(percentageOfGraph2*len(edgeList))
	edgesToDrop = random.sample(edgeList,noOfEdgesToDrop)

	# Add new nodes to the graph from the old training Data after dropping nodes
	for i in onlyTrainLabels_keys:
		
		newLabels[nodeIdCounter] = onlyTrainLabels[i]
		map_onlyTrainLabelId_newLabelId[ i ] = nodeIdCounter
		nodeIdCounter += 1


	# Modify the edge connection based on the new nodes id and add them to the newGraph
	for node,neighbors in onlyTrainGraph.iteritems():

		mappedId = map_onlyTrainLabelId_newLabelId[ node ]

		newNeighbors = set()
		for neighbor in neighbors:
			# If [node,neighbor] is one of the edges to be dropped, just continue
			if (node,neighbor) in edgesToDrop or (neighbor,node) in edgesToDrop:
				continue
			# If not add the neighbours	
			newNeighbors.add( map_onlyTrainLabelId_newLabelId[ neighbor ] )

		newGraph[ mappedId ] = newNeighbors


	# Randomly sample a percentage of new labels and flip it
	noOfLabelsToFlip = int(percentageOfGraph*len( newGraph.keys() ))
	labelsToFlip = random.sample(  newGraph.keys() ,noOfLabelsToFlip)

	# Flip the label
	for i in labelsToFlip:
		# Flip the labels. The XORing with 1 reverses the labels
		# 0^1 = 1
		# 1^1 = 0
		newLabels[i] = newLabels[i]^1		


	return (newGraph,newLabels)	



# Function to Flip the data
def makeNoisyGraphs(action,percentageOfGraph,noOfTimesToRepeat,originalGraph,originalLabels,testLabels,percentageOfGraph2,nodeAttributes):

	print "In makeNoisyGraphs() ...."
	# Copy the Graph
	newGraph = dict(originalGraph)
	newLabels = dict(originalLabels)
	newNodeAttributes = dict(nodeAttributes)
	nodeIdCounter = len(newLabels)

	# Since we have to replicate only the training graph, remove all the test node 
	#	and edges before making flips. Otherwise, if they remain connected to the
	#	original testLabels, the degree of the testLabels will artifically inflate.
	#	In other words, we would have a new flipped graph connected to all the 
	#	original testLabels.

	onlyTrainGraph = dict(newGraph)
	onlyTrainLabels = dict(newLabels)
	onlyTrainNodeAttributes = dict(newNodeAttributes)

	for node, neighbors in newGraph.iteritems():
		# If the node is in test labels remove it.
		if node in testLabels:
			del onlyTrainGraph[node]
			del onlyTrainLabels[node]
			del onlyTrainNodeAttributes[node]
		# If it is not, check if it is connected to a test label and remove that edge
		else:
			newNeighbors = set(neighbors)
			for neighbor in neighbors:
				if neighbor in testLabels:
					newNeighbors.remove( neighbor )
			onlyTrainGraph[node] = newNeighbors

	
	# Store edges as list of lists
	edgeList = Set()
	for node, neighbors in onlyTrainGraph.iteritems():
		for neighbor in neighbors:
			if (neighbor,node) not in edgeList:
				edgeList.add( ( node, neighbor) )


	arguments = []
	for i in range(noOfTimesToRepeat):
		ctr = nodeIdCounter + i*len(originalLabels)
		l = []
		l.append( onlyTrainLabels.keys() )
		l.append( onlyTrainLabels )
		l.append( onlyTrainGraph )
		l.append( ctr )
		l.append( edgeList )
		l.append( onlyTrainNodeAttributes )
		arguments.append(l)

	if action == "flipLabel":
		functionToCall = func_star_FlipLabels
	elif action == "dropLabel":
		functionToCall = func_star_DropLabels
	elif action == "dropEdges":
		functionToCall = func_star_DropEdges	
	elif action == "rewireEdges":
		functionToCall = func_star_RewireEdges
	elif action == "flipLabelDropEdges":
		functionToCall = func_star_FlipLabelDropEdges

	pool = Pool(processes=noofProcesses)
	y = pool.map(functionToCall, arguments)
	pool.close()
	pool.join()

	allGraphs,allLabels,allNodeAttributes = zip(*y)		

	for dic in allGraphs:
		newGraph.update(dic)

	for dic in allLabels:
		newLabels.update(dic)

	for dic in allNodeAttributes:
		newNodeAttributes.update(dic)
	
	# Freeup Space
	del arguments[:]
	gc.collect()

	return (newGraph,newLabels,newNodeAttributes)



def func_star1(a_b):
	"""Convert `f([1,2])` to `f(1,2)` call."""
	return gibbsSampling_MPLE(*a_b)




def computeDifference_bias_variance(predictedLabels,originalLabels):
	differences = []
	for label in predictedLabels:
		differences.append( abs(originalLabels[label] - predictedLabels[label]) )

	mean = numpy.mean(differences,0)
	sd = numpy.std(differences,0)
	var = math.pow(sd,2)
	bias = math.pow(mean,2)
	return (bias,var)


Action = "flipLabel"
performInfernceOnly = False

noofProcesses = 7
noOfTimeToRunGibbsSampling = 5
iteration = 500
threshold = 0.5

if performInfernceOnly:
	arg1,arg2,arg3,arg4,arg5,arg6,arg7 = sys.argv[1].split(' ')

	trainingSizeList = [ float(arg1) ]
	percentageOfGraphList = [ float(arg2) ]
	noOfTimesToRepeatList = [ int(arg3) ]
	percentageOfGraph2 = float(arg4) 

	parm_priorClass0 = float(arg5)
	parm_0given0 = float(arg6)
	parm_1given1 = float(arg7)
	parameters = (parm_priorClass0,parm_0given0,parm_1given1)
else:
	arg1,arg2,arg3,arg4 = sys.argv[1].split(' ')

	trainingSizeList = [ float(arg1) ]
	percentageOfGraphList = [ float(arg2) ]
	noOfTimesToRepeatList = [ int(arg3) ]
	percentageOfGraph2 =  float(arg4) 



"""
#Read the testLabels from the files to make it constant across runs
testLabelsList = []
testSize = 1-trainingSizeList[0]
f = open(basePath + "RandomTestLabelsForIterations/" + str(testSize) + "_testLabels.txt")
tLL = f.readlines()
f.close


for line in tLL:
	t = line.strip().split(',')
	testLabelsList.append([int(i) for i in t])
"""

for trainingSize in trainingSizeList:

	print "\n\n\n\n\ntrainingSize:",trainingSize
				
	testSize = 1-trainingSize
	noOfLabelsToMask = int(testSize*len(originalLabels))
	print "testLabels Size:",noOfLabelsToMask


for trainingSize in trainingSizeList:
	for percentageOfGraph in percentageOfGraphList:
		outputTofile = []
		for noOfTimesToRepeat in noOfTimesToRepeatList:


				s1 = []
				lb = []
				lv = []
				tb = []
				tv = []
				ib = []
				iv = []

				for i in range(2):
					print "\nRepetition No.:",i+1

					# Uncomment the first line to generate random testLables for each iteration
					# Uncomment the second line to read the generated random testLables for each iteration. Based on Jen's suggestion to keep the testLabels constant across iterations.

					testLabels = random.sample(originalLabels,noOfLabelsToMask)
					#testLabels = testLabelsList[i]

					# When there is no need to repeat just work with the original graph
					if noOfTimesToRepeat == 0:
						currentGraph,currentLabels,currentNodeAttributes = originalGraph,originalLabels,nodeAttributes
					else:
						currentGraph,currentLabels,currentNodeAttributes = makeNoisyGraphs(Action,percentageOfGraph,noOfTimesToRepeat,originalGraph,originalLabels,testLabels,percentageOfGraph2,nodeAttributes)
					
					print "Size of graph:",len(currentLabels)


					learningPredictions = []
					totalPredictions = []
					for j in range(5):
						testLabels_for_oneLabel = random.sample(originalLabels,noOfLabelsToMask)
						trainingLabels = [i for i in currentLabels if i not in testLabels_for_oneLabel]
						#print "Start trainLabels:",len(trainingLabels)
						mleParameters,indepModel_accuracy = independentModel(currentLabels,currentNodeAttributes,trainingLabels)
						lp, tp, sq = MPLE(currentGraph,currentLabels,testLabels,currentNodeAttributes,mleParameters,testLabels_for_oneLabel)

						learningPredictions.append(lp)
						for tt in tp:
							totalPredictions.append(tt)

						for tt in sq:
							s1.append(tt)


					print "length learningPredictions:",len(learningPredictions)
					print "length totalPredictions:",len(totalPredictions)

					ylm = {}
					ytm = {}	

					for label in testLabels:
						ylm[label] = 0				
						ytm[label] = 0

					for dic in learningPredictions:
						for label in ylm:
							ylm[label] += dic[label]

					for label in ylm:
						ylm[label] /= len(learningPredictions)

					for dic in totalPredictions:
						for label in ytm:
							ytm[label] += dic[label]

					for label in ytm:
						ytm[label] /= len(totalPredictions)


					learningBias, learningVariance = computeDifference_bias_variance(ylm,originalLabels)
					totalBias, totalVariance = computeDifference_bias_variance(ytm,originalLabels)

					lb.append(learningBias)
					lv.append(learningVariance)
					tb.append(totalBias)
					tv.append(totalVariance)
					ib.append(totalBias-learningBias)
					iv.append(totalVariance-learningVariance)


					#Freeup space
					#del arguments[:]
					gc.collect()
				#print se

				print "#####"
				print "length s1:",len(s1)				
				avgSquaredLoss,useless1,useless2,useless3 = computeMeanAndStandardError(s1)
				avgLearningBias,useless1,useless2,useless3 = computeMeanAndStandardError(lb)
				avgLearningVariance,useless1,useless2,useless3 = computeMeanAndStandardError(lv)
				avgTotalBias,useless1,useless2,useless3 = computeMeanAndStandardError(tb)
				avgTotalVariance,useless1,useless2,useless3 = computeMeanAndStandardError(tv)
				avgInferenceBias,useless1,useless2,useless3 = computeMeanAndStandardError(ib)
				avgInferenceVariance,useless1,useless2,useless3 = computeMeanAndStandardError(iv)

				# Calculating medianPrecision and medianRecall might not make sense... because precicion and recall are dependent .... and median can select different values for them.
				#f1_median = (2*medianPrecision*medianRecall)/(medianPrecision+medianRecall)

				prefix = str(int(percentageOfGraph*100)) + "perc_" + str(noOfTimesToRepeat) + "repeat"
				if percentageOfGraph2 != 0:
					prefix = str(int(percentageOfGraph*100)) + "FL_" + str(int(percentageOfGraph2*100)) + "DE_" + str(noOfTimesToRepeat) + "repeat"
				
				print "\nFINAL .................. "
				print "avgSquaredLoss :",avgSquaredLoss
				print "avgLearningBias :",avgLearningBias
				print "avgLearningVariance :",avgLearningVariance
				print "avgTotalBias :",avgTotalBias
				print "avgTotalVariance :",avgTotalVariance
				print "avgInferenceBias :",avgInferenceBias
				print "avgInferenceVariance :",avgInferenceVariance

				#outputTofile.append( [ Action + "ResultsBaselines.txt",prefix , str(trainingSize) , str(round(meanAccuracy,4)) , str(round(sd,4)) , str(round(se,4)) , str(round(meanPrecision,4)) , str(round(meanRecall,4)) , str(round(f1,4)), str(round(medianAccuracy,4))])
				outputTofile.append( [ Action + "Results.txt",prefix , str(trainingSize) , str(round(avgSquaredLoss,4)) , str(round(avgLearningBias,4)) , str(round(avgLearningVariance,4)) , str(round(avgTotalBias,4)) , str(round(avgTotalVariance,4)) , str(round(avgInferenceBias,4)) , str(round(avgInferenceVariance,4)) ])
				#outputTofile.append( [ Action + "Results.txt","Median_"+prefix , str(trainingSize) , str(round(medianAccuracy,4)) , str(round(0,4)) , str(round(0,4)) , str(round(0,4)) , str(round(0,4)) , str(round(0,4)) ])
				#print e1


		fileName = "flipLabel.txt"
		path = basePath + '../results/Rongjing-mple_noise-biasVariance' + school + '-' + schoolLabel + '-' 
		f_out = open(path+fileName,'a')

		for otf in outputTofile:
			f_out.write("\t".join(otf)  + "\n")

		f_out.close()





