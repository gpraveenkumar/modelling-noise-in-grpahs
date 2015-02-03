import sys, math
from sets import Set
from collections import Counter
import random
import numpy
from multiprocessing import Pool
import gc


basePath = '/homes/pgurumur/jen/noise/py/'
school = "school074"
#school = "polblogs"
#school = "cora"
#school = "facebook"
schoolLabel = "label0"


# The .pyc file is not updated as it should be with changes in p_nob_gibbs.py. 
#Hence, I am manually removing it so that it can be recomplied and a new .pyc 
#can be generated everytime. THis is just to avoid potential problems.
import os, errno

def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e: # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occured
silentremove(basePath + 'p_noB_gibbs.pyc')

from p_noB_gibbs import *


#random.seed(1)

binary = True
directed = False

#testSize = 0.3

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

#f_in = open(basePath + '../data/polblogs-edges.txt')
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



# Compute the pairings
# pairings for computing correlations
pairs = []
for id, neighbors in edges.iteritems():
	# cycle through the neighbors
	for neighbor in neighbors:
		pairs.append([label[id], label[neighbor]])




def computeEstimatedCounts(G,label):
	estimatedProbabities = numpy.zeros([2,2])

	for id, neighbors in G.iteritems():
		# cycle through the neighbors
		for neighbor in neighbors:
			estimatedProbabities[ label[id], label[neighbor] ] += 1
			#if not directed:
			#	estimatedProbabities[ label[neighbor], label[id] ] += 1

	return estimatedProbabities



#Map AID to integers
AID_nodeId_map = {}
originalGraph = {}
originalLabels = {}
nodeIdCounter = 0

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


print "Original Attr. Cor.:", computeCorrelation(computePairs(originalGraph,originalLabels))



## noB starts here

"""
print "Original +/- label counts:",computeLabelCounts(originalLabels)

# masking a fraction(=testsize) of labels
noOfLabelsToMask = int(testSize*len(originalLabels))
testLabels = random.sample(originalLabels,noOfLabelsToMask)

print "No. of test labels:",len(testLabels)
originalTrainLabels = [i for i in originalLabels if i not in testLabels]

x = computeEstimatedCounts(originalGraph, originalLabels)
print "Label-Label Count across Edges:\n"
print x
print sum(sum(x))
"""





def func_star_FlipLabels(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return makeGraph_FlipLabels(*a_b)

def makeGraph_FlipLabels(onlyTrainLabels_keys,onlyTrainLabels,onlyTrainGraph,nodeIdCounter,edgeList):
	# Map to store the relationships between nodes/labels in the training set and the new 
	# node formed due to flips. The is needed in order to replace old edges with new edges.
	map_onlyTrainLabelId_newLabelId = {}
	newGraph = {}
	newLabels = {}

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

	return (newGraph,newLabels)		



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
def makeNoisyGraphs(action,percentageOfGraph,noOfTimesToRepeat,originalGraph,originalLabels,testLabels,percentageOfGraph2):

	print "In makeNoisyGraphs() ...."
	# Copy the Graph
	newGraph = dict(originalGraph)
	newLabels = dict(originalLabels)
	nodeIdCounter = len(newLabels)

	# Since we have to replicate only the training graph, remove all the test node 
	#	and edges before making flips. Otherwise, if they remain connected to the
	#	original testLabels, the degree of the testLabels will artifically inflate.
	#	In other words, we would have a new flipped graph connected to all the 
	#	original testLabels.

	onlyTrainGraph = dict(newGraph)
	onlyTrainLabels = dict(newLabels)

	for node, neighbors in newGraph.iteritems():
		# If the node is in test labels remove it.
		if node in testLabels:
			del onlyTrainGraph[node]
			del onlyTrainLabels[node]
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

	allGraphs,allLabels = zip(*y)		

	for dic in allGraphs:
		newGraph.update(dic)

	for dic in allLabels:
		newLabels.update(dic)
	
	# Freeup Space
	del arguments[:]
	gc.collect()

	return (newGraph,newLabels)

#gibbsSampling(originalGraph,originalLabels,testLabels)




def func_star(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return gibbsSampling(*a_b)
    #return smartBaseline(*a_b)
    


# ListOfObject can be a list of numbers or a list of vectors or a list of matrices
def computeMeanAndStandardError(listOfObjects):
	mean = numpy.mean(listOfObjects,0)
	sd = numpy.std(listOfObjects,0)
	se = sd / math.sqrt(len(listOfObjects))
	median = numpy.median(listOfObjects,0)
	return (mean,sd,se,median)



# Function that is used to set all the labels in the testLabels to a 
# particular label value (0 or 1). This is primarily used to check if the 
# performace of the classifier is better than predicting all 0's or 1's.
# Input : originalLabels, testLabels, Value (Label like 0 or 1 to which 
#	all test labels should be set)
# Output : predictedLabels - all testLabels set to that Value

def setLabelForBaselineAccuracies(originalLabels, testLabels, value):
	predictedLabels = {}

	for i in originalLabels:
		if i in testLabels:
			predictedLabels[i] = value
		else:
			predictedLabels[i] = originalLabels[i]

	return predictedLabels



# Function is used to compute to the range of Baseline Predictions. This 
#	is used to understand if using a small training set size is meaningful
#	or not i.e. If a prediction result is worse than baseline, then we are not 
#	making any progress.
# Input : baseline vector of two elements, representing the min and max value 
# 	of baseline. curVal - Current value of Baseline.
# Optput Updated baseline vector

def updateBaselineRanges(baseline,curValue):
	if curValue < baseline[0]:
		baseline[0] = curValue
	if curValue > baseline[1]:
		baseline[1] = curValue
	return baseline



# Function to write the results to a file
# Input : A list consisting of fileName,Label,trainingSize,Accuracy_Mean,Accuracy_SD,Accuracy_SE,Precision_Mean(optional),Recall_Mean(optional)
# Output : None
def writeToFile(l):
	fileName = l[0]
	# Remove the fileName from the list, so as to facilitate join
	l.pop(0)
	path = basePath + '../results/' + school + '-' + schoolLabel + '_moreRepeats_'
	f_out = open(path+fileName,'a')
	f_out.write("\t".join(l)  + "\n")
	f_out.close()



#trainingSize = 0.7
#percentageOfGraph = 0.05   # express in fraction instead of percentage...incorrect naming, will update soon
#noOfTimesToRepeat = 10

Action = "flipLabel"
noofProcesses = 7
performInfernceOnly = True

#writeToFile( [ Action + "ResultsBaselines.txt", "Label" , "trainingSize" , "Accuracy_Mean","Accuracy_SD","Accuracy_SE","Precision_Mean","Recall_Mean","F1"] )
#writeToFile( [ Action + "Results.txt", "Label" , "trainingSize" , "Accuracy_Mean","Accuracy_SD","Accuracy_SE","Precision_Mean","Recall_Mean","F1"])

"""

for trainingSize in [0.05,0.1,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.7,0.8,0.9]:
	for percentageOfGraph in [0]:
		for noOfTimesToRepeat in [0]:
for trainingSize in [0.05,0.1,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.7,0.8,0.9]:
	for percentageOfGraph in [0.05,0.10,0.15,0.20,0.25,0.30]:
		for noOfTimesToRepeat in [2,5,10]:
				
"""

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


#Read the testLabels from the files to make it constant across runs
testLabelsList = []
testSize = 1-trainingSizeList[0]
f = open(basePath + "RandomTestLabelsForIterations/" + str(testSize) + "_testLabels.txt")
tLL = f.readlines()
f.close

for line in tLL:
	t = line.strip().split(',')
	testLabelsList.append([int(i) for i in t])



for trainingSize in trainingSizeList:
	for percentageOfGraph in percentageOfGraphList:
		outputTofile = []
		for noOfTimesToRepeat in noOfTimesToRepeatList:

				print "\n\n\n\n\ntrainingSize:",trainingSize," percentageOfGraph: ",percentageOfGraph," noOfTimesToRepeat: ",noOfTimesToRepeat," percentageOfGraph2: ",percentageOfGraph2
				
				testSize = 1-trainingSize
				noOfLabelsToMask = int(testSize*len(originalLabels))
				print "testLabels Size:",noOfLabelsToMask

				a1 = []
				p1 = []
				r1 = []
				e1 = []
				c1 = []
				s1 = []

				Baseline_0 =[1,0]
				Baseline_1 =[1,0]

				for i in range(200):
					print "\nRepetition No.:",i+1

					# Uncomment the first line to generate random testLables for each iteration
					# Uncomment the second line to read the generated random testLables for each iteration. Based on Jen's suggestion to keep the testLabels constant across iterations.

					#testLabels = random.sample(originalLabels,noOfLabelsToMask)
					testLabels = testLabelsList[i]

					# When there is no need to repeat just work with the original graph
					if noOfTimesToRepeat == 0:
						currentGraph,currentLabels = originalGraph,originalLabels
					else:
						currentGraph,currentLabels = makeNoisyGraphs(Action,percentageOfGraph,noOfTimesToRepeat,originalGraph,originalLabels,testLabels,percentageOfGraph2)
					
					print "Size of graph:",len(currentLabels)

					if performInfernceOnly:
						arg_t = [currentGraph,currentLabels,testLabels,parameters]
					else:
						arg_t = [currentGraph,currentLabels,testLabels,None]	

					arguments = []
					for i in range(25):
						arguments.append(list(arg_t))

					pool = Pool(processes=noofProcesses)
					y = pool.map(func_star, arguments)
					pool.close()
					pool.join()

					accuracy, precision, recall, classPrior, estimatedProbabities, squaredLoss = zip(*y)
					meanAccuracy,sd,se,uselessMedian = computeMeanAndStandardError(accuracy)
					meanPrecision,uselessSd,uselessSe,uselessMedian = computeMeanAndStandardError(precision)
					meanRecall,uselessSd,uselessSe,uselessMedian = computeMeanAndStandardError(recall)
					meanSquaredLoss,sd,se,uselessMedian = computeMeanAndStandardError(squaredLoss)

					meanClassPrior,sdClassPrior,seClassPrior,uselessMedian = computeMeanAndStandardError(classPrior)
					meanEstimatedProbabilities,seEstimatedProbabilities,seEstimatedProbabilities,uselessMedian = computeMeanAndStandardError(estimatedProbabities)
					

					predictedLabels = setLabelForBaselineAccuracies(currentLabels, testLabels, 0)
					curBaselineValue,uselessPrecision,uselessRecall = computeAccuracy(currentLabels,testLabels, predictedLabels )
					#curBaselineValue = computeSquaredLoss(currentLabels,testLabels, predictedLabels )
					Baseline_0 = updateBaselineRanges(Baseline_0,curBaselineValue)
					print "Baseline_0:", curBaselineValue
					predictedLabels = setLabelForBaselineAccuracies(currentLabels, testLabels, 1)
					curBaselineValue,uselessPrecision,uselessRecall = computeAccuracy(currentLabels,testLabels, predictedLabels )
					#curBaselineValue = computeSquaredLoss(currentLabels,testLabels, predictedLabels )
					Baseline_1 = updateBaselineRanges(Baseline_1,curBaselineValue)
					print "Baseline_1:", curBaselineValue
					print "MeanAccuracy:",meanAccuracy
					print "SD:",sd
					print "SE:",se
					print "MeanPrecision:",meanPrecision
					print "MeanRecall:",meanRecall
					print "MeanSquaredLoss:",meanSquaredLoss
					
					# This sd and variance are less than 10<-16
					#print "MeanClassPrior:",meanClassPrior
					#print "ClassPriorSD:",classPriorSd
					#print "ClassPriorSE:",classPriorSe
					#print "MeanEstimatedProbabilities:\n",meanEstimatedProbabilities
					#print "EstimatedProbabilitiesSD:\n",sdEstimatedProbabilities
					#print "EstimatedProbabilitiesSE:\n",seEstimatedProbabilities
					a1.append(meanAccuracy)
					p1.append(meanPrecision)
					r1.append(meanRecall)
					c1.append(meanClassPrior)
					e1.append(meanEstimatedProbabilities)
					s1.append(meanSquaredLoss)

					#Freeup space
					del arguments[:]
					gc.collect()
				#print se

				meanAccuracy,sd,se,medianAccuracy = computeMeanAndStandardError(a1)
				meanPrecision,useless1,useless2,medianPrecision = computeMeanAndStandardError(p1)
				meanRecall,useless1,useless2,medianRecall = computeMeanAndStandardError(r1)
				meanClassPrior,sdClassPrior,seClassPrior,uselessMedian = computeMeanAndStandardError(c1)
				meanEstimatedProbabilities,sdEstimatedProbabilities,sdEstimatedProbabilities,uselessMedian = computeMeanAndStandardError(e1)
				meanSquaredLoss,useless1,useless2,useless3 = computeMeanAndStandardError(s1)

				f1 = 0
				# Precision and Recall are calculated w.r.t label 1. So if everything converges to label 0, both P and R will be 0
				if meanPrecision != 0 and meanRecall != 0:
					f1 = (2*meanPrecision*meanRecall)/(meanPrecision+meanRecall)
				
				# Calculating medianPrecision and medianRecall might not make sense... because precicion and recall are dependent .... and median can select different values for them.
				#f1_median = (2*medianPrecision*medianRecall)/(medianPrecision+medianRecall)

				prefix = str(int(percentageOfGraph*100)) + "perc_" + str(noOfTimesToRepeat) + "repeat"
				if percentageOfGraph2 != 0:
					prefix = str(int(percentageOfGraph*100)) + "FL_" + str(int(percentageOfGraph2*100)) + "DE_" + str(noOfTimesToRepeat) + "repeat"
				
				print "\nFINAL .................. "
				print "Baseline_0 range:", Baseline_0
				t = sum(Baseline_0)/len(Baseline_0)
				t1 = t - Baseline_0[0]
				print "B_0_mean:",t
				print "B_0_std:",t1
				print "Baseline_1 range:", Baseline_1
				#outputTofile.append( [ Action + "ResultsBaselines.txt",prefix + "_Baseline_0" , str(trainingSize) , str(round(t,4)) , str(round(t1,4)) ] )
				t = sum(Baseline_1)/len(Baseline_1)
				t1 = t - Baseline_1[0]
				print "B_1_mean:",t
				print "B_1_std:",t1
				#outputTofile.append( [ Action + "ResultsBaselines.txt",prefix + "_Baseline_1" , str(trainingSize) , str(round(t,4)) , str(round(t1,4)) ] )
				print a1
				print "Prediction medianAccuracy:",medianAccuracy
				print "Prediction meanAccuracy:",meanAccuracy
				print "Prediction SD:",sd
				print "Prediction SE:",se
				print "Prediction MeanPrecision:",meanPrecision
				print "Prediction MeanRecall:",meanRecall
				print "Prediction F1:",f1
				print "MeanClassPrior:",meanClassPrior
				print "ClassPriorSD:",sdClassPrior
				print "ClassPriorSE:",seClassPrior
				print "MeanEstimatedProbabilities:\n",meanEstimatedProbabilities
				print "EstimatedProbabilitiesSD:\n",sdEstimatedProbabilities
				print "EstimatedProbabilitiesSE:\n",seEstimatedProbabilities
				print s1
				print "MeanSquaredLoss:",meanSquaredLoss
				#outputTofile.append( [ Action + "ResultsBaselines.txt",prefix , str(trainingSize) , str(round(meanAccuracy,4)) , str(round(sd,4)) , str(round(se,4)) , str(round(meanPrecision,4)) , str(round(meanRecall,4)) , str(round(f1,4)), str(round(medianAccuracy,4))])
				outputTofile.append( [ Action + "Results.txt",prefix , str(trainingSize) , str(round(meanSquaredLoss,4)) , str(round(meanAccuracy,4)) , str(round(sd,4)) , str(round(se,4)) , str(round(meanPrecision,4)) , str(round(meanRecall,4)) , str(round(f1,4)), str(round(medianAccuracy,4)), str(round(meanClassPrior[0],4)), str(round(sdClassPrior[0],4)), str(round(seClassPrior[0],4)), str(round(meanEstimatedProbabilities[0,0],4)), str(round(sdEstimatedProbabilities[0,0],4)), str(round(seEstimatedProbabilities[0,0],4)), str(round(meanEstimatedProbabilities[1,1],4)), str(round(sdEstimatedProbabilities[1,1],4)), str(round(seEstimatedProbabilities[1,1],4))])
				#outputTofile.append( [ Action + "Results.txt","Median_"+prefix , str(trainingSize) , str(round(medianAccuracy,4)) , str(round(0,4)) , str(round(0,4)) , str(round(0,4)) , str(round(0,4)) , str(round(0,4)) ])
				#print e1
				q1 = [i[0] for i in c1]
				print "prior"
				print q1
				print numpy.mean(q1)
				print numpy.median(q1)
				q2 =  [i[0,0] for i in e1]
				print "0given0"
				print q2
				print numpy.mean(q2)
				print numpy.median(q2)
				q3 = [i[1,1] for i in e1]
				print "1given1"
				print q3
				print numpy.mean(q3)
				print numpy.median(q3)
		for otf in outputTofile:
			writeToFile(otf)
