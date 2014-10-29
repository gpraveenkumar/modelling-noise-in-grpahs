import sys, math
from sets import Set
from collections import Counter
import random
import numpy
from multiprocessing import Pool
import gc

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

#silentremove('p_noB_gibbs.pyc')

from p_noB_gibbs import *


#random.seed(1)

binary = True
directed = False

#testSize = 0.3

# graph
edges = {}
label = {}

#f_in = open('../data/polblogs-nodes.txt')
f_in = open('../data/school074-nodes.txt')

# no need for first line...Skipping the header
junk_ = f_in.readline()

for line in f_in:
	fields = line.strip().split()
	label[int(float(fields[0]))] = int(float(fields[1]))
	edges[int(float(fields[0]))] = set([])

f_in.close()

#f_in = open('../data/polblogs-edges.txt')
f_in = open('../data/school074-edges.txt')

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





# Function to Flip the data
def func1(a,b):
	# Make a new graph with noise
	newGraph = dict(originalGraph)
	newLabels = dict(originalLabels)
	newTestLabels = list(testLabels)
	global nodeIdCounter

	percentageOfLabelFlips = a
	noOfTimesFlipLabels = b

	for notfl in range(noOfTimesFlipLabels):
		# Randomly sample a percentage of original label and flip it
		noOfLabelsToFlip = int(percentageOfLabelFlips*len(originalTrainLabels))
		labelsToFlip = random.sample(originalLabels,noOfLabelsToFlip)


		# Add new nodes and edges to the graph
		for i in originalTrainLabels:
			t = originalLabels[i]
			
			# Flip the labels. The XORing with 1 reverses the labels
			# 0^1 = 1
			# 1^1 = 0
			if i in labelsToFlip:
				t = t^1
			
			newLabels[nodeIdCounter] = t
			newGraph[nodeIdCounter] = set(originalGraph[i])
			nodeIdCounter += 1

		for i in testLabels:
			newLabels[nodeIdCounter] = originalLabels[i]
			newGraph[nodeIdCounter] = set(originalGraph[i])
			newTestLabels.append(nodeIdCounter) 
			nodeIdCounter += 1


	print "New Attr. Cor.:", computeCorrelation(computePairs(newGraph,newLabels))
	print "New +/- label counts:",computeLabelCounts(newLabels)

	acc = gibbsSampling(newGraph,newLabels,newTestLabels)
	return acc


# Function to Drop Labels
def func2(a,b):
	# Make a new graph with noise
	newGraph = dict(originalGraph)
	newLabels = dict(originalLabels)
	newTestLabels = list(testLabels)
	#newGraph = {}
	#newLabels = {}
	#newTestLabels = {}
	global nodeIdCounter

	percentageOfLabelFlips = a
	noOfTimesFlipLabels = b

	for notfl in range(noOfTimesFlipLabels):

		noOfLabelsToFlip = int(percentageOfLabelFlips*len(originalTrainLabels))
		labelsToDrop = random.sample(originalLabels,noOfLabelsToFlip)		

		for id in originalGraph:
			if id in labelsToDrop:
				continue
			else:
				newNeighbors = set([])
				for neighbor in originalGraph[id]:	
					if neighbor not in labelsToDrop:
						newNeighbors.add(neighbor)
				newGraph[nodeIdCounter] = newNeighbors
				newLabels[nodeIdCounter] = originalLabels[id]

				if id in testLabels:
					newTestLabels.append(nodeIdCounter)

				nodeIdCounter += 1


	print "New Attr. Cor.:", computeCorrelation(computePairs(newGraph,newLabels))
	print "New +/- label counts:",computeLabelCounts(newLabels)

	acc = gibbsSampling(newGraph,newLabels,newTestLabels)
	return acc



def func_star1(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return makeGraph(*a_b)

def makeGraph(onlyTrainLabels_keys,onlyTrainLabels,onlyTrainGraph,nodeIdCounter):
	# Randomly sample a percentage of original label and flip it
	#onlyTrainLabels_keys = onlyTrainLabels.keys()
	noOfLabelsToFlip = int(percentageOfLabelFlips*len(onlyTrainLabels_keys))
	labelsToFlip = random.sample(onlyTrainLabels_keys,noOfLabelsToFlip)

	# Map to store the relationships between nodes/labels in the training set and the new 
	# node formed due to flips. The is needed in order to replace old edges with new edges.
	map_onlyTrainLabelId_newLabelId = {}
	newGraph = {}
	newLabels = {}

	# Add new nodes and edges to the graph from the old training Data with label flips
	for i in onlyTrainLabels:
		t = onlyTrainLabels[i]
		
		# Flip the labels. The XORing with 1 reverses the labels
		# 0^1 = 1
		# 1^1 = 0
		if i in labelsToFlip:
			t = t^1
		
		# Add the modify labels to newLabels
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


# Function to Flip the data
def Flip(percentageOfLabelFlips,noOfTimesToFlipLabel,originalGraph,originalLabels,testLabels):
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

	arguments = []
	for i in range(10):
		ctr = nodeIdCounter + i*len(originalLabels)
		l = []
		l.append( onlyTrainLabels.keys() )
		l.append( onlyTrainLabels )
		l.append( onlyTrainGraph )
		l.append( ctr )
		arguments.append(l)

	pool = Pool(processes=7)
	y = pool.map(func_star1, arguments)

	allGraphs,allLabels = zip(*y)		

	for dic in allGraphs:
		newGraph.update(dic)

	for dic in allLabels:
		newLabels.update(dic)

	return (newGraph,newLabels)

#gibbsSampling(originalGraph,originalLabels,testLabels)


"""
noOftimes = 100
avg = 0

perc = 0.10
noOfFlips = [1,2,5,10]

f = open("../data/res/onlydrop-perc"+str(perc)+".txt",'w')
f.write("Results\n")
f.close()

for nop in noOfFlips:
	for i in range(noOftimes):
		avg += 1#func2(perc,nop)
	avg /= noOftimes
	f = open("../data/res/onlyflip-perc"+str(perc)+".txt",'a')
	f.write(str(nop) + " " + str(avg)+"\n")
	f.close()
#print "Average of 10 runs:", avg
"""
"""
arg_t = [originalGraph,originalLabels,testLabels]

arguments = []

for i in range(100):
	arguments.append(arg_t)
"""

def func_star(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return gibbsSampling(*a_b)



def computeMeanAndStandardError(vector):
	mean = numpy.mean(vector)
	sd = numpy.std(vector)
	se = sd / math.sqrt(len(vector))
	return (mean,sd,se)



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
# Input : fileName,Label,trainingSize,Accuracy_Mean,Accuracy_SD
# Output : None
def writeToFile(fileName,a,b,c,d):
	x = 2
	"""
	path = '../results/' + 'polBlogs_'
	f_out = open(path+fileName,'a')
	f_out.write(a + "\t" + b + "\t" + c + "\t" + d + "\n")
	f_out.close()
	"""



#trainingSize = 0.7
#percentageOfLabelFlips = 0.05   # express in fraction instead of percentage...incorrect naming, will update soon
#noOfTimesToFlipLabel = 10


for trainingSize in [0.7]:
	for percentageOfLabelFlips in [0]:
		for noOfTimesToFlipLabel in [0]:


				print "\n\n\n\n\ntrainingSize:",trainingSize," percentageOfLabelFlips: ",percentageOfLabelFlips," noOfTimesToFlipLabel: ",noOfTimesToFlipLabel

				testSize = 1-trainingSize
				noOfLabelsToMask = int(testSize*len(originalLabels))
				print "testLabels Size:",noOfLabelsToMask

				a1 = []
				e1 = []
				Baseline_0 =[1,0]
				Baseline_1 =[1,0]

				for i in range(1):

					testLabels = random.sample(originalLabels,noOfLabelsToMask)
					#originalTrainLabels = [i for i in originalLabels if i not in testLabels]

					#currentGraph,currentLabels = originalGraph,originalLabels
					currentGraph,currentLabels = Flip(percentageOfLabelFlips,noOfTimesToFlipLabel,originalGraph,originalLabels,testLabels)

					print "\nRepetition No.:",i
					print "Size of graph:",len(currentLabels)

					arg_t = [currentGraph,currentLabels,testLabels]
					arguments = []
					for i in range(7):
						arguments.append(list(arg_t))

					pool = Pool(processes=7)
					y = pool.map(func_star, arguments)

					accuracy, estimatedProbabities = zip(*y)
					mean,sd,se = computeMeanAndStandardError(accuracy)
					#print accuracy
					#print estimatedProbabities[0]
					e1.append(estimatedProbabities[0])

					predictedLabels = setLabelForBaselineAccuracies(currentLabels, testLabels, 0)
					curBaselineValue = computeAccuracy(currentLabels,testLabels, predictedLabels )
					Baseline_0 = updateBaselineRanges(Baseline_0,curBaselineValue)
					print "Baseline_0:", curBaselineValue
					predictedLabels = setLabelForBaselineAccuracies(currentLabels, testLabels, 1)
					curBaselineValue = computeAccuracy(currentLabels,testLabels, predictedLabels )
					Baseline_1 = updateBaselineRanges(Baseline_1,curBaselineValue)
					print "Baseline_1:", curBaselineValue
					print "Mean:",mean
					print "SD:",sd
					print "SE:",se
					print "estimatedProbabities:\n",estimatedProbabities[0]
					a1.append(mean)

					#Freeup space
					del arguments[:]
					gc.collect()
				#print se

				mean,sd,se = computeMeanAndStandardError(a1)

				prefix = str(percentageOfLabelFlips) + "perc_" + str(noOfTimesToFlipLabel) + "flips_" + str(trainingSize) + "trainSize"
				
				print "\nFINAL .................. "
				print "Baseline_0 range:", Baseline_0
				t = sum(Baseline_0)/len(Baseline_0)
				t1 = t - Baseline_0[0]
				print "B_0_mean:",t
				print "B_0_std:",t1
				print "Baseline_1 range:", Baseline_1
				writeToFile("flipResultsBaselines.txt",prefix + "_Baseline_0" , str(trainingSize) , str(round(t,4)) , str(round(t1,4)) )
				t = sum(Baseline_1)/len(Baseline_1)
				t1 = t - Baseline_1[0]
				print "B_1_mean:",t
				print "B_1_std:",t1
				writeToFile("flipResultsBaselines.txt",prefix + "_Baseline_1" , str(trainingSize) , str(round(t,4)) , str(round(t1,4)) )
				print a1
				print "Prediction Mean:",mean
				print "Prediction SD:",sd
				print "Prediction SE:",se
				writeToFile("flipResultsBaselines.txt",prefix , str(trainingSize) , str(round(t,4)) , str(round(t1,4)) )
				writeToFile("flipResults.txt",prefix , str(trainingSize) , str(round(t,4)) , str(round(t1,4)) )
				#print e1
