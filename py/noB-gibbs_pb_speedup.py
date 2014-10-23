import sys, math
from sets import Set
from collections import Counter
import random
import numpy


binary = True
directed = False

testSize = 0.3

# graph
edges = {}
label = {}


f_in = open('../data/polblogs-nodes.txt')

# no need for first line...Skipping the header
junk_ = f_in.readline()

for line in f_in:
	fields = line.strip().split()
	label[int(float(fields[0]))] = int(float(fields[1]))
	edges[int(float(fields[0]))] = set([])

f_in.close()

f_in = open('../data/polblogs-edges.txt')

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

#print "length of edges : " + str(len(edges))

# Compute the pairings
# pairings for computing correlations
pairs = []
for id, neighbors in edges.iteritems():
	# cycle through the neighbors
	for neighbor in neighbors:

		pairs.append([label[id], label[neighbor]])



# Function the computes correlation. 
# Take in a list of pairs each of which a list of [startnode,endnode]
# Returns correlation
def computeCorrelation(pairs):
	mean0_0 = 0.0
	mean0_1 = 0.0
	std0_0 = 0.0
	std0_1 = 0.0
	cov_0 = 0.0

	for pair in pairs:
		#print pair
		mean0_0 += pair[0]
		mean0_1 += pair[1]

	mean0_0 /= len(pairs)
	mean0_1 /= len(pairs)

	for pair in pairs:
		cov_0 += (pair[0] - mean0_0)*(pair[1] - mean0_1)
		std0_0 += (pair[0] - mean0_0)**2
		std0_1 += (pair[1] - mean0_1)**2

	std0_0 = math.sqrt(std0_0)
	std0_1 = math.sqrt(std0_1)
	return cov_0 / (std0_0*std0_1)


print 'Label 0:', computeCorrelation(pairs) 



# A function to Compute the pairings to calculate correlation.
# Input - Graph
# Output - list of pairs each of which a list of [startnode,endnode]
def computePairs(edges,label):

	pairs = []
	
	for id, neighbors in edges.iteritems():
		# cycle through the neighbors
		for neighbor in neighbors:
			pairs.append([label[id], label[neighbor]])

			if not directed:
				pairs.append([label[neighbor], label[id]])

	return pairs	
	

# A function to Compute the number of labels in each class
# Input - list of labels
# Output - count of labels in each
def computeLabelCounts(label,testLabels=None):
	if testLabels != None:
		t = Counter()
		for x in testLabels:
			t[label[x]] += 1
	else:	
		t = Counter()
		for x in label:
			t[label[x]] += 1
	return t

def computeEstimatedProbabilites(G,label):
	estimatedProbabities = numpy.zeros([2,2])

	for id, neighbors in G.iteritems():
		# cycle through the neighbors
		for neighbor in neighbors:
			estimatedProbabities[ label[id], label[neighbor] ] += 1
			#if not directed:
			#	estimatedProbabities[ label[neighbor], label[id] ] += 1

	return estimatedProbabities


#label = labels_0

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
#class priors
t = Counter()
for x in originalLabels:
	t[originalLabels[x]] += 1
"""
print "Original +/- label counts:",computeLabelCounts(originalLabels)

# masking a fraction(=testsize) of labels
noOfLabelsToMask = int(testSize*len(originalLabels))
testLabels = random.sample(originalLabels,noOfLabelsToMask)

print "No. of test labels:",len(testLabels)
originalTrainLabels = [i for i in originalLabels if i not in testLabels]

x = computeEstimatedProbabilites(originalGraph, originalLabels)
print x
print sum(sum(x))

#print len(label)
#print len(testLabels)
#print len(trainLabels)



def computeInitialParameters(G,label,testLabels):
	#class priors
	t = Counter()
	for x in label:
		if x in testLabels:
			continue
		t[label[x]] += 1
	print t

	classPriorCounts = {}
	classPriorCounts[0] = t[0]
	classPriorCounts[1] = t[1]

	classPrior = [0]*2
	classPrior[0] = t[0] / (t[0] + t[1] + 0.0)
	classPrior[1] = 1 - classPrior[0]
	print classPrior

	# conditional probabilites
	estimatedCounts = numpy.zeros([2,2])

	for id, neighbors in G.iteritems():
		if id in testLabels:
			continue
		# cycle through the neighbors
		for neighbor in neighbors:
			if neighbor in testLabels:
				continue
			estimatedCounts[ label[id], label[neighbor] ] += 1
			#if not directed:
			#	estimatedCounts[ label[neighbor], label[id] ] += 1

	# Check if there is still attr. corr.

	print "\nInitial Parameter Estimates before estimating UNKNOWN labels:"
	print t
	print estimatedCounts
	print sum(sum(estimatedCounts))
	estimatedProbabities = estimatedCounts / sum(sum(estimatedCounts))
	print estimatedProbabities,"\n"
	return (classPrior,estimatedProbabities,classPriorCounts,estimatedCounts)



def computeParameters(G,label):
	#class priors
	t = Counter()
	for x in label:
		t[label[x]] += 1
	
	#print '\n','\n',t,'\n','\n'

	classPrior = [0]*2
	classPrior[0] = t[0] / (t[0] + t[1] + 0.0)
	classPrior[1] = 1 - classPrior[0]

	#print t
	#print classPrior

	# conditional probabilites
	estimatedCounts = numpy.zeros([2,2])

	#global edges
	for id, neighbors in G.iteritems():
		# cycle through the neighbors
		for neighbor in neighbors:
			estimatedCounts[ label[id], label[neighbor] ] += 1
			#if not directed:
			#	estimatedCounts[ label[neighbor], label[id] ] += 1

	estimatedProbabities = estimatedCounts / sum(sum(estimatedCounts))

	return (t,classPrior,estimatedProbabities,estimatedCounts)



def computeParametersOneNode(G,label,node,newNodeLabel,oldNodeLabel,classPriorCounts,estimatedCounts):

	classPriorCounts[oldNodeLabel] -= 1
	classPriorCounts[newNodeLabel] += 1

	classPrior = [0]*2
	classPrior[0] = classPriorCounts[0] / (classPriorCounts[0] + classPriorCounts[1] + 0.0)
	classPrior[1] = 1 - classPrior[0]

	neighbors = G[node]
	#Remove edges for old lable
	for neighbor in neighbors:
		estimatedCounts[ oldNodeLabel, label[neighbor] ] -= 1
		estimatedCounts[ label[neighbor], oldNodeLabel ] -= 1

		estimatedCounts[ newNodeLabel, label[neighbor] ] += 1
		estimatedCounts[ label[neighbor], newNodeLabel ] += 1

	estimatedProbabities = estimatedCounts / sum(sum(estimatedCounts))
	
	return (classPriorCounts,classPrior,estimatedProbabities,estimatedCounts)		


def f1(nodeLabel, currentLabelEstimates, neighbors, estimatedProbabities, classPrior):
	noOfZeroLabeledNeighbours = 0
	#noOfNeighbours = 0
	#print estimatedProbabities
	for i in neighbors:
		if currentLabelEstimates[i] == 0:
			noOfZeroLabeledNeighbours += 1
	#print str(noOfZeroLabeledNeighbours) + "/ " + str(len(neighbors))
	#print str(nodeLabel) + " ---- " + str(classPrior[nodeLabel]) + "----" + str(estimatedProbabities[nodeLabel,0]) + " , " + str(estimatedProbabities[nodeLabel,1])
	prob = classPrior[nodeLabel] * math.pow( estimatedProbabities[nodeLabel,0] , noOfZeroLabeledNeighbours ) * math.pow(estimatedProbabities[nodeLabel,1] ,len(neighbors)-noOfZeroLabeledNeighbours)
	return prob



def f2(currentLabelEstimates, neighbors, estimatedProbabities, classPrior):
	class0 = f1(0,currentLabelEstimates, neighbors, estimatedProbabities, classPrior)
	class1 = f1(1,currentLabelEstimates, neighbors, estimatedProbabities, classPrior)
	denominator = class0 + class1
	#print str(class0) + " " + str(class1)
	class0 = class0/denominator
	class1 = class1/denominator

	#print str(class0) + " " + str(class1)
	if random.uniform(0,1) < class0:
		return 0
	else:
		return 1

# Note - estimated probability is actually estimated counts

def initializeUnknownLabelsForGibbsSampling(G,label,testLabels):
	# Assign initial labels to all test labels just using the priors
	currentLabelEstimates = dict(label)
	classPrior, estimatedProbabities, baseClassPriorCounts, baseEstimatedCounts = computeInitialParameters(G,label,testLabels)

	for node in testLabels:
		neighbors = G[node]

		#removing all the edges to labels in the test set for computing initial estimates. Original Graph is unaffected.
		newNeighbors = set(neighbors)
		for i in neighbors: 
			if i in testLabels:
				newNeighbors.remove(i)
		neighbors = set(newNeighbors)

		currentLabelEstimates[node] = f2(currentLabelEstimates, neighbors, estimatedProbabities, classPrior)

	classPriorCounts, classPrior, estimatedProbabities, estimatedCounts = computeParameters(G,currentLabelEstimates)

	print "Initial Parameter Estimates after estimating UNKNOWN labels:"
	print classPriorCounts
	print "Current Attr. Cor.:", computeCorrelation(computePairs(G,currentLabelEstimates))
	print classPrior
	print estimatedCounts
	print sum(sum(estimatedCounts))
	print estimatedProbabities,"\n"

	return (classPrior,estimatedProbabities,currentLabelEstimates,classPriorCounts,estimatedCounts)



## Gibbs Sampling

def gibbsSampling(edges,label,testLabels):
		
	## Step 2 of algo
	classPrior,estimatedProbabities,currentLabelEstimates,classPriorCounts,estimatedCounts = initializeUnknownLabelsForGibbsSampling(edges,label,testLabels)

	nodeTraversalOrder = testLabels
	random.shuffle(nodeTraversalOrder)

	burnin = 25
	iteration = 100

	resultingLabels = {}
	for i in label:
		resultingLabels[i] = 0

	LabelDifferenceBetweenIterationsCounter = 0
	previousLabelDifferenceBetweenIterations = 0

	## Step 3 of algo
	for i in range(iteration):
		
		LabelDifferenceBetweenIterations = 0
		for node in nodeTraversalOrder:
			#print "\nNode ",node
			#print "Before Attr. Cor.:", computeCorrelation(computePairs(edges,currentLabelEstimates))
			neighbors = edges[node]
			previousEstimate = currentLabelEstimates[node]
			currentLabelEstimates[node] = f2(currentLabelEstimates, neighbors, estimatedProbabities, classPrior)
			
			# Recompute probabilites only if the previousEstimate of label value is different from its current Value.
			if previousEstimate != currentLabelEstimates[node]:
				classPriorCounts, classPrior, estimatedProbabities, estimatedCounts = computeParametersOneNode(edges,currentLabelEstimates,node,currentLabelEstimates[node],previousEstimate,classPriorCounts,estimatedCounts)
			#print "After Attr. Cor.:", computeCorrelation(computePairs(edges,currentLabelEstimates))
			#print classPrior
			#print estimatedProbabities
		
		if i > burnin:
			for j in currentLabelEstimates:
				if currentLabelEstimates[j] == 1:
					resultingLabels[j] += 1

				temp = (resultingLabels[j] + 0.0)/(i - burnin) 
				temp = int(temp >= 0.5)
				if temp != label[j]:
					LabelDifferenceBetweenIterations += 1

		if i >= 2*burnin:
			# Check if the numbers of labels estimated differ from the previous interation
			if LabelDifferenceBetweenIterations == previousLabelDifferenceBetweenIterations:
				LabelDifferenceBetweenIterationsCounter += 1
			else:
				LabelDifferenceBetweenIterationsCounter = 0
				previousLabelDifferenceBetweenIterations = LabelDifferenceBetweenIterations
			
			#If the estimates don't change for 100 interations, we can exit considering it has converged
			if LabelDifferenceBetweenIterationsCounter >= 100:
				print "Interations ended at " + str(i) + " as estimates have not changed!"

				# In the absence on this line
				iteration = i
				break

		if 0:#not i%10:
			print "\n--------------------------------------------------\n" + "Iteration no : " +str(i)
			print "LabelDifferenceBetweenIterations : " + str(LabelDifferenceBetweenIterations)	
			print "Current Attr. Cor.:", computeCorrelation(computePairs(edges,currentLabelEstimates))
			print classPriorCounts
			print classPrior
			print estimatedCounts
			#print sum(sum(estimatedCounts))
			print estimatedProbabities
	#print resultingLabels
	for i in resultingLabels:
		resultingLabels[i] = (resultingLabels[i] + 0.0)/(iteration - burnin) 
		resultingLabels[i] = int(resultingLabels[i]  >= 0.5)
	#print resultingLabels
	ctr = 0
	for i in label:
		if label[i] != resultingLabels[i]:
			ctr += 1


	print "\nFinal Results\nNo. of Labels Mismatched:",ctr

	accuracy = numpy.zeros([2,2])
	for i in testLabels:
		accuracy[ label[i], resultingLabels[i] ] += 1

	accp = (accuracy[0,0]+accuracy[1,1])/sum(sum(accuracy))

	print "Accuracy:",accuracy
	print "% = ",accp
	print "No. of Test Example:",computeLabelCounts(label,testLabels)
	print "Final Labels:",computeLabelCounts(resultingLabels,testLabels)

	return accp

print "\nStart of Gibbs...."
#gibbsSampling(originalGraph,originalLabels,testLabels)


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


#gibbsSampling(originalGraph,originalLabels,testLabels)



noOftimes = 100
avg = 0

perc = 0.10
noOfFlips = [1,2,5,10]

f = open("../data/res/onlydrop-perc"+str(perc)+".txt",'w')
f.write("Results\n")
f.close()

for nop in noOfFlips:
	for i in range(noOftimes):
		avg += func2(perc,nop)
	avg /= noOftimes
	f = open("../data/res/onlyflip-perc"+str(perc)+".txt",'a')
	f.write(str(nop) + " " + str(avg)+"\n")
	f.close()
#print "Average of 10 runs:", avg


g1 = {}
"""
g1[0] = set([1,2,3])
g1[1] = set([2,3,4])
g1[2] = set([3,4,0])
g1[3] = set([4,1,0])
g1[4] = set([5,0,1])
g1[5] = set([6,7])
g1[6] = set([7,8])
g1[7] = set([8,9])
g1[8] = set([9,6])
g1[9] = set([5,0])
"""

g1[0] = set([1,9])
g1[1] = set([0,2])
g1[2] = set([1,3])
g1[3] = set([2,4])
g1[4] = set([3,5])
g1[5] = set([4,6])
g1[6] = set([5,7])
g1[7] = set([6,8])
g1[8] = set([7,9])
g1[9] = set([8,0])


ol = {}
ol[0] = 0 
ol[1] = 0 
ol[2] = 0 
ol[3] = 0 
ol[4] = 0 
ol[5] = 1
ol[6] = 1
ol[7] = 1
ol[8] = 1
ol[9] = 1

tl = [4,1]
#gibbsSampling(g1,ol,tl)

g1 = {}
g1[0] = set([1,2,3])
g1[1] = set([2,3,0])
g1[2] = set([3,1,0])
g1[3] = set([2,1,0])
ol = {}
ol[0] = 0 
ol[1] = 1 
ol[2] = 0 
ol[3] = 1 
tl = [0]
#gibbsSampling(g1,ol,tl)