import sys, math
from sets import Set
from collections import Counter
import random
import numpy



#f_in = open('../data/school034-parsed.txt')
f_in = open('../data/school074-parsed.txt')

binary = True
directed = False

testSize = 0.3

# graph
edges = {}
labels_0 = {}
labels_1 = {}
labels_2 = {}

# no need for first line...
junk_ = f_in.readline()

# read them all
for line in f_in:
	fields = line.strip().split()
	id = fields[0]
	neighbors = fields[1:11] # 10 neighbors
	label_0 = fields[11] # the three possible labels
	label_1 = fields[12]
	label_2 = fields[13]

	# check ID
	if id == 'NA':
		continue

	# check for a label
	if label_0 == 'NA' or label_1 == 'NA' or label_1 == '99'or label_2 == 'NA':
		continue

	# toint / store in sets
	label_0 = int(label_0)
	label_1 = int(label_1)
	label_2 = int(label_2)

	# If we're using a 1/0 representation
	binaryCutoff = 1
	if binary:
		label_0 = int(label_0 >= binaryCutoff)
		label_1 = int(label_1 >= binaryCutoff)
		label_2 = int(label_2 >= binaryCutoff)

	# assign them all their labels
	labels_0[id] = label_0
	labels_1[id] = label_1
	labels_2[id] = label_2


	# insert new edges if not done yet
	if id not in edges:
		edges[id] = Set([])

	# for each neighbor
	for neighbor in neighbors:
		if neighbor != 'NA':
			if neighbor not in edges:
				edges[neighbor] = Set([])

			edges[id].add(neighbor)
			if not directed:
				edges[neighbor].add(id)

f_in.close()

l = []

for id, neighbors in edges.iteritems():
	if id not in labels_0 or id not in labels_1 or id not in labels_2:
		continue
	#print "l"
	t = []
	t.append(id)
	for neighbor in neighbors:
		t.append(neighbor)
	t.append(str(labels_0[id]))
	t.append(str(labels_1[id]))
	t.append(str(labels_2[id]))
	l.append(t)

"""
f = open('joel.txt','w')
f.write('\n'.join( ' '.join(i) for i in l))
f.close()
"""

#print labels
"""
t = Counter()
for x in labels_0:
	t[labels_0[x]] += 1
print t
"""

# pairings for computing correlations
pairs_0 = []
pairs_1 = []
pairs_2 = []

edges1 = dict(edges)
#print "length of edges : " + str(len(edges))

for id, neighbors in edges1.iteritems():
	# praveen removed all of these
	if id not in labels_0 or id not in labels_1 or id not in labels_2:
		if id in edges:
			del edges[id]
			continue
	
	n = set(neighbors)
	for neighbor in neighbors:
		if neighbor not in labels_0 or neighbor not in labels_1 or neighbor not in labels_2:
			n.remove(neighbor)
	edges[id] = n

# remove those node with no neighbours
for id, neighbors in edges1.iteritems():
	if len(neighbors) == 0:
		del edges[id]
		del labels_0[id]
		del labels_1[id]
		del labels_2[id]

del edges1

#print "length of edges : " + str(len(edges))

# Compute the pairings
for id, neighbors in edges.iteritems():
	# cycle through the neighbors
	for neighbor in neighbors:

		pairs_0.append([labels_0[id], labels_0[neighbor]])
		pairs_1.append([labels_1[id], labels_1[neighbor]])
		pairs_2.append([labels_2[id], labels_2[neighbor]])

		"""
		if not directed:
			pairs_0.append([labels_0[neighbor], labels_0[id]])
			pairs_1.append([labels_1[neighbor], labels_1[id]])
			pairs_2.append([labels_2[neighbor], labels_2[id]])
		"""


#print pairs_0
#print "length : " + str(len(pairs_0))

f = open("../data/attributeCorrelationCheck.txt", 'w')
f.write("A B"+'\n')
f.write('\n'.join( ' '.join([str(int(j)) for j in i]) for i in pairs_0))
f.close()


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


print 'Label 0:', computeCorrelation(pairs_0) 
print 'Label 1:', computeCorrelation(pairs_1) 
print 'Label 2:', computeCorrelation(pairs_2) 



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


label = labels_0

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



def computeParameters(G,label,testLabels,baseClassPriorCounts, baseEstimatedCounts):
	#class priors
	# Compute only for the test labels based on current estimates
	t = Counter()
	for x in testLabels:
		t[label[x]] += 1
	
	#print '\n','\n',t,'\n','\n'

	# class prior = no. of training labels of the training class + no. of test labels in the current esitmate belonging to that class
	classPriorCount = Counter()
	classPriorCount[0] = baseClassPriorCounts[0] + t[0]
	classPriorCount[1] = baseClassPriorCounts[1] + t[1]

	classPrior = [0]*2
	classPrior[0] = classPriorCount[0] / (classPriorCount[0] + classPriorCount[1] + 0.0)
	classPrior[1] = 1 - classPrior[0]

	#print t
	#print classPrior

	# conditional probabilites
	
	# Assign it to the base values of counts
	estimatedCounts = numpy.zeros([2,2])
	estimatedCounts[0,0] = baseEstimatedCounts[0,0]
	estimatedCounts[0,1] = baseEstimatedCounts[0,1]
	estimatedCounts[1,0] = baseEstimatedCounts[1,0]
	estimatedCounts[1,1] = baseEstimatedCounts[1,1]

	#global edges
	#for id, neighbors in G.iteritems():
	for id in testLabels:
		neighbors = G[id]
		# cycle through the neighbors
		for neighbor in neighbors:
			estimatedCounts[ label[id], label[neighbor] ] += 1

			# Adding this as a part of speedup. This won't work for directed graphs. Speeds wont work for directed graphs
			if neighbor not in testLabels:
				estimatedCounts[ label[neighbor], label[id] ] += 1
			#if not directed:
			#	estimatedCounts[ label[neighbor], label[id] ] += 1

	estimatedProbabities = estimatedCounts / sum(sum(estimatedCounts))

	return (classPriorCount,classPrior,estimatedProbabities,estimatedCounts)



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

	t, classPrior, estimatedProbabities, estimatedCounts = computeParameters(G,currentLabelEstimates,testLabels,baseClassPriorCounts, baseEstimatedCounts)

	print "Initial Parameter Estimates after estimating UNKNOWN labels:"
	print t
	print "Current Attr. Cor.:", computeCorrelation(computePairs(G,currentLabelEstimates))
	print classPrior
	print estimatedCounts
	print sum(sum(estimatedCounts))
	print estimatedProbabities,"\n"

	return (classPrior,estimatedProbabities,currentLabelEstimates,baseClassPriorCounts, baseEstimatedCounts)



## Gibbs Sampling

def gibbsSampling(edges,label,testLabels):
		
	## Step 2 of algo
	classPrior,estimatedProbabities,currentLabelEstimates,baseClassPriorCounts, baseEstimatedCounts = initializeUnknownLabelsForGibbsSampling(edges,label,testLabels)

	nodeTraversalOrder = testLabels
	random.shuffle(nodeTraversalOrder)

	burnin = 2
	iteration = 10

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
			currentLabelEstimates[node] = f2(currentLabelEstimates, neighbors, estimatedProbabities, classPrior)

			t, classPrior, estimatedProbabities, estimatedCounts = computeParameters(edges,currentLabelEstimates,testLabels,baseClassPriorCounts, baseEstimatedCounts)
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
				break

		if i:#not i%10:
			print "\n--------------------------------------------------\n" + "Iteration no : " +str(i)
			print "LabelDifferenceBetweenIterations : " + str(LabelDifferenceBetweenIterations)	
			print "Current Attr. Cor.:", computeCorrelation(computePairs(edges,currentLabelEstimates))
			print t
			print classPrior
			print estimatedCounts
			print sum(sum(estimatedCounts))
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
	for i in label:
		accuracy[ label[i], resultingLabels[i] ] += 1

	print "Accuracy:",accuracy
	print "No. of Test Example:",computeLabelCounts(label,testLabels)
	print "Final Labels:",computeLabelCounts(resultingLabels,testLabels)
	q = computeEstimatedProbabilites(edges,resultingLabels)
	print q
	print sum(sum(q))

print "\nStart of Gibbs...."
#gibbsSampling(originalGraph,originalLabels,testLabels)



# Make a new graph with noise
newGraph = dict(originalGraph)
newLabels = dict(originalLabels)
newTestLabels = list(testLabels)


percentageOfLabelFlips = 5
noOfTimesFlipLabels = 2

for notfl in range(noOfTimesFlipLabels):
	# Randomly sample a percentage of original label and flip it
	noOfLabelsToFlip = int(testSize*len(originalTrainLabels))
	labelsToFlip = random.sample(originalLabels,noOfLabelsToMask)


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



gibbsSampling(originalGraph,originalLabels,testLabels)
#gibbsSampling(newGraph,newLabels,newTestLabels)

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
