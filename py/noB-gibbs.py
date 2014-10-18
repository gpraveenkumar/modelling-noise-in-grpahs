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
	if binary:
		label_0 = int(label_0 >= 1)
		label_1 = int(label_1 >= 1)
		label_2 = int(label_2 >= 1)

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
t = Counter()
for x in labels_0:
	t[labels_0[x]] += 1
print t

# pairings for computing correlations
pairs_0 = []
pairs_1 = []
pairs_2 = []

edges1 = dict(edges)
print "length of edges : " + str(len(edges))

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

print "length of edges : " + str(len(edges))

# Compute the pairings
for id, neighbors in edges.iteritems():
	# cycle through the neighbors
	for neighbor in neighbors:

		pairs_0.append([labels_0[id], labels_0[neighbor]])
		pairs_1.append([labels_1[id], labels_1[neighbor]])
		pairs_2.append([labels_2[id], labels_2[neighbor]])

		if not directed:
			pairs_0.append([labels_0[neighbor], labels_0[id]])
			pairs_1.append([labels_1[neighbor], labels_1[id]])
			pairs_2.append([labels_2[neighbor], labels_2[id]])


#print pairs_0
print "length : " + str(len(pairs_0))

f = open("../data/attributeCorrelationCheck.txt", 'w')
f.write("A B"+'\n')
f.write('\n'.join( ' '.join([str(int(j)) for j in i]) for i in pairs_0))
f.close()


### Label 0
mean0_0 = 0.0
mean0_1 = 0.0
std0_0 = 0.0
std0_1 = 0.0
cov_0 = 0.0

for pair in pairs_0:
	#print pair
	mean0_0 += pair[0]
	mean0_1 += pair[1]

mean0_0 /= len(pairs_0)
mean0_1 /= len(pairs_0)

for pair in pairs_0:
	cov_0 += (pair[0] - mean0_0)*(pair[1] - mean0_1)
	std0_0 += (pair[0] - mean0_0)**2
	std0_1 += (pair[1] - mean0_1)**2

std0_0 = math.sqrt(std0_0)
std0_1 = math.sqrt(std0_1)
print 'Label 0:', cov_0 / (std0_0*std0_1)



### Label 1
mean1_0 = 0.0
mean1_1 = 0.0
std1_0 = 0.0
std1_1 = 0.0
cov_1 = 0.0

for pair in pairs_1:
	mean1_0 += pair[0]
	mean1_1 += pair[1]

mean1_0 /= len(pairs_1)
mean1_1 /= len(pairs_1)

for pair in pairs_1:
	cov_1 += (pair[0] - mean1_0)*(pair[1] - mean1_1)
	std1_0 += (pair[0] - mean1_0)**2
	std1_1 += (pair[1] - mean1_1)**2

std1_0 = math.sqrt(std1_0)
std1_1 = math.sqrt(std1_1)
print 'Label 1:', cov_1 / (std1_0*std1_1)




### Label 2
mean2_0 = 0.0
mean2_1 = 0.0
std2_0 = 0.0
std2_1 = 0.0
cov_2 = 0.0

for pair in pairs_2:
	mean2_0 += pair[0]
	mean2_1 += pair[1]

mean2_0 /= len(pairs_2)
mean2_1 /= len(pairs_2)

for pair in pairs_2:
	cov_2 += (pair[0] - mean2_0)*(pair[1] - mean2_1)
	std2_0 += (pair[0] - mean2_0)**2
	std2_1 += (pair[1] - mean2_1)**2

std2_0 = math.sqrt(std2_0)
std2_1 = math.sqrt(std2_1)
print 'Label 2:', cov_2 / (std2_0*std2_1)





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



## noB starts here

#class priors
t = Counter()
for x in originalLabels:
	t[originalLabels[x]] += 1
print t

# making a fraction(=testsize) of labels
noOfLabelsToMask = int(testSize*len(originalLabels))
#print noOfLabelsToMask
testLabels = random.sample(originalLabels,noOfLabelsToMask)

print len(testLabels)
originalTrainLabels = [i for i in originalLabels if i not in testLabels]

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

	classPrior = [0]*2
	classPrior[0] = t[0] / (t[0] + t[1] + 0.0)
	classPrior[1] = 1 - classPrior[0]
	print classPrior

	# conditional probabilites
	estimatedProbabities = numpy.zeros([2,2])

	for id, neighbors in G.iteritems():
		if id in testLabels:
			continue
		# cycle through the neighbors
		for neighbor in neighbors:
			if neighbor in testLabels:
				continue
			estimatedProbabities[ label[id], label[neighbor] ] += 1
			if not directed:
				estimatedProbabities[ label[neighbor], label[id] ] += 1

	# Check if there is still attr. corr.

	print estimatedProbabities
	print sum(sum(estimatedProbabities))
	estimatedProbabities /= sum(sum(estimatedProbabities))
	print estimatedProbabities
	return (classPrior,estimatedProbabities)

def computeParameters(G,label):
	#class priors
	t = Counter()
	for x in label:
		t[label[x]] += 1
	
	classPrior = [0]*2
	classPrior[0] = t[0] / (t[0] + t[1] + 0.0)
	classPrior[1] = 1 - classPrior[0]

	#print t
	#print classPrior

	# conditional probabilites
	estimatedProbabities = numpy.zeros([2,2])

	#global edges
	for id, neighbors in G.iteritems():
		# cycle through the neighbors
		for neighbor in neighbors:
			estimatedProbabities[ label[id], label[neighbor] ] += 1
			if not directed:
				estimatedProbabities[ label[neighbor], label[id] ] += 1

	return (t,classPrior,estimatedProbabities)




def f1(nodeLabel, currentLabelEstimates, neighbors, estimatedProbabities, classPrior):
	noOfZeroLabeledNeighbours = 0
	#noOfNeighbours = 0
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

def initializeUnknownLabelsForGibbsSampling(G,label,testLabels):
	# Assign initial labels to all test labels just using the priors
	currentLabelEstimates = dict(label)
	classPrior, estimatedProbabities = computeInitialParameters(G,label,testLabels)

	for node in testLabels:
		neighbors = G[node]

		#removing all the edges to labels in the test set for computing initial estimates. Original Graph is unaffected.
		newNeighbors = set(neighbors)
		for i in neighbors: 
			if i in testLabels:
				newNeighbors.remove(i)
		neighbors = set(newNeighbors)

		currentLabelEstimates[node] = f2(currentLabelEstimates, neighbors, estimatedProbabities, classPrior)

	t, classPrior, estimatedProbabities = computeParameters(G,currentLabelEstimates)
	print classPrior
	print estimatedProbabities
	return (classPrior,estimatedProbabities,currentLabelEstimates)


## Gibbs Sampling

def gibbsSampling(edges,label,testLabels):
		
	## Step 2 of algo
	classPrior,estimatedProbabities,currentLabelEstimates = initializeUnknownLabelsForGibbsSampling(edges,label,testLabels)

	nodeTraversalOrder = testLabels
	random.shuffle(nodeTraversalOrder)

	burnin = 2
	iteration = 10

	resultingLabels = {}
	for i in label:
		resultingLabels[i] = 0


	## Step 3 of algo
	for i in range(iteration):
		checkLabelDifferenceBetweenIterations = 0
		for node in nodeTraversalOrder:
			neighbors = edges[node]
			currentLabelEstimates[node] = f2(currentLabelEstimates, neighbors, estimatedProbabities, classPrior)

			t, classPrior, estimatedProbabities = computeParameters(edges,currentLabelEstimates)

		if i > burnin:
			for j in currentLabelEstimates:
				if currentLabelEstimates[j] == 1:
					resultingLabels[j] += 1

				temp = (resultingLabels[j] + 0.0)/(i - burnin) 
				temp = int(temp >= 0.5)
				if temp != label[j]:
					checkLabelDifferenceBetweenIterations += 1
		
		print "LabelDifferenceBetweenIterations : " + str(checkLabelDifferenceBetweenIterations)	
		print "----------------------------------\n" + "Iteration no : " +str(i)
		print t
		print classPrior
		print estimatedProbabities

	for i in resultingLabels:
		resultingLabels[i] = (resultingLabels[i] + 0.0)/(iteration - burnin) 
		resultingLabels[i] = int(resultingLabels[i]  > 0.5)

	ctr = 0
	for i in label:
		if label[i] != resultingLabels[i]:
			ctr += 1
	print ctr
	accuracy = numpy.zeros([2,2])

	for i in label:
		accuracy[ label[i], resultingLabels[i] ] += 1

	print accuracy

#gibbsSampling(edges,labels)





percentageOfLabelFlips = 5

gibbsSampling(originalGraph,originalLabels,testLabels)