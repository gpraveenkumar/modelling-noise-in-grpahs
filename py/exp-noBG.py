import sys, math
from sets import Set
from collections import Counter
import random
import numpy
import os

os.remove('p_noB_gibbs.pyc')
from p_noB_gibbs import *
#import p_noB_gibbs

binary = True
directed = False

testSize = 0.3

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


gibbsSampling(originalGraph,originalLabels,testLabels)



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


