import math
from sets import Set
from collections import Counter
import random
import numpy




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


# A function to Compute the pairings to calculate correlation.
# Input - Graph
# Output - list of pairs each of which a list of [startnode,endnode]
def computePairs(edges,label):

	pairs = []
	
	for id, neighbors in edges.iteritems():
		# cycle through the neighbors
		for neighbor in neighbors:
			pairs.append([label[id], label[neighbor]])

			#if not directed:
			#	pairs.append([label[neighbor], label[id]])

	return pairs	



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



def computeInitialParameters(G,label,testLabels):
	#class priors
	t = Counter()
	for x in label:
		if x in testLabels:
			continue
		t[label[x]] += 1
	#print t

	classPriorCounts = {}
	classPriorCounts[0] = t[0]
	classPriorCounts[1] = t[1]

	classPrior = [0]*2
	classPrior[0] = t[0] / (t[0] + t[1] + 0.0)
	classPrior[1] = 1 - classPrior[0]
	#print classPrior

	# conditional probabilites
	estimatedCounts = numpy.zeros([2,2])
	estimatedProbabities = numpy.zeros([2,2])

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

	#print "\nInitial Parameter Estimates before estimating UNKNOWN labels:"
	#print t
	#print estimatedCounts
	#print sum(sum(estimatedCounts))
	#estimatedProbabities = estimatedCounts / sum(sum(estimatedCounts))
	estimatedProbabities[:,0] = (estimatedCounts[:,0] + 1) / (sum(estimatedCounts[:,0]) + 2)
	estimatedProbabities[:,1] = (estimatedCounts[:,1] + 1) / (sum(estimatedCounts[:,1]) + 2)
	#print estimatedProbabities,"\n"
	return (classPrior,estimatedProbabities,classPriorCounts,estimatedCounts)



def f1(nodeLabel, currentLabelEstimates, neighbors, estimatedProbabities, classPrior):
	noOfZeroLabeledNeighbours = 0
	for i in neighbors:
		if currentLabelEstimates[i] == 0:
			noOfZeroLabeledNeighbours += 1
	#print str(noOfZeroLabeledNeighbours) + "/ " + str(len(neighbors))
	#print str(nodeLabel) + " ---- " + str(classPrior[nodeLabel]) + "----" + str(estimatedProbabities[nodeLabel,0]) + " , " + str(estimatedProbabities[nodeLabel,1])
	
	#prob = classPrior[nodeLabel] * math.pow( estimatedProbabities[nodeLabel,0] , noOfZeroLabeledNeighbours ) * math.pow(estimatedProbabities[nodeLabel,1] ,len(neighbors)-noOfZeroLabeledNeighbours)
	
	# Converting to log for better precicion and avoiding overflow
	t = math.log( classPrior[nodeLabel] ) +  noOfZeroLabeledNeighbours*math.log( estimatedProbabities[nodeLabel,0] ) +  (len(neighbors)-noOfZeroLabeledNeighbours)*math.log( estimatedProbabities[nodeLabel,1] )
	prob = math.exp( t )
	#print prob
	return prob



def f2(currentLabelEstimates, neighbors, estimatedProbabities, classPrior):
	class0 = f1(0,currentLabelEstimates, neighbors, estimatedProbabities, classPrior)
	class1 = f1(1,currentLabelEstimates, neighbors, estimatedProbabities, classPrior)
	denominator = class0 + class1
	class0 = class0/denominator
	class1 = class1/denominator

	if random.uniform(0,1) < class0:
		return 0
	else:
		return 1



# Note - estimated probability is actually estimated counts

def initializeUnknownLabelsForGibbsSampling(G,label,testLabels):
	currentLabelEstimates = dict(label)

	# Compute Parameters before making initial estimates
	classPrior, estimatedProbabities, classPriorCounts, estimatedCounts = computeInitialParameters(G,label,testLabels)

	# Assign initial labels to all test labels just using the priors and the estimated probability of edges.

	for node in testLabels:
		neighbors = G[node]

		#removing all the edges to labels in the test set for computing initial estimates. Original Graph is unaffected.
		newNeighbors = set(neighbors)
		for i in neighbors: 
			if i in testLabels:
				newNeighbors.remove(i)
		neighbors = set(newNeighbors)

		currentLabelEstimates[node] = f2(currentLabelEstimates, neighbors, estimatedProbabities, classPrior)

	#print "Initial Parameter Estimates:"
	#print "Training Labels:",classPriorCounts
	#print "Current Attr. Cor.:", computeCorrelation(computePairs(G,currentLabelEstimates))
	#print "Class Prior Probabilites:",classPrior
	#print "Label-Label Count across Edges:\n",estimatedCounts
	#print sum(sum(estimatedCounts))
	#print "Probability Estimates:\n",estimatedProbabities,"\n"

	return (classPrior,estimatedProbabities,currentLabelEstimates,classPriorCounts,estimatedCounts)



# Function to compute the Accuracy, Precision, Recall
# Input : Original Labels, test labels and the predicted labels
# Output : Accuracy, Precision, Recall
def computeAccuracy(label,testLabels,resultingLabels):
	counts = numpy.zeros([2,2])
	for i in testLabels:
		counts[ label[i], resultingLabels[i] ] += 1

	accuracy = (counts[0,0]+counts[1,1])/sum(sum(counts))
	precision = (counts[0,1]+counts[1,1])/sum(sum(counts))
	recall = (counts[1,0]+counts[1,1])/sum(sum(counts))
	return accuracy,precision,recall


## Gibbs Sampling

def gibbsSampling(edges,label,testLabels):
		
	## Step 2 of algo
	classPrior,estimatedProbabities,currentLabelEstimates,classPriorCounts,estimatedCounts = initializeUnknownLabelsForGibbsSampling(edges,label,testLabels)

	nodeTraversalOrder = testLabels
	random.shuffle(nodeTraversalOrder)

	burnin = 100
	iteration = 500

	resultingLabels = {}
	for i in label:
		resultingLabels[i] = 0

	LabelDifferenceBetweenIterationsCounter = 0
	previousLabelDifferenceBetweenIterations = 0

	## Step 3 of algo
	#print "\nStart of Gibbs....\n"

	for i in range(iteration):
		
		LabelDifferenceBetweenIterations = 0
		for node in nodeTraversalOrder:
			#print "\nNode ",node
			#print "Before Attr. Cor.:", computeCorrelation(computePairs(edges,currentLabelEstimates))
			neighbors = edges[node]
			previousEstimate = currentLabelEstimates[node]
			currentLabelEstimates[node] = f2(currentLabelEstimates, neighbors, estimatedProbabities, classPrior)
		
		if i > burnin:
			for j in currentLabelEstimates:
				if currentLabelEstimates[j] == 1:
					resultingLabels[j] += 1

				temp = (resultingLabels[j] + 0.0)/(i - burnin) 
				temp = int(temp >= 0.5)
				if temp != label[j]:
					LabelDifferenceBetweenIterations += 1

		"""
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
		"""

		#if not i%10:
			#print "\n--------------------------------------------------\n" + "Iteration no : " +str(i)
			#print "Iteration no : " +str(i) + " -> LabelDifferenceBetweenIterations : " + str(LabelDifferenceBetweenIterations)	
			#print "Current Attr. Cor.:", computeCorrelation(computePairs(edges,currentLabelEstimates))
			#print classPriorCounts
			#print classPrior
			#print estimatedCounts
			#print sum(sum(estimatedCounts))
			#print estimatedProbabities
	

	for i in resultingLabels:
		resultingLabels[i] = (resultingLabels[i] + 0.0)/(iteration - burnin) 
		resultingLabels[i] = int(resultingLabels[i]  >= 0.5)
	
	ctr = 0
	for i in label:
		if label[i] != resultingLabels[i]:
			ctr += 1


	#print "\nFinal Results:\nNo. of Labels Mismatched:",ctr

	accuracy,precision,recall = computeAccuracy(label,testLabels,resultingLabels)

	#print "Accuracy:\n",accuracy
	#print "% = ",accp
	#print "Ground Truth:",computeLabelCounts(label,testLabels)
	#print "Predicted Labels:",computeLabelCounts(resultingLabels,testLabels)

	return (accuracy,precision,recall,estimatedProbabities)

