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

	# This is code used for downsampling.
	# True = Downsampling
	# False = No Downampling
	if False:
		print "DownSampling...."
		downSamplePercentage = 0.25

		# We want to downsample only the training data.
		originalTrainingLabels = [i for i in label if i not in testLabels]

		noOfLabelsToMask = int(downSamplePercentage*len(originalTrainingLabels))
		originalTrainingLabelsToMask = random.sample(originalTrainingLabels,noOfLabelsToMask)

		print len(originalTrainingLabels)
		print len(originalTrainingLabelsToMask)
		print len(testLabels)

		# In this function, the only use of testLabels is to remove those nodes from the graph. 
		# So, I am adding the label to be revomed for downsampling to the testLabels set, so 
		# that the code below removes all those nodes from the graph and then calculates the probabily estiamtes.
		for i in originalTrainingLabelsToMask:
			testLabels.append(i)
		print len(testLabels)

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



# Function to set to the value of the intial parameters to fixed values without computing them
# Input : classPrior0,parameter_0given0,parameter_1given1
# Output : classPrior,estimatedProbabities,classPriorCounts(dummy parameter),estimatedCounts(dummy parameter)
# The (dummy parameter) are in place just for consistency with the function computeInitialParameters

def setInitialParameterValues(classPrior0,parameter_0given0,parameter_1given1):

	# dummy parameter; All set to 0
	classPriorCounts = {}
	classPriorCounts[0] = 0
	classPriorCounts[1] = 0
	estimatedCounts = numpy.zeros([2,2])
	
	classPrior = [0]*2
	classPrior[0] = classPrior0
	classPrior[1] = 1 - classPrior[0]
	
	# conditional probabilites
	estimatedProbabities = numpy.zeros([2,2])
	estimatedProbabities[0,0] = parameter_0given0 
	estimatedProbabities[1,0] = 1 - estimatedProbabities[0,0]
	estimatedProbabities[1,1] = parameter_1given1
	estimatedProbabities[0,1] = 1 - estimatedProbabities[1,1]

	return (classPrior,estimatedProbabities,classPriorCounts,estimatedCounts)



# Function to generate the class label based on a cutoff value by comparing it with a uniformly generated random number
# Input : Cutoff
# Output : Class Label (0 or 1)

def generateClassLabelBasedOnCutoff(cutoff):
	x = random.uniform(0,1)
	if x < cutoff:
		t = 0
	else:
		t = 1
	return t


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

	"""
	x = random.uniform(0,1)
	if x < class0:
		t = 0
	else:
		t = 1
	"""

	t = generateClassLabelBasedOnCutoff(class0)

	#print str(class0),str(class1)
	#print x
	#print str(t),str(int(class1/class0 > 1)),'\n'

	return t#int(class1/class0 > 1)


# This function is very similar to f2. But it is used to return the raw probability value for Maximum Entrophy Inference.
def f_maxEntInf(currentLabelEstimates, neighbors, estimatedProbabities, classPrior):
	class0 = f1(0,currentLabelEstimates, neighbors, estimatedProbabities, classPrior)
	class1 = f1(1,currentLabelEstimates, neighbors, estimatedProbabities, classPrior)
	denominator = class0 + class1
	class0 = class0/denominator
	class1 = class1/denominator

	return class0


"""
# This function is very similar to f2. It just return the label value based on prior value. This is used as smart Baseline.
def f_maxEntInf(currentLabelEstimates, neighbors, estimatedProbabities, classPrior):
	class0 = f1(0,currentLabelEstimates, neighbors, estimatedProbabities, classPrior)
	class1 = f1(1,currentLabelEstimates, neighbors, estimatedProbabities, classPrior)
	denominator = class0 + class1
	class0 = class0/denominator
	class1 = class1/denominator

	return class0
"""


# Note - estimated probability is actually estimated counts - update: not true anymore

def initializeUnknownLabelsForGibbsSampling(G,label,testLabels,parameters):
	currentLabelEstimates = dict(label)

	if parameters == None:
		# Compute Parameters before making initial estimates
		classPrior, estimatedProbabities, classPriorCounts, estimatedCounts = computeInitialParameters(G,label,testLabels)
	else:
		# Set the Parameters values determisnistically without computing based on te data to understand how the 
		# prediciton perrforms with a given parameter values
		classPrior0 = parameters[0]
		parameter_0given0 = parameters[1]
		parameter_1given1 = parameters[2]
		classPrior, estimatedProbabities, classPriorCounts, estimatedCounts = setInitialParameterValues(classPrior0,parameter_0given0,parameter_1given1)


	# Assign initial labels to all test labels just using the priors and the estimated probability of edges.

	for node in testLabels:
		neighbors = G[node]

		#removing all the edges to nodes/labels in the test set for computing initial estimates. Original Graph is unaffected.
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




# Note - estimated probability is actually estimated counts - update: not true anymore

def initializeUnknownLabelsForGibbsSampling_biasVariance(G,label,testLabels,parameters,testLabels_for_oneLabel):
	currentLabelEstimates = dict(label)
	currentLabelEstimates_oneLabel = dict(label)

	if parameters == None:
		# Compute Parameters before making initial estimates
		classPrior, estimatedProbabities, classPriorCounts, estimatedCounts = computeInitialParameters(G,label,testLabels_for_oneLabel)
	else:
		# Set the Parameters values determisnistically without computing based on te data to understand how the 
		# prediciton perrforms with a given parameter values
		classPrior0 = parameters[0]
		parameter_0given0 = parameters[1]
		parameter_1given1 = parameters[2]
		classPrior, estimatedProbabities, classPriorCounts, estimatedCounts = setInitialParameterValues(classPrior0,parameter_0given0,parameter_1given1)


	# Assign initial labels to all test labels just using the priors and the estimated probability of edges.
	# For One Label
	for node in testLabels:
		neighbors = G[node]

		currentLabelEstimates_oneLabel[node] = f2(label, neighbors, estimatedProbabities, classPrior)


	# Assign initial labels to all test labels just using the priors and the estimated probability of edges.
	# For All Labels - joint
	for node in testLabels:
		neighbors = G[node]
		
		#removing all the edges to nodes/labels in the test set for computing initial estimates. Original Graph is unaffected.
		newNeighbors = set()
		for i in neighbors: 
			if i not in testLabels:
				newNeighbors.add(i)
		#neighbors = set(newNeighbors)

		currentLabelEstimates[node] = f2(currentLabelEstimates, newNeighbors, estimatedProbabities, classPrior)

	#print "Initial Parameter Estimates:"
	#print "Training Labels:",classPriorCounts
	#print "Current Attr. Cor.:", computeCorrelation(computePairs(G,currentLabelEstimates))
	#print "Class Prior Probabilites:",classPrior
	#print "Label-Label Count across Edges:\n",estimatedCounts
	#print sum(sum(estimatedCounts))
	#print "Probability Estimates:\n",estimatedProbabities,"\n"

	return (classPrior,estimatedProbabities,currentLabelEstimates,classPriorCounts,estimatedCounts,currentLabelEstimates_oneLabel)




# Function to compute the Accuracy, Precision, Recall
# Input : Original Labels, test labels and the predicted labels
# Output : Accuracy, Precision, Recall
def computeAccuracy(label,testLabels,resultingLabels):
	counts = numpy.zeros([2,2])
	for i in testLabels:
		counts[ label[i], resultingLabels[i] ] += 1

	print counts
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


# Function to compute the sigmod or logistic
# Input : x
# Output : value between 0 and 1

def logistic(x):
  return 1 / (1 + math.exp(-x))


# Function to compute the logit
# Input : x
# Output : value between -Inf and Inf

def logit(x):
	# 100 seems to be a big enough number.
	if x == 0:
		return -100
	elif x == 1:
		return 100
	return numpy.log( x/(1-x) )



## Gibbs Sampling

def gibbsSampling(edges,label,testLabels,parameters):
		
	## Step 2 of algo
	classPrior,estimatedProbabities,currentLabelEstimates,classPriorCounts,estimatedCounts = initializeUnknownLabelsForGibbsSampling(edges,label,testLabels,parameters)

	nodeTraversalOrder = testLabels
	random.shuffle(nodeTraversalOrder)

	burnin = 100
	iteration = 500

	# if the maxEntInfFlag is set to true, the permform the Maximum Entropy Inference Correct from Joel's WWW 15.
	# Basically this is done so that the proporation of the labels in the unlabels set match the proportion of labels in the labels set.
	maxEntInfFlag = False


	# Although the resulting labels has the training Labels also set to zero, they are not used anywhere, so it doesnt matter what value they have.
	resultingLabels = {}
	for i in label:
		resultingLabels[i] = 0

	LabelDifferenceBetweenIterationsCounter = 0
	previousLabelDifferenceBetweenIterations = 0

	## Step 3 of algo
	#print "\nStart of Gibbs....\n"

	for i in range(iteration):
		
		LabelDifferenceBetweenIterations = 0

		if maxEntInfFlag:
			Z = []

		for node in nodeTraversalOrder:
			#print "\nNode ",node
			#print "Before Attr. Cor.:", computeCorrelation(computePairs(edges,currentLabelEstimates))
			neighbors = edges[node]
			previousEstimate = currentLabelEstimates[node]
			#currentLabelEstimates[node] = f2(currentLabelEstimates, neighbors, estimatedProbabities, classPrior)

			if maxEntInfFlag:
				currentLabelEstimates[node] = f_maxEntInf(currentLabelEstimates, neighbors, estimatedProbabities, classPrior)
				Z.append( currentLabelEstimates[node] )
			else:
				currentLabelEstimates[node] = f2(currentLabelEstimates, neighbors, estimatedProbabities, classPrior)
		
		# The algorithm as described in Joel's Paper.
		if maxEntInfFlag:
			Z = sorted(Z)
			phi = len(Z) * classPrior[0]

			# We need to find the corresponding value in the Z array. So, the following trick to get a interger index.
			phi = int(math.floor(phi))

			for node in nodeTraversalOrder:
				#print currentLabelEstimates[node]
				cle = currentLabelEstimates[node]
				cle = logistic( logit(cle) - Z[phi] )

				# Assign them a label based on probability
				currentLabelEstimates[node] = generateClassLabelBasedOnCutoff(cle)		

		if i > burnin:
			for j in currentLabelEstimates:
				if currentLabelEstimates[j] == 1:
					resultingLabels[j] += 1

				temp = (resultingLabels[j] + 0.0)/(i - burnin) 
				temp = int(temp >= 0.5)
				if temp != label[j]:
					LabelDifferenceBetweenIterations += 1

		
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
		

		#if not i%10:
			#print "\n--------------------------------------------------\n" + "Iteration no : " +str(i)
			#print "Iteration no : " +str(i) + " -> LabelDifferenceBetweenIterations : " + str(LabelDifferenceBetweenIterations)	
			#print "Current Attr. Cor.:", computeCorrelation(computePairs(edges,currentLabelEstimates))
			#print classPriorCounts
			#print classPrior
			#print estimatedCounts
			#print sum(sum(estimatedCounts))
			#print estimatedProbabities
	
	# Will be used for computing Squared loss. Is a dictionary because of the 
	# second line in the for loop and for it to be consistent to resultingLabels
	resultingLabelsForSquaredLoss = {}

	for i in resultingLabels:
		resultingLabels[i] = (resultingLabels[i] + 0.0)/(iteration - burnin) 
		resultingLabelsForSquaredLoss[i] = resultingLabels[i]
		resultingLabels[i] = int(resultingLabels[i]  >= 0.5)
	
	ctr = 0
	for i in label:
		if label[i] != resultingLabels[i]:
			ctr += 1

	#print "\nFinal Results:\nNo. of Labels Mismatched:",ctr

	accuracy,precision,recall = computeAccuracy(label,testLabels,resultingLabels)

	# Compute Squared Loss with the averages of Gibbs sampling before assigning them a single value
	squaredLoss = computeSquaredLoss(label,testLabels,resultingLabelsForSquaredLoss)
	

	#print "Accuracy:\n",accuracy
	#print "% = ",accp
	#print "Ground Truth:",computeLabelCounts(label,testLabels)
	#print "Predicted Labels:",computeLabelCounts(resultingLabels,testLabels)

	print "No. of interation in which labels have not changed:",LabelDifferenceBetweenIterationsCounter

	return (accuracy,precision,recall,classPrior,estimatedProbabities,squaredLoss)



# Function to compute the smart baseline based on prior. This function mirrors gibbsSampling in both the Inputs and the Outputs.
# This was designed to be called in exp-nob_parallel.py when required by just replace this function name with that of gibbsSampling
# Input : edges,label,testLabels,parameters
# Output : (accuracy,precision,recall,classPrior,estimatedProbabities,squaredLoss)

def smartBaseline(edges,label,testLabels,parameters):
			
	classPrior, estimatedProbabities, classPriorCounts, estimatedCounts = computeInitialParameters(edges,label,testLabels)

	# Although the resulting labels has the training Labels also set to zero, they are not used anywhere, so it doesnt matter what value they have.
	resultingLabels = {}
	for i in label:
		resultingLabels[i] = 0
	
	# Predict the resulting class label of a node just based on the prior probabilities of the class.
	for i in testLabels:
		resultingLabels[i] = generateClassLabelBasedOnCutoff(classPrior[0])


	accuracy,precision,recall = computeAccuracy(label,testLabels,resultingLabels)

	# Compute Squared Loss with the averages of Gibbs sampling before assigning them a single value
	squaredLoss = computeSquaredLoss(label,testLabels,resultingLabels)
	
	return (accuracy,precision,recall,classPrior,estimatedProbabities,squaredLoss)



# Following are the implementations of different variations of Relaxation Labling algorithm of Sofus' 2003 paper.

# Function implements the relaxation labeling (RN). This function mirrors gibbsSampling in both the Inputs and the Outputs.
# This was designed to be called in exp-nob_parallel.py when required by just replace this function name with that of gibbsSampling
# Input : edges,label,testLabels,parameters
# Output : (accuracy,precision,recall,classPrior,estimatedProbabities,squaredLoss)

def RN(edges,label,testLabels,parameters):

	classPrior, estimatedProbabities, classPriorCounts, estimatedCounts = computeInitialParameters(edges,label,testLabels)

	# Although the resulting labels has the training Labels also set to zero, they are not used anywhere, so it doesnt matter what value they have.
	resultingLabels = {}
	for i in label:
		resultingLabels[i] = 0
	
	nodeTraversalOrder = testLabels
	random.shuffle(nodeTraversalOrder)

	# Predict the resulting class label of a node just based on the prior probabilities of the class if there are no 
	# neighbours with know labels, otherwise it is the majority label of the .
	for node in nodeTraversalOrder:
		neighbors = edges[node]

		noOfNeighboursInTestLabels = 0
		noOfZeroLabeledNeighbours = 0
		for i in neighbors:
			if i in testLabels:
				noOfNeighboursInTestLabels +=1
			elif label[i] == 0:
				noOfZeroLabeledNeighbours += 1

		if noOfNeighboursInTestLabels == len(neighbors):
			resultingLabels[node] = generateClassLabelBasedOnCutoff(classPrior[0])
		elif noOfZeroLabeledNeighbours > len(neighbors) - noOfZeroLabeledNeighbours:
			resultingLabels[node]


	accuracy,precision,recall = computeAccuracy(label,testLabels,resultingLabels)

	# Compute Squared Loss with the averages of Gibbs sampling before assigning them a single value
	squaredLoss = computeSquaredLoss(label,testLabels,resultingLabels)
	
	return (accuracy,precision,recall,classPrior,estimatedProbabities,squaredLoss)






## Gibbs Sampling for doing bias variance analysis

def gibbsSampling_biasVariance(edges,label,testLabels,parameters,originalLabels,noOfLabelsToMask):

	testLabels_for_oneLabel = random.sample(originalLabels,noOfLabelsToMask)

	# Randomly sample new training nodes
	# The way "computeInitialParameters" is written - it takes it test labels and find out the training labels by removing the ltest labels. So this is equivalent to sampling new test labels.
	
	# One-Label (currentLabelEstimates_oneLabel) and All Label Joint (currentLabelEstimates)
	classPrior,estimatedProbabities,currentLabelEstimates_initial,classPriorCounts,estimatedCounts, currentLabelEstimates_oneLabel = initializeUnknownLabelsForGibbsSampling_biasVariance(edges,label,testLabels,parameters,testLabels_for_oneLabel)

	nodeTraversalOrder = testLabels
	#random.shuffle(nodeTraversalOrder)

	burnin = 100
	iteration = 500

	# if the maxEntInfFlag is set to true, the permform the Maximum Entropy Inference Correct from Joel's WWW 15.
	# Basically this is done so that the proporation of the labels in the unlabels set match the proportion of labels in the labels set.
	maxEntInfFlag = False

	resultingLabelEstimates_differentRuns = []
	squaredLoss_differentRuns = []

	for times in range(5):

		random.shuffle(nodeTraversalOrder)
		
		currentLabelEstimates = dict(currentLabelEstimates_initial)

		# Although the resulting labels has the training Labels also set to zero, they are not used anywhere, so it doesnt matter what value they have.
		resultingLabels = {}
		for i in label:
			resultingLabels[i] = 0

		LabelDifferenceBetweenIterationsCounter = 0
		previousLabelDifferenceBetweenIterations = 0

		## Step 3 of algo
		#print "\nStart of Gibbs....\n"

		for i in range(iteration):
			
			LabelDifferenceBetweenIterations = 0

			if maxEntInfFlag:
				Z = []

			for node in nodeTraversalOrder:
				#print "\nNode ",node
				#print "Before Attr. Cor.:", computeCorrelation(computePairs(edges,currentLabelEstimates))
				neighbors = edges[node]
				previousEstimate = currentLabelEstimates[node]
				#currentLabelEstimates[node] = f2(currentLabelEstimates, neighbors, estimatedProbabities, classPrior)

				if maxEntInfFlag:
					currentLabelEstimates[node] = f_maxEntInf(currentLabelEstimates, neighbors, estimatedProbabities, classPrior)
					Z.append( currentLabelEstimates[node] )
				else:
					currentLabelEstimates[node] = f2(currentLabelEstimates, neighbors, estimatedProbabities, classPrior)
			
			# The algorithm as described in Joel's Paper.
			if maxEntInfFlag:
				Z = sorted(Z)
				phi = len(Z) * classPrior[0]

				# We need to find the corresponding value in the Z array. So, the following trick to get a interger index.
				phi = int(math.floor(phi))

				for node in nodeTraversalOrder:
					#print currentLabelEstimates[node]
					cle = currentLabelEstimates[node]
					cle = logistic( logit(cle) - Z[phi] )

					# Assign them a label based on probability
					currentLabelEstimates[node] = generateClassLabelBasedOnCutoff(cle)		

			if i > burnin:
				for j in currentLabelEstimates:
					if currentLabelEstimates[j] == 1:
						resultingLabels[j] += 1

					temp = (resultingLabels[j] + 0.0)/(i - burnin) 
					temp = int(temp >= 0.5)
					if temp != label[j]:
						LabelDifferenceBetweenIterations += 1

			
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
			

			#if not i%10:
				#print "\n--------------------------------------------------\n" + "Iteration no : " +str(i)
				#print "Iteration no : " +str(i) + " -> LabelDifferenceBetweenIterations : " + str(LabelDifferenceBetweenIterations)	
				#print "Current Attr. Cor.:", computeCorrelation(computePairs(edges,currentLabelEstimates))
				#print classPriorCounts
				#print classPrior
				#print estimatedCounts
				#print sum(sum(estimatedCounts))
				#print estimatedProbabities
		
		# Will be used for computing Squared loss. Is a dictionary because of the 
		# second line in the for loop and for it to be consistent to resultingLabels
		resultingLabelsForSquaredLoss = {}

		for i in resultingLabels:
			resultingLabels[i] = (resultingLabels[i] + 0.0)/(iteration - burnin) 
			resultingLabelsForSquaredLoss[i] = resultingLabels[i]


		# Compute Squared Loss with the averages of Gibbs sampling before assigning them a single value
		squaredLoss = computeSquaredLoss(label,testLabels,resultingLabelsForSquaredLoss)

		resultingLabelEstimates_differentRuns.append(resultingLabels)
		squaredLoss_differentRuns.append(squaredLoss)
	
		print "Times :",times
		print "No. of interation in which labels have not changed:",LabelDifferenceBetweenIterationsCounter

		print squaredLoss


	return (currentLabelEstimates_oneLabel,resultingLabelEstimates_differentRuns,squaredLoss_differentRuns)


	