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
school = "school074"
#school = "polblogs"
#school = "cora"
#school = "facebook"
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
# Output : sigmoid(-Z)
def sigmoid(Z):
	g = 1.0 / (1.0 + numpy.exp(-Z))
	return g



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
def MPLE(G,label,testLabels,nodeAttributes,mleParameters):

	# Construct a graph with only those nodes in the training Labels. This is requried for the training phase to learn the model parametes.
	onlyTrainingG = {}

	for id, neighbors in G.iteritems():
		if id in testLabels:
			continue

		newNeighbor = []
		# cycle through the neighbors
		for neighbor in neighbors:
			if neighbor in testLabels:
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
	testLabels = []

	for node,neighbors in onlyTestG_withTrainingNeighours.iteritems():

		n1,n2 = compute_phiFeatures(node,neighbors,label)	

		l = []
		l.append(1)
		for i in nodeAttributes[node]:
			l.append( i )

		l.append( n1 )
		l.append( n2 )

		testFeatures.append( l )
		testLabels.append( label[node] )
		
	predicted = result.predict(testFeatures)
	resultingLabels = (predicted >= threshold).astype(int)
	accuracy,precision,recall = computeAccuracy1(testLabels,resultingLabels)
	print accuracy
	# Compute Initial Estimate -- Completed

	#Gibbs Sampling part
	mpleParameters = result.params
	arg_t = [G,label,testLabels,nodeAttributes,resultingLabels,mpleParameters]
	
	arguments = []
	for i in range(noOfTimeToRunGibbsSampling):
		arguments.append(list(arg_t))

	pool = Pool(processes=noofProcesses)
	y = pool.map(func_star1, arguments)
	pool.close()
	pool.join()

	accuracy, precision, recall, squaredLoss = zip(*y)
	meanAccuracy,sd,se,uselessMedian = computeMeanAndStandardError(accuracy)
	meanPrecision,uselessSd,uselessSe,uselessMedian = computeMeanAndStandardError(precision)
	meanRecall,uselessSd,uselessSe,uselessMedian = computeMeanAndStandardError(recall)
	meanSquaredLoss,sd,se,uselessMedian = computeMeanAndStandardError(squaredLoss)

	return result.params,meanAccuracy



# Function the computes the number of label matches with its neighbours for the MPLE model.
def getCounts(label1,label2,n1,n2):

	n3 = 0
	n4 = 0

	if label1 == 1 and label2 == 1:
		n3 = n1 + 1  
	elif label1 == 1 and label2 == 0:
		n4 = n2 + 1  
	elif label1 == 0 and label2 == 1:
		n4 = n2 + 1 
	elif label1 == 0 and label2 == 0:
		n3 = n1 + 1

	return n3,n4



# Computes an upper bound on the propafgation error.
def computeProgagationUpperBound(onlyTrainingG,label,L,mpleParameters):

	# Note: L == TrainingLabels for conceptual clarity.
	
	newOnlyTrainingG = {}

	for node, neighbors in onlyTrainingG.iteritems():
		if node in L:
			continue

		newNeighbor = []
		# cycle through the neighbors
		for neighbor in neighbors:
			if neighbor not in L:
				continue
			else:
				newNeighbor.append( neighbor )
			
		newOnlyTrainingG[ node ] = newNeighbor

	ki_list = {}
	for node,neighbors in newOnlyTrainingG.iteritems():

		if len(neighbors) == 0:
			ki_list[node] = 1
			continue

		noOfZeroLabeledNeighbours = 0
		for neighbor in neighbors:
			if label[neighbor] == 0:
				noOfZeroLabeledNeighbours += 1

		maxforNode = []
		for neighbor in neighbors:

			# Not the -= sign
			if label[neighbor] == 0:
				noOfZeroLabeledNeighbours -= 1

			n1 = 0
			n2 = 0
			n3 = 0
			n4 = 0

			if label[node] == 0:
				n1 = noOfZeroLabeledNeighbours
				n2 = len(neighbors) - noOfZeroLabeledNeighbours
			else:
				n1 = len(neighbors) - noOfZeroLabeledNeighbours
				n2 = noOfZeroLabeledNeighbours	

			n3,n4 = getCounts(1,label[neighbor],n1,n2)
			phi1 =  n3*mpleParameters[3] + n4*mpleParameters[4]
			n3,n4 = getCounts(1,1-label[neighbor],n1,n2)
			phi0 = n3*mpleParameters[3] + n4*mpleParameters[4]
			maxforNode.append( 2*abs(phi1 - phi0) )

			n3,n4 = getCounts(0,label[neighbor],n1,n2)
			phi1 =  n3*mpleParameters[3] + n4*mpleParameters[4]
			n3,n4 = getCounts(0,1-label[neighbor],n1,n2)
			phi0 = n3*mpleParameters[3] + n4*mpleParameters[4]
			maxforNode.append( 2*abs(phi1 - phi0) )

		
		delta = max(maxforNode) 
		ki_list[node] =  delta/8 

	return ki_list



def computeMu(node,neighbors,label,nodeAttributes,mleParameters,mpleParameters,lamda):

	n3,n4 = compute_phiFeatures(node,neighbors,label)

	# Logistic Regression Calculate the probability of y=1

	p_mple = mpleParameters[0] + mpleParameters[1]*nodeAttributes[0] + mpleParameters[2]*nodeAttributes[1] + mpleParameters[3]*n3 + mpleParameters[4]*n4
	p_mple = sigmoid(p_mple)
	p_mle = mleParameters[0] + mleParameters[1]*nodeAttributes[0] + mleParameters[2]*nodeAttributes[1]
	p_mle = sigmoid(p_mle)

	mu_1 = lamda*p_mple + (1-lamda)*p_mle
	#mu_0 = lamda*(1-p_mple) + (1-lamda)*(1-p_mle)
	#print mu_0,mu_1,mu_0+mu_1

	return mu_1



# Compute initial label for the unlabled or the test nodes for rongjing's algorithm. This is used both during the training and the test phases.
def computeInitialEstimate(nodeAttributes,onlyTestG,L,finalTestLabels_test,label,mleParameters,mpleParameters,t,k0,ki_list):

	#newOnlyTestG = onlyTestG
	
	newOnlyTestG = {}
	
	for node, neighbors in onlyTestG.iteritems():
		if node in L:
			continue

		newNeighbor = []
		# cycle through the neighbors
		for neighbor in neighbors:
			if neighbor not in L:
				continue
			else:
				newNeighbor.append( neighbor )
			
		newOnlyTestG[ node ] = newNeighbor
	

	#print len(newOnlyTestG)

	mu = {}
	currentLabelEstimates = {}
	lamda_list = {}
	#print ki_list
	#for node,neighbors in newOnlyTestG.iteritems():
	#	print node
	#print len(newOnlyTestG)
	for node,neighbors in newOnlyTestG.iteritems():

		ki = ki_list[node]

		lamda = math.exp(-t*max(ki-k0,0))
		lamda_list[ node ] = lamda

		mu_1 = computeMu(node,neighbors,label,nodeAttributes[node],mleParameters,mpleParameters,lamda)
		mu[node] = [1-mu_1,mu_1]

		t = 0
		if mu_1	> 0.5:
			t = 1
		currentLabelEstimates[node] = t

	#print "Lamda list:",len(lamda_list)
	return mu,currentLabelEstimates,lamda_list




def gibbsSampling(G,label,testLabels,nodeAttributes,currentLabelEstimates,mleParameters,mpleParameters,lamda_list):		
	## Step 2 of algo

	nodeTraversalOrder = testLabels
	random.shuffle(nodeTraversalOrder)

	burnin = 100
	iteration = 500

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

		for node in nodeTraversalOrder:
			#print "\nNode ",node
			neighbors = G[node]
			
			mu_1 = computeMu(node,neighbors,label,nodeAttributes[node],mleParameters,mpleParameters,lamda_list[node])
			t = 0
			if mu_1	> 0.5:
				t = 1
			currentLabelEstimates[node] = t
		
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
	
	print "No. of interation in which labels have not changed:",LabelDifferenceBetweenIterationsCounter

	return (accuracy,precision,recall,squaredLoss)



# Function to calculate the labels with doing gibbs sampling for MPLE model

def compute_p_mple(node,neighbors,label,nodeAttributes,mpleParameters):
	
	n3,n4 = compute_phiFeatures(node,neighbors,label)

	# Logistic Regression by default calculates the probability of y=1

	p_mple = mpleParameters[0] + mpleParameters[1]*nodeAttributes[0] + mpleParameters[2]*nodeAttributes[1] + mpleParameters[3]*n3 + mpleParameters[4]*n4
	p_mple = sigmoid(p_mple)

	return p_mple



# Gibbs Sampling for MPLE
def gibbsSampling_MPLE(G,label,testLabels,nodeAttributes,currentLabelEstimates,mpleParameters):		
	## Step 2 of algo

	nodeTraversalOrder = testLabels
	random.shuffle(nodeTraversalOrder)

	burnin = 100
	iteration = 500

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

		for node in nodeTraversalOrder:
			#print "\nNode ",node
			neighbors = G[node]
			
			p_mple = compute_p_mple(node,neighbors,label,nodeAttributes[node],mpleParameters)
			t = 0
			if p_mple >= 0.5:
				t = 1
			currentLabelEstimates[node] = t
		
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
	
	print "No. of interation in which labels have not changed:",LabelDifferenceBetweenIterationsCounter

	return (accuracy,precision,recall,squaredLoss)



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



def func_star(a_b):
	"""Convert `f([1,2])` to `f(1,2)` call."""
	return gibbsSampling(*a_b)



def func_star1(a_b):
	"""Convert `f([1,2])` to `f(1,2)` call."""
	return gibbsSampling_MPLE(*a_b)






noofProcesses = 7
noOfTimeToRunGibbsSampling = 25


threshold = 0.5

arg1 = sys.argv[1]

trainingSizeList = [ float(arg1) ]


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

	print "\n\n\n\n\ntrainingSize:",trainingSize
				
	testSize = 1-trainingSize
	noOfLabelsToMask = int(testSize*len(originalLabels))
	print "testLabels Size:",noOfLabelsToMask

	a1 = []
	p1 = []
	r1 = []
	s1 = []
	initialEstimate = []

	i1 = []
	i2 = []

	for i in range(10):
		print "\nRepetition No.:",i+1

		# Uncomment the first line to generate random testLables for each iteration
		# Uncomment the second line to read the generated random testLables for each iteration. Based on Jen's suggestion to keep the testLabels constant across iterations.
		testLabels = random.sample(originalLabels,noOfLabelsToMask)
		#testLabels = testLabelsList[i]
		
		#print "Start test:",len(testLabels)
		trainingLabels = [i for i in originalLabels if i not in testLabels]
		#print "Start trainLabels:",len(trainingLabels)
		mleParameters,indepModel_accuracy = independentModel(originalLabels,nodeAttributes,trainingLabels)
		mpleParameters,mpleModel_accuracy = MPLE(originalGraph,originalLabels,testLabels,nodeAttributes,mleParameters)

		i1.append(indepModel_accuracy)
		i2.append(mpleModel_accuracy)
		
		onlyTrainingG = {}

		for node, neighbors in originalGraph.iteritems():
			if node in testLabels:
				continue

			newNeighbor = []
			# cycle through the neighbors
			for neighbor in neighbors:
				if neighbor in testLabels:
					continue
				else:
					newNeighbor.append( neighbor )
				
			if len(newNeighbor) > 0:
				onlyTrainingG[ node ] = newNeighbor

		#print len(trainingLabels)
		trainingLabels = [node for node in onlyTrainingG]
		#print len(onlyTrainingG)
		#print len(trainingLabels)

		kappas = []
		for labeledProportion in numpy.arange(0.0,1.0,0.1):
			print labeledProportion

			if labeledProportion == 0:
				continue

			lSize = int(labeledProportion*len(trainingLabels))
			L = random.sample(trainingLabels,lSize)
			#print len(trainingLabels) - lSize
			ki_list = computeProgagationUpperBound(onlyTrainingG,originalLabels,L,mpleParameters)
			#print len(ki_list)
			for node,ki in ki_list.iteritems():
				if ki != 0:
					kappas.append(ki)
		
		kappas.sort()

		
		kappa = []
		for j in numpy.arange(0.1,1,0.2):
			index = int(len(kappas)*j)
			kappa.append( kappas[index] )

		print kappa

		tau = [0.01, 0.1, 0.5, 1.0]
		
		#kappa = [0.1]
		#tau = [0.1]
		"""
		onlyTestG = {}

		for node, neighbors in originalGraph.iteritems():
			if node not in testLabels:
				continue

			newNeighbor = []
			# cycle through the neighbors
			for neighbor in neighbors:
				if neighbor not in testLabels:
					continue
				else:
					newNeighbor.append( neighbor )
				
			if len(newNeighbor) > 0:
				onlyTestG[ node ] = newNeighbor

		finalTestLabels = [node for node in onlyTestG]
		print len(finalTestLabels)
		"""

		lowestError = math.pow(10,9)
		tau_hat = 0
		k0_hat = 0
		
		for t in tau:
			for k0 in kappa:

				print "Tau:",t," k0:",k0
				error = 0
				#error = innerLoop_trainTestAlgo(trainingLabels,onlyTrainingG,nodeAttributes,mleParameters,mpleParameters,t,k0)

				for labeledProportion in numpy.arange(0.0,1.0,0.1):

					lSize = int(labeledProportion*len(trainingLabels))
					L = random.sample(trainingLabels,lSize)
					finalTrainingLabels_test = [node for node in trainingLabels if node not in L]

					ki_list = computeProgagationUpperBound(onlyTrainingG,originalLabels,L,mpleParameters)

					"""
					onlyTrainingG_test = {}
					for node, neighbors in onlyTrainingG.iteritems():
						if node in L:
							continue

						newNeighbor = []
						# cycle through the neighbors
						for neighbor in neighbors:
							if neighbor not in L:
								continue
							else:
								newNeighbor.append( neighbor )
							
						onlyTrainingG_test[ node ] = newNeighbor
					"""


					mu,currentLabelEstimates,lamda_list = computeInitialEstimate(nodeAttributes,onlyTrainingG,L,finalTrainingLabels_test,originalLabels,mleParameters,mpleParameters,t,k0,ki_list)

					arg_t = [onlyTrainingG,originalLabels,finalTrainingLabels_test,nodeAttributes,currentLabelEstimates,mleParameters,mpleParameters,lamda_list]
					
					arguments = []
					for i in range(noOfTimeToRunGibbsSampling):
						arguments.append(list(arg_t))

					pool = Pool(processes=noofProcesses)
					y = pool.map(func_star, arguments)
					pool.close()
					pool.join()

					accuracy, precision, recall, squaredLoss = zip(*y)
					meanAccuracy,sd,se,uselessMedian = computeMeanAndStandardError(accuracy)
					meanPrecision,uselessSd,uselessSe,uselessMedian = computeMeanAndStandardError(precision)
					meanRecall,uselessSd,uselessSe,uselessMedian = computeMeanAndStandardError(recall)
					meanSquaredLoss,sd,se,uselessMedian = computeMeanAndStandardError(squaredLoss)
					
					print "Classification Error:",1 - meanAccuracy
					error += 1 - meanAccuracy

				print "Error:",error

				if error < lowestError:
					lowestError = error
					tau_hat = t
					k0_hat = k0

		print lowestError
		print t
		print k0

		print "Training Complete....."

		"""
		onlyTestG = {}

		for node, neighbors in originalGraph.iteritems():
			if node not in testLabels:
				continue

			newNeighbor = []
			# cycle through the neighbors
			for neighbor in neighbors:
				if neighbor not in testLabels:
					continue
				else:
					newNeighbor.append( neighbor )
				
			if len(newNeighbor) > 0:
				onlyTestG[ node ] = newNeighbor

		finalTestLabels = [node for node in onlyTestG]
		print len(finalTestLabels)
		"""

		#error = innerLoop_trainTestAlgo(trainingLabels,originalGraph,nodeAttributes,mleParameters,mpleParameters,t,k0)
		#print "Final Error", error
		
		#lSize = int(labeledProportion*len(trainingLabels))
		#L = random.sample(trainingLabels,lSize)
		#originalTestLabels = [node for node in originalLabels if node not in trainLabels]

		#print "results:"
		trainingLabels = [i for i in originalLabels if i not in testLabels]
		#print len(trainingLabels)
		#print len(testLabels)
		#print len(originalLabels)
		#print len(originalGraph)
		ki_list = computeProgagationUpperBound(originalGraph,originalLabels,trainingLabels,mpleParameters)
		#print len(testLabels)
		#print len(ki_list)

		"""
		onlyTrainingG_test = {}
		for node, neighbors in onlyTrainingG.iteritems():
			if node in L:
				continue

			newNeighbor = []
			# cycle through the neighbors
			for neighbor in neighbors:
				if neighbor not in L:
					continue
				else:
					newNeighbor.append( neighbor )
				
			onlyTrainingG_test[ node ] = newNeighbor
		"""

		#print "here"
		mu,currentLabelEstimates,lamda_list = computeInitialEstimate(nodeAttributes,originalGraph,trainingLabels,testLabels,originalLabels,mleParameters,mpleParameters,t,k0,ki_list)
		""""
		print "********************************"
		print sorted(testLabels)
		print currentLabelEstimates
		print len(testLabels)
		print len(currentLabelEstimates)
		print "********************************"
		"""
		accuracy,precision,recall = computeAccuracy(originalLabels,testLabels,currentLabelEstimates)
		initialEstimate.append(accuracy)

		#accuracy, precision, recall, squaredLoss = gibbsSampling(originalGraph,originalLabels,testLabels,nodeAttributes,currentLabelEstimates,mleParameters,mpleParameters,lamda_list)
		arg_t = [originalGraph,originalLabels,testLabels,nodeAttributes,currentLabelEstimates,mleParameters,mpleParameters,lamda_list]
		
		arguments = []
		for i in range(noOfTimeToRunGibbsSampling):
			arguments.append(list(arg_t))

		pool = Pool(processes=noofProcesses)
		y = pool.map(func_star, arguments)
		pool.close()
		pool.join()

		accuracy, precision, recall, squaredLoss = zip(*y)
		meanAccuracy,sd,se,uselessMedian = computeMeanAndStandardError(accuracy)
		meanPrecision,uselessSd,uselessSe,uselessMedian = computeMeanAndStandardError(precision)
		meanRecall,uselessSd,uselessSe,uselessMedian = computeMeanAndStandardError(recall)
		meanSquaredLoss,sd,se,uselessMedian = computeMeanAndStandardError(squaredLoss)

		print "Classification Error:",1 - meanAccuracy
		a1.append(meanAccuracy)
		p1.append(meanPrecision)
		r1.append(meanRecall)
		s1.append(meanSquaredLoss)

		#Freeup space
		del arguments[:]
		gc.collect()


	i1_meanAccuracy,i1_sd,i1_se,i1_medianAccuracy = computeMeanAndStandardError(i1)
	i2_meanAccuracy,i2_sd,i2_se,i2_medianAccuracy = computeMeanAndStandardError(i2)

	initialEstimate_meanAccuracy,initialEstimate_sd,initialEstimate_se,initialEstimate_medianAccuracy = computeMeanAndStandardError(initialEstimate)

	meanAccuracy,sd,se,medianAccuracy = computeMeanAndStandardError(a1)
	meanPrecision,useless1,useless2,medianPrecision = computeMeanAndStandardError(p1)
	meanRecall,useless1,useless2,medianRecall = computeMeanAndStandardError(r1)
	meanSquaredLoss,useless1,useless2,useless3 = computeMeanAndStandardError(s1)

	f1 = 0
	# Precision and Recall are calculated w.r.t label 1. So if everything converges to label 0, both P and R will be 0
	if meanPrecision != 0 and meanRecall != 0:
		f1 = (2*meanPrecision*meanRecall)/(meanPrecision+meanRecall)

	print a1
	print "Prediction meanAccuracy",meanAccuracy
	print "Prediction SD:",sd
	print "Prediction SE:",se
	print "Prediction MeanPrecision:",meanPrecision
	print "Prediction MeanRecall:",meanRecall
	print "Prediction F1:",f1
	print s1
	print "MeanSquaredLoss:",meanSquaredLoss

	outputTofile = []
	header = ['trainingSize','meanSquaredLoss','meanAccuracy','sd','se','meanPrecision','meanRecall','f1','mle_meanAccuracy','mle_sd','mle_se','mple_meanAccuracy','mple_sd','mple_se','initialEstimate_meanAccuracy','initialEstimate_sd','initialEstimate_se']
	#outputTofile.append(header)

	outputTofile.append( [str(trainingSize), str(round(meanSquaredLoss,4)) , str(round(meanAccuracy,4)) , str(round(sd,4)) , str(round(se,4)) , str(round(meanPrecision,4)) , str(round(meanRecall,4)) , str(round(f1,4)) , str(round(i1_meanAccuracy,4)) , str(round(i1_sd,4)) , str(round(i1_se,4)) , str(round(i2_meanAccuracy,4)) , str(round(i2_sd,4)) , str(round(i2_se,4)) , str(round(initialEstimate_meanAccuracy,4)) , str(round(initialEstimate_sd,4)) , str(round(initialEstimate_se,4))])


	fileName = "algo.txt"
	path = basePath + '../results/Rongjing-' + school + '-' + schoolLabel + '-'
	f_out = open(path+fileName,'a')

	for otf in outputTofile:
		f_out.write("\t".join(otf)  + "\n")

	f_out.close()
				
