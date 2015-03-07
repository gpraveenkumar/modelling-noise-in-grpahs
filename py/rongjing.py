import sys, math
from sets import Set
from collections import Counter
import random
import numpy
from multiprocessing import Pool
import gc
#import scipy.optimize
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

#print nodeAttributes



def sigmoid(Z):
	g = 1.0 / (1.0 + numpy.exp(-Z))
	return g
"""


def lrCostFunction(theta,X,y):

	m = y.shape
	J = 0

	h = sigmoid( numpy.dot(X,theta) )
	t1 = numpy.dot( numpy.transpose(y),numpy.log(h) ) + numpy.dot( (numpy.transpose(1-y)),numpy.log(1-h) )
	J = -t1/m

	return J



def lrGradient(theta,X,y):

	#Set the value of Aplha. Because python fmin_bfgs does not do it automatically and hence the algorithm does not converge.
	alpha = 0.01
	m = y.shape
	grad = numpy.zeros( theta.shape )

	h = sigmoid( numpy.dot(X,theta) )

	beta = h - y;
	delta = numpy.dot( numpy.transpose(X),beta );

	grad = delta / m

	grad = grad*alpha

	return grad



def lrTrain(X,y):

	m,n = X.shape

	theta0 = numpy.zeros(n)
	#theta = scipy.optimize.fmin_bfgs(lrCostFunction, theta0, fprime=lrGradient, args=(X,y)) 
	theta = scipy.optimize.fmin_bfgs(lrCostFunction, theta0, fprime=lrGradient, args=(X,y)) 
	print "theta",theta
"""

"""
# Test data to test logisitc regression

f_in = open(basePath + '../data/' + 'binary.csv')

trainFeatures = []
trainLabels = []

# no need for first line...Skipping the header
junk_ = f_in.readline()
for line in f_in:
	line = line.strip().split(',')
	trainLabels.append(int(line[0]))
	l = []
	l.append(1)
	l.append(int(line[1]))
	l.append(float(line[2]))
	l.append(int(line[3]))
	trainFeatures.append(l)

f_in.close()

#print trainLabels
#print trainFeatures

X = numpy.array(trainFeatures)
y = numpy.array(trainLabels)
print X
print y

from time import time
t = time()
lrTrain(X,y)
print round(time()-t,3)
"""



def independentModel(originalLabels,nodeAttributes,trainingLabels):
	trainLabels = []
	trainFeatures = []

	for i in trainingLabels:
		trainLabels.append( originalLabels[i] )
		#print nodeAttributes[i]
		l = [1] + nodeAttributes[i]
		#print l
		trainFeatures.append( l )

	logit = sm.Logit(trainLabels, trainFeatures)
	 
	# fit the model

	result = logit.fit()

	#print result.summary()
	print result.params
	return result.params


"""
def logLinearCostFunction(theta,G,label,nodeAttributes):


	logLikekyhood = 0

	for node in G:

		neighbors = G[node]
		noOfZeroLabeledNeighbours = 0
		for nei in neighbors:
			if currentLabelEstimates[nei] == 0:
				noOfZeroLabeledNeighbours += 1

		n1 = 0
		n2 = 0

		if label[node] == 0:
			n1 = noOfZeroLabeledNeighbours
			n2 = len(neighbor) - noOfZeroLabeledNeighbours
		else:
			n1 = len(neighbor) - noOfZeroLabeledNeighbours
			n2 = noOfZeroLabeledNeighbours	

		x = nodeAttributes[node]
		# may be there should be one theta value for every value of x[0] and x[1]
		phi_s = x[0]*theta[0] + x[1]*theta[1]
		phi_i = n1*theta[3] + n2*theta[4] 
		phi1 = phi_s + phi_i

		# Swapping n1 and n2 assuming as the node label has changed
		phi_i = n2*theta[3] + n1*theta[4] 
		phi2 = phi_s + phi_i

		logLikekyhood += phi1 - math.log( math.exp(phi1) + math.exp(phi2) )


	return logLikekyhood
"""



def neglogLinearCostFunction(theta,trainFeatures,trainLabels):

	logLikekyhood = 0

	for i in range(len(trainFeatures)):

		phi_s = trainFeatures[i][0]*theta[0] + trainFeatures[i][1]*theta[1]
		if trainLabels[i] == 0:
			phi_s = -phi_s

		phi_1 = n1*theta[3] + n2*theta[4]
		phi1 = phi_s + phi_i

		# Swapping n1 and n2 assuming as the node label has changed. Also, negating phi_s
		phi_i = n2*theta[3] + n1*theta[4] 
		phi2 = -phi_s + phi_i

		logLikekyhood += phi1 - math.log( math.exp(phi1) + math.exp(phi2) )

	neglogLikekyhood = -logLikekyhood
	return neglogLikekyhood



def logLinearGradient(theta,trainFeatures,trainLabels):

	#Set the value of Aplha. Because python fmin_bfgs does not do it automatically and hence the algorithm does not converge.
	alpha = 0.01
	m = y.shape
	grad = numpy.zeros( theta.shape )

	h = sigmoid( numpy.dot(X,theta) )

	beta = h - y;
	delta = numpy.dot( numpy.transpose(X),beta );

	grad = delta / m

	grad = grad*alpha

	return grad


# This function is essentially same as the "computeInitialParameters" in p_noB_gibbs.py
def MPLE(G,label,testLabels,nodeAttributes,mleParameters):

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

	estimatedProbabities[:,0] = (estimatedCounts[:,0] + 1) / (sum(estimatedCounts[:,0]) + 2)
	estimatedProbabities[:,1] = (estimatedCounts[:,1] + 1) / (sum(estimatedCounts[:,1]) + 2)
	#return (classPrior,estimatedProbabities,classPriorCounts,estimatedCounts)

	edgeCliqueCounts = {} 
	edgeCliqueCounts["match"] = estimatedCounts[0,0] + estimatedCounts[1,1] 
	edgeCliqueCounts["otherwise"] = estimatedCounts[0,1] + estimatedCounts[1,0]

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
			
		if len(newNeighbor) > 0:
			onlyTrainingG[ id ] = newNeighbor


	trainLabels = []
	trainFeatures = []

	for node,neighbors in G.iteritems():

		#neighbors = G[node]
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
	#print result.summary()
	print result.params
	return result.params	

	#theta0 = numpy.zeros(4)
	#theta = scipy.optimize.fmin_bfgs(logLinearCostFunction, theta0, fprime=logLinearGradient, args=(X,y)) 
	#print "theta",theta


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


def computeProgagationUpperBound(onlyTrainingG,label,L,mpleParameters):

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
			ki_list[node] = 0
			continue

		noOfZeroLabeledNeighbours = 0
		for neighbor in neighbors:
			if label[neighbor] == 0:
				noOfZeroLabeledNeighbours += 1

		maxforNode = []
		for neighbor in neighbors:

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



def computeMu(neighbors,label,nodeAttributes,mleParameters,mpleParameters,lamda):
	noOfZeroLabeledNeighbours = 0
	for neighbor in neighbors:
		if label[neighbor] == 0:
			noOfZeroLabeledNeighbours += 1

	#n1 = 0
	#n2 = 0
	n3 = 0
	n4 = 0

	#n1 = noOfZeroLabeledNeighbours
	#n2 = len(neighbors) - noOfZeroLabeledNeighbours
	
	n3 = len(neighbors) - noOfZeroLabeledNeighbours
	n4 = noOfZeroLabeledNeighbours

	# Logistic Regression Calculate the probability of y=1

	p_mple = mpleParameters[0] + mpleParameters[1]*nodeAttributes[0] + mpleParameters[2]*nodeAttributes[1] + mpleParameters[3]*n3 + mpleParameters[4]*n4
	p_mple = sigmoid(p_mple)
	p_mle = mleParameters[0] + mleParameters[1]*nodeAttributes[0] + mleParameters[2]*nodeAttributes[1]
	p_mle = sigmoid(p_mle)

	mu_1 = lamda*p_mple + (1-lamda)*p_mle

	return mu_1



def computeInitialEstimate(nodeAttributes,onlyTestG,finalTestLabels_test,label,L,mleParameters,mpleParameters,t,k0,ki_list):

	newOnlyTestG = onlyTestG
	#newOnlyTestG = {}

	"""
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
	"""

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

		mu_1 = computeMu(neighbors,label,nodeAttributes[node],mleParameters,mpleParameters,lamda)
		mu[node] = [1-mu_1,mu_1]

		t = 0
		if mu_1	> 0.5:
			t = 1
		currentLabelEstimates[node] = t

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
			#print "Before Attr. Cor.:", computeCorrelation(computePairs(edges,currentLabelEstimates))
			neighbors = G[node]
			##previousEstimate = currentLabelEstimates[node]
			#currentLabelEstimates[node] = f2(currentLabelEstimates, neighbors, estimatedProbabities, classPrior)

			##currentLabelEstimates[node] = f2(currentLabelEstimates, neighbors, estimatedProbabities, classPrior)

			mu_1 = computeMu(neighbors,label,nodeAttributes[node],mleParameters,mpleParameters,lamda_list[node])
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

	return (accuracy,precision,recall,squaredLoss)



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



noofProcesses = 7

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


	for i in range(1):
		print "\nRepetition No.:",i+1

		# Uncomment the first line to generate random testLables for each iteration
		# Uncomment the second line to read the generated random testLables for each iteration. Based on Jen's suggestion to keep the testLabels constant across iterations.

		#testLabels = random.sample(originalLabels,noOfLabelsToMask)
		testLabels = testLabelsList[i]

		trainingLabels = [i for i in originalLabels if i not in testLabels]

		mleParameters = independentModel(originalLabels,nodeAttributes,trainingLabels)
		mpleParameters = MPLE(originalGraph,originalLabels,testLabels,nodeAttributes,mleParameters)


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

				for labeledProportion in numpy.arange(0.0,1.0,0.1):

					lSize = int(labeledProportion*len(trainingLabels))
					L = random.sample(trainingLabels,lSize)
					finalTrainingLabels_test = [node for node in trainingLabels if node not in L]

					ki_list = computeProgagationUpperBound(onlyTrainingG,originalLabels,L,mpleParameters)

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

					mu,currentLabelEstimates,lamda_list = computeInitialEstimate(nodeAttributes,onlyTrainingG_test,finalTrainingLabels_test,originalLabels,L,mleParameters,mpleParameters,t,k0,ki_list)

					#accuracy, precision, recall, squaredLoss = gibbsSampling(onlyTrainingG_test,originalLabels,finalTrainingLabels_test,nodeAttributes,currentLabelEstimates,mleParameters,mpleParameters,lamda_list)
				
					arg_t = [onlyTrainingG_test,originalLabels,finalTrainingLabels_test,nodeAttributes,currentLabelEstimates,mleParameters,mpleParameters,lamda_list]
					
					arguments = []
					for i in range(10):
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
					

				if error < lowestError:
					lowestError = error
					tau_hat = t
					k0_hat = k0

		print lowestError
		print t
		print k0
		

		"""
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
		"""