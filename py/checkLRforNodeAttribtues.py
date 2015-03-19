import sys, math
from sets import Set
from collections import Counter
import random
import numpy
from multiprocessing import Pool
import gc
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.metrics import accuracy_score


basePath = '/homes/pgurumur/jen/noise/data/'
#school = "facebook"
school = "school074"
schoolLabel = "_label0"

# graph
edges = {}
label = {}
nodeAttributes = {}

#f_in = open(basePath + '../data/polblogs-nodes.txt')
f_in = open(basePath + '../data/' + school + schoolLabel +'-nodes.txt')

# no need for first line...Skipping the header
junk_ = f_in.readline()

for line in f_in:
	fields = line.strip().split()
	label[int(float(fields[0]))] = int(float(fields[1]))
	edges[int(float(fields[0]))] = set([])

f_in.close()

f_in = open(basePath + '../data/' + school +'.attr')

for line in f_in:
	fields = line.strip().split("::")
	index = int(float(fields[0]))
	nodeAttributes[ index ] = []
	nodeAttributes[ index ].append( int(fields[1]) )
	nodeAttributes[ index ].append( int(fields[2]) )

f_in.close()



# Function to compute the Accuracy, Precision, Recall
# Input : test labels and the predicted labels
# Output : Accuracy, Precision, Recall
def computeAccuracy(testLabels,resultingLabels):
	counts = numpy.zeros([2,2])
	for i in range(len(testLabels)):
		counts[ testLabels[i], resultingLabels[i] ] += 1

	print counts
	accuracy = (counts[0,0]+counts[1,1] + 0.0)/sum(sum(counts))

	precision = 0.0
	recall = 0.0
	if (counts[0,1]+counts[1,1]) != 0:
		precision = counts[1,1] /(counts[0,1]+counts[1,1])
	if (counts[1,0]+counts[1,1]) != 0:
		recall = counts[1,1]  / (counts[1,0]+counts[1,1])
	return accuracy,precision,recall


# ListOfObject can be a list of numbers or a list of vectors or a list of matrices
def computeMeanAndStandardError(listOfObjects):
	mean = numpy.mean(listOfObjects,0)
	sd = numpy.std(listOfObjects,0)
	se = sd / math.sqrt(len(listOfObjects))
	median = numpy.median(listOfObjects,0)
	return (mean,sd,se,median)



def basicModel(originalLabels,trainingLabels,testingLabels,nodeAttributes,features=None):
	trainLabels = []
	trainFeatures = []
	testFeatures = []
	testLabels = []

	for i in originalLabels:
		l = [1] + nodeAttributes[i]
		if features is not None:
			l = l + features[i]
		if i in trainingLabels:
			trainLabels.append( originalLabels[i] )
			trainFeatures.append( l )
		elif i in testingLabels:
			testLabels.append( originalLabels[i] )
			testFeatures.append( l )

	
	logit = sm.Logit(trainLabels, trainFeatures)	 
	# fit the model
	result = logit.fit()
	#print result.summary()
	print result.params
	
	predicted = result.predict(testFeatures)
	resultingLabels = (predicted > threshold).astype(int)
	accuracy,precision,recall = computeAccuracy(testLabels,resultingLabels)
	print accuracy

	#return result.params,accuracy
	

	clf = linear_model.LogisticRegression()
	clf.fit(trainFeatures, trainLabels)
	pred = clf.predict(testFeatures)
	#accuracy = accuracy_score(testLabels,pred)
	#print accuracy
	#print clf.get_params()
	#print clf.coef_[0]
	#print clf.intercept_

	accuracy,precision,recall = computeAccuracy(testLabels,pred)
	print "Accuracy:",accuracy
	print "Precision:",precision
	print "Recall:",recall

	return accuracy,precision,recall 
	


noofProcesses = 7
noOfTimeToRunGibbsSampling = 25
testInputSize = 0.3

threshold = 0.5
identifier = ""
#identifier = "politics"

arg1 = sys.argv[1]
trainingSizeList = [ float(arg1) ]

"""
useSexAsLabel = True
if useSexAsLabel == True:
	identifier = "sex"
	print "Using 'Sex' as prediction label"
	index = 1
	for i in label:
		temp = label[i]
		label[i] = nodeAttributes[i][ index ]
		nodeAttributes[i][ index ] = temp
"""

"""
#Read the testLabels from the files to make it constant across runs
testLabelsList = []
testSize = 1-trainingSizeList[0]
f = open(basePath + "RandomTestLabelsForIterations/" + str(testSize) + "_testLabels.txt")
tLL = f.readlines()
f.close

for line in tLL:
	t = line.strip().split(',')
	testLabelsList.append([int(i) for i in t])
"""

outputTofile = []

for trainingSize in trainingSizeList:

	print "\n\n\n\n\ntrainingSize:",trainingSize
				
	testSize = testInputSize
	noOfLabelsToMask = int(testSize*len(label))
	print "testLabels Size:",noOfLabelsToMask

	b1 = []
	b2 = []
	b3 = []
	b4 = []

	for i in range(1):
		print "\nRepetition No.:",i+1

		# Uncomment the first line to generate random testLables for each iteration
		# Uncomment the second line to read the generated random testLables for each iteration. Based on Jen's suggestion to keep the testLabels constant across iterations.
		testLabels = random.sample(label,noOfLabelsToMask)
		#testLabels = testLabelsList[i]
		
		#print "Start test:",len(testLabels)
		trainingLabels_temp = [i for i in label if i not in testLabels]
		trainingLabels = random.sample(trainingLabels_temp, int(len(trainingLabels_temp)*trainingSize) )
		#print "Start trainLabels:",len(trainingLabels)
		print "\nJust NodeFeatures:"
		a,p,r = basicModel(label,trainingLabels,testLabels,nodeAttributes)
		b1.append((a,p,r))

		
	op = []
	op.append(str(trainingSize))

	a1,p1,r1 = zip(*b1)	
	meanAccuracy,sd,se,medianAccuracy = computeMeanAndStandardError(a1)
	meanPrecision,useless1,useless2,medianPrecision = computeMeanAndStandardError(p1)
	meanRecall,useless1,useless2,medianRecall = computeMeanAndStandardError(r1)
	op.extend( [ str(round(meanAccuracy,4)) , str(round(sd,4)) , str(round(se,4)) , str(round(meanPrecision,4)) , str(round(meanRecall,4))])

	outputTofile.append(op)



fileName = "mleModels_" + identifier + ".txt"
path = basePath + '../results/' 
f_out = open(path+fileName,'a')

header = ['trainingSize','b1_accuracy','b1_sd','b1_se','b1_precision','b1_recall']
f_out.write("\t".join(header)  + "\n\n")

for otf in outputTofile:
	f_out.write("\t".join(otf)  + "\n")

f_out.close()