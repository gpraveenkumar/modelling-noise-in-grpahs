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


"""
def sigmoid(Z):
	g = 1.0 / (1.0 + numpy.exp(-Z))
	return g



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


trainLabels = []
trainFeatures = []

for i in originalLabels:
	trainLabels.append( originalLabels[i] )
	#print nodeAttributes[i]
	l = [1] + nodeAttributes[i]
	print l
	trainFeatures.append( l )

logit = sm.Logit(trainLabels, trainFeatures)
 
# fit the model

result = logit.fit()

print result.summary()
print result.params
