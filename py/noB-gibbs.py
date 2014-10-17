import numpy
from collections import Counter
import random

path = "../data/"
schoolNo = 74
labelNo = -3
labelCutoff = 1
testSize = 0.3

#using the last label

input = "preProcessed-school0" + str(schoolNo) + ".txt"

f = open(path + input)
lines = f.readlines()
f.close()

#id_number_mapping = {}
#ctr = 1
startId = set()
allId = set()

#labelCheck = Counter()


for i in range(1,len(lines)):
	t = lines[i].strip().split(" ")

	#assert len(t)==14
	
	#labelCheck[int(t[-3])] += 1
	
	for j in range(0,len(t)-3):
		if t[j] == "NA":
			continue
		
		if j==0:
			startId.add(t[j])
		else:
			allId.add(t[j])		

		"""
		if t[j] not in id_number_mapping:
			id_number_mapping[ t[j] ] = ctr
			#temp.append(str(ctr))
			ctr += 1
		"""
			
			
		#else:
			#temp.append(str(id_number_mapping[ t[j] ]))

"""
print(len(lines)-1)
print(ctr-1)
print(len(startId))
print(len(allId))
print(len(allId - startId))
print(len(startId - allId))
"""
#print(startId - allId)
#print(allId - startId)
#print labelCheck

n = len(startId)
data = numpy.zeros([n,n])

id_AID_mapping = {}
AID_id_mapping = {}
id_label_mapping = {}

ctr = 0
for i in startId:
	id_AID_mapping[i] = ctr
	AID_id_mapping[ctr] = i
	ctr += 1


for i in range(1,len(lines)):
	t = lines[i].strip().split(" ")

	#if t[0] not in startId:
	#	print "Error!"

	node1 = id_AID_mapping[t[0]]
	id_label_mapping[ node1 ] = int(t[labelNo]) >= labelCutoff

	for j in range(1,len(t)-3):
		if t[j] == "NA" or t[j] not in startId:
			continue

		node2 = id_AID_mapping[ t[j] ]

		data[node1,node2] = 1
		data[node2,node1] = 1

#print id_label_mapping
#print data


# Delete nodes with no neighbours
nodesToDelete = []
for i in range(n):
	if sum(data[i,:])==0:
		nodesToDelete.append(i)

data = numpy.delete(data,nodesToDelete,axis=0)
data = numpy.delete(data,nodesToDelete,axis=1)

n = n - len(nodesToDelete)
#print n
#print data.shape
#print nodesToDelete

"""
t = Counter()
for x in id_label_mapping:
	t[id_label_mapping[x]] += 1
print t
"""

## noB starts here

# making a fraction(=testsize) of labels
noOfLabelsToMask = int(testSize*len(id_label_mapping))
#print noOfLabelsToMask
labelsToMask = random.sample(xrange(n),noOfLabelsToMask)

testLabels = id_label_mapping

for i in labelsToMask:
	testLabels[i] = -1;

#class priors
t = Counter()
for x in testLabels:
	t[testLabels[x]] += 1
print t

classPrior = [0]*2
classPrior[0] = t[0] / (t[0] + t[1] + 0.0)
classPrior[1] = 1 - classPrior[0]

print classPrior


estimatedTestLabelsProbabilites = numpy.zeros([n,2])
estimatedTestLabels = {}
currentTestLabels = testLabels

for i in labelsToMask:
	estimatedTestLabelsProbabilites[i,0] = classPrior[0]
	estimatedTestLabelsProbabilites[i,1] = classPrior[1]

	unif = random.uniform(0,1)
	if unif < classPrior[0]:
		estimatedTestLabels[i] = 0
		currentTestLabels[i] = 0
	else:
		estimatedTestLabels[i] = 1
		currentTestLabels[i] = 1

"""
t = Counter()
for x in estimatedTestLabels:
	t[estimatedTestLabels[x]] += 1
print t
"""



## Gibbs Sampling

## Step 2 of algo
nodeTraversalOrder = labelsToMask
random.shuffle(nodeTraversalOrder)

## Step 3 of algo

for node in nodeTraversalOrder:

	neighbours = [i for i in range(n) if data[i,node]==1 ]
	#print AID_id_mapping[node]
	n0 = 0
	for i in neighbours:
		if currentTestLabels[i] == 0:
			n0 += 1
	n1= len(neighbours) - n0
	#print(neighbours)
	#print(str(n0)+" "+str(n1))

	p0 = (n0 + 0.0 + 1)/(len(neighbours) + 2)
	p1 = (n1 + 0.0 + 1)/(len(neighbours) + 2)
	#print(str(p0)+" "+str(p1))

