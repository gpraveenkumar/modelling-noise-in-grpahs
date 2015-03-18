import sys, math
from sets import Set
from collections import Counter
import random
import numpy

path = "../data/"
school = "school074"
f_in = open(path + school + '-parsed.txt')

binary = True
directed = False

#testSize = 0.3

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


# pairings for computing correlations
pairs_0 = []
pairs_1 = []
pairs_2 = []

edges1 = dict(edges)

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
edges1 = dict(edges)
for id, neighbors in edges1.iteritems():
	if len(neighbors) == 0:
		del edges[id]
		del labels_0[id]
		del labels_1[id]
		del labels_2[id]

del edges1


# Compute the pairings
for id, neighbors in edges.iteritems():
	# cycle through the neighbors
	for neighbor in neighbors:

		pairs_0.append([labels_0[id], labels_0[neighbor]])
		pairs_1.append([labels_1[id], labels_1[neighbor]])
		pairs_2.append([labels_2[id], labels_2[neighbor]])

		# This code is not necessary. Joel added it by mistake.
		"""
		if not directed:
			pairs_0.append([labels_0[neighbor], labels_0[id]])
			pairs_1.append([labels_1[neighbor], labels_1[id]])
			pairs_2.append([labels_2[neighbor], labels_2[id]])
		"""



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


# Function takes in a map of nodeId, label and return a distribution of labels
def getDistributionCount(label):
	c = Counter()

	for i in label:
		c[ label[i] ] += 1

	print c

getDistributionCount( labels_0 )
getDistributionCount( labels_1 )
getDistributionCount( labels_2 )

label = labels_0
labelName ="label0"


# Write the nodeId and Label to a file - space separated
# Next, write edge pairs to a file
f_out = open(path + school + '_' + labelName + '-nodes.txt','w')
f_out1 = open(path + school + '_' + labelName + '-edges.txt','w')
f_out2 = open(path + school + '.attr','w')

f_out.write("Id value\n")
f_out1.write("Source Target\n")

for id, neighbors in edges.iteritems():
	f_out.write( str(id) + " " + str(label[id]) + "\n" )
	f_out2.write( str(id) + "::" + str(labels_1[id]) + "::" + str(labels_2[id]) + "\n" )
	for neighbor in neighbors:
		f_out1.write( str(id) + " " + str(neighbor) + "\n" )

f_out.close()
f_out1.close()