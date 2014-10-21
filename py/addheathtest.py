import sys, math
from sets import Set
from collections import Counter

f_in = open('../data/school074-parsed.txt')
# f_in = open('../Data/AddHealth074/school074-parsed.txt')

binary = True
directed = False

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
		label_0 = label_0 > 1
		label_1 = label_1 > 1
		label_2 = label_2 > 1

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

#print l
f = open('joel.txt','w')
f.write('\n'.join( ' '.join(i) for i in l))
f.close()

#class priors
t = Counter()
for x in labels_0:
	t[labels_0[x]] += 1
print t

# pairings for computing correlations
pairs_0 = []
pairs_1 = []
pairs_2 = []


# Compute the pairings
for id, neighbors in edges.iteritems():
	# praveen removed all of these
	if id not in labels_0 or id not in labels_1 or id not in labels_2:
		continue

	# cycle through the neighbors
	for neighbor in neighbors:
		if neighbor not in labels_0 or neighbor not in labels_1 or neighbor not in labels_2:
			continue

		pairs_0.append([labels_0[id], labels_0[neighbor]])
		pairs_1.append([labels_1[id], labels_1[neighbor]])
		pairs_2.append([labels_2[id], labels_2[neighbor]])

		if not directed:
			pairs_0.append([labels_0[neighbor], labels_0[id]])
			pairs_1.append([labels_1[neighbor], labels_1[id]])
			pairs_2.append([labels_2[neighbor], labels_2[id]])


print "length : " + str(len(pairs_0))

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


