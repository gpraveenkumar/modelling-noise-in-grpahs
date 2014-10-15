import numpy
from collections import Counter

path = "../data/"
schoolNo = 74
labelNo = -3
labelCutoff = 1

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
id_label_mapping = {}

ctr = 0
for i in startId:
	id_AID_mapping[i] = ctr
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

"""
t = Counter()
for x in id_label_mapping:
	t[id_label_mapping[x]] += 1
print t
"""

