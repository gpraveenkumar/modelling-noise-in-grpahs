import random

basePath = '/homes/pgurumur/jen/noise/py/'
school = "school074"
schoolLabel = "label0"

directoryToWriteFile = "RandomTestLabelsForIterations/"

f_in = open(basePath + '../data/' + school + '_' + schoolLabel +'-nodes.txt')

# no need for first line...Skipping the header
# header = 1 implies that the file has header
# header = 0 implies that the file has header
header = 1

l = f_in.readlines()
f_in.close()


# noofLabels = no of lines in the file - 1(if header is present). This is how the nodes.txt files is defined/created.
# noOfLabels = noOfNodes
noOfLabels = len(l) - header

print "No. of Labels",noOfLabels

# range(3) = [0,1,2]

originalLabels = range(noOfLabels)

for trainingSize in [0.05,0.1,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.7,0.8,0.9]:
	testSize = 1-trainingSize
	noOfLabelsToMask = int(testSize*noOfLabels)
	print trainingSize,noOfLabelsToMask
	
	testLabelsList = []
	for i in range(200):
		testLabels = random.sample(originalLabels,noOfLabelsToMask)
		testLabelsList.append([str(i) for i in testLabels])

	f = open(basePath + directoryToWriteFile + str(testSize) + "_testLabels.txt",'w')
	f.write('\n'.join(','.join(i) for i in testLabelsList)) 
	f.close()
