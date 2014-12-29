basePath = '/homes/pgurumur/jen/noise/results/'

fileName = 'school074-label0_run2_flipLabelResults'
fileName = 'polblogs-label0_rewireEdgesResults'


result = {}
trainingSize_index_map = {}
trainingSize = [0.05,0.1,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.7,0.8,0.9]

i = 0
for l in trainingSize:
	trainingSize_index_map[str(l)] = i
	i += 1
print trainingSize_index_map


index = 3
indexName = "Accuracy"

f = open(basePath + fileName + '.txt')

# Throw away the header line, but use it to count the number of lines in each line in that file.
line = f.readline()
line = line.strip().split('\t')
print line
count = len(line)

ctr = 0
for i in line:
	print ctr,i
	ctr += 1

for line in f:
	line = line.strip().split('\t')
	
	if len(line) < count:
		continue

	if line[0] not in result:
		result[ line[0] ] = [0]*len(trainingSize)
    
	result[ line[0] ][ trainingSize_index_map[ line[1] ] ] = line[index]

f.close()






orderToOutput = sorted(result.keys())
orderToOutput.remove("100perc_10repeat")
orderToOutput.remove("100perc_2repeat")
orderToOutput.remove("100perc_5repeat")
orderToOutput.remove("original")

orderToOutput.append("100perc_10repeat")
orderToOutput.append("100perc_2repeat")
orderToOutput.append("100perc_5repeat")
orderToOutput.insert(0,"original")



#keys = ['100perc_10repeat','100perc_2repeat','100perc_5repeat','60perc_10repeat','60perc_2repeat','60perc_5repeat','original']
keys = orderToOutput
tSize = ['0.1','0.2','0.5','0.9']

tSizeIndex = []

for i in tSize:
	tSizeIndex.append( trainingSize_index_map[i] )

print tSizeIndex


f = open(basePath + fileName + '_' + indexName + 'full.csv', 'w')
f.write(indexName + '\t' + '\t'.join(str(j) for i,j in enumerate(trainingSize) if i in tSizeIndex) + '\n')

f.write('\n')

for line in orderToOutput:
	if line not in keys:
		continue
	print result[line]
	toWrite = line + '\t' + '\t'.join(j for i,j in enumerate(result[line]) if i in tSizeIndex ) + '\n'
	f.write(toWrite)

f.close()