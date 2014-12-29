basePath = '/homes/pgurumur/jen/noise/py/'
school = "facebook"
schoolLabel = "label0"

nodes = []

f_in = open(basePath + '../data/' + school + '_' + schoolLabel +'-nodes.txt')
junk_ = f_in.readline()

for line in f_in:
	line = line.strip().split()
	nodes.append(line[0])

f_in.close()

print len(nodes)

f_in = open(basePath + '../data/' + school + '_' + schoolLabel +'-edges.txt')
junk_ = f_in.readline()

ctr = 0

for line in f_in:
	line = line.strip().split()
	if line[0] not in nodes:
		print line[0]
	if line[1] not in nodes:
		print line[1]
	ctr += 1

f_in.close()

print ctr