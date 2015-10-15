basePath = '/homes/pgurumur/jen/noise/results/'

filename = 'Rongjing-noisefacebook-label0-algo.txt'

f = open(basePath + filename)
lines = f.readlines()
f.close()

m = {}

for line in lines:
	#print line
	t = line.strip().split('\t')
	#print t
	#print t[0] + "-" + t[1]
	m[ t[0] + "-" + t[1] ] = line


filename = 'Rongjing-noisefacebook-label0-algo_cleanedRepeats.txt'

f = open(basePath + filename, 'w')
for i,j in m.iteritems():
	f.write(j)

f.close()