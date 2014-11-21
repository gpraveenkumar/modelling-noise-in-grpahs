
f = open('toRun_variants1.txt','w')
for trainingSize in [0.05,0.1,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.7,0.8,0.9]:
	for percentageOfGraph in [0.15]:
		f.write(str(trainingSize) + ' ' + str(percentageOfGraph) + '\n')
f.close()