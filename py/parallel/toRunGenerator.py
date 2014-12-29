
"""
f = open('toRun_variants_double.txt','w')
for trainingSize in [0.05,0.1,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.7,0.8,0.9]:
	for percentageOfGraph in [0.05,0.15,0.30,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
		for noOfTimesToRepeat in [2,5,10]:
			for percentageOfGraph2 in [0.5,0.9]:
				f.write(str(trainingSize) + ' ' + str(percentageOfGraph) + ' ' + str(noOfTimesToRepeat) + ' ' + str(percentageOfGraph2) + '\n')
f.close()
"""

"""
f = open('toRun_variants_polblog.txt','w')
for trainingSize in [0.05,0.1,0.2,0.4,0.5,0.7,0.8,0.9]:
	for percentageOfGraph in [0.05,0.15,0.30,0.4,0.5,0.6,0.7,0.9,1.0]:
		for noOfTimesToRepeat in [2,5,10]:
			for percentageOfGraph2 in [0.0]:
				f.write(str(trainingSize) + ' ' + str(percentageOfGraph) + ' ' + str(noOfTimesToRepeat) + ' ' + str(percentageOfGraph2) + '\n')
f.close()
"""
"""
f = open('toRun_original_cora.txt','w')
for trainingSize in [0.05,0.1,0.2,0.4,0.5,0.7,0.8,0.9]:
	for percentageOfGraph in [0]:
		for noOfTimesToRepeat in [0]:
			for percentageOfGraph2 in [0.0]:
				f.write(str(trainingSize) + ' ' + str(percentageOfGraph) + ' ' + str(noOfTimesToRepeat) + ' ' + str(percentageOfGraph2) + '\n')
f.close()
"""

f = open('toRun_temp_cora.txt','w')
for trainingSize in [0.05,0.1,0.2,0.4,0.5,0.7,0.8,0.9]:
	for percentageOfGraph in [0.8]:
		for noOfTimesToRepeat in [2,5,10]:
			for percentageOfGraph2 in [0.0]:
				f.write(str(trainingSize) + ' ' + str(percentageOfGraph) + ' ' + str(noOfTimesToRepeat) + ' ' + str(percentageOfGraph2) + '\n')
f.close()
