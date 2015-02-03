
"""
f = open('toRun_variants_double.txt','w')
for trainingSize in [0.05,0.1,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.7,0.8,0.9]:
	for percentageOfGraph in [0.05,0.15,0.30,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
		for noOfTimesToRepeat in [2,5,10]:
			for percentageOfGraph2 in [0.5,0.9]:
				f.write(str(trainingSize) + ' ' + str(percentageOfGraph) + ' ' + str(noOfTimesToRepeat) + ' ' + str(percentageOfGraph2) + '\n')
f.close()
"""


f = open('toRun_variants_school074_new.txt','w')
for trainingSize in [0.05,0.1,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.7,0.8,0.9]:
	for percentageOfGraph in [0.05,0.15,0.30,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
		for noOfTimesToRepeat in [25,50,75,100,250]:
			for percentageOfGraph2 in [0.0]:
				f.write(str(trainingSize) + ' ' + str(percentageOfGraph) + ' ' + str(noOfTimesToRepeat) + ' ' + str(percentageOfGraph2) + '\n')
f.close()



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

"""
f = open('toRun_temp_cora.txt','w')
for trainingSize in [0.05,0.1,0.2,0.4,0.5,0.7,0.8,0.9]:
	for percentageOfGraph in [0.8]:
		for noOfTimesToRepeat in [2,5,10]:
			for percentageOfGraph2 in [0.0]:
				f.write(str(trainingSize) + ' ' + str(percentageOfGraph) + ' ' + str(noOfTimesToRepeat) + ' ' + str(percentageOfGraph2) + '\n')
f.close()
"""

"""
[0.5,0.5,0.5],
 [0.6,0.6,0.6],
 [0.4,0.4,0.4],
 [0.7,0.7,0.7],
 [0.5,0.6,0.6],
 [0.5,0.5,0.5],
"""

# Parameter for school074
"""
parameterValues = [
 [0.4977,0.5034,0.513],
 [0.4994,0.5131,0.5124],
 [0.4982,0.5116,0.5154],
 [0.4958,0.5171,0.5207],
 [0.4979,0.5178,0.5181],
 [0.4982,0.5272,0.5238],
 [0.4908,0.6164,0.6215],
 [0.4921,0.6171,0.6214],
 [0.4913,0.6178,0.6239],
 [0.4909,0.6169,0.6225],
 [0.4912,0.6164,0.6216],
 [0.4909,0.6192,0.6186],
 [0.4908,0.6148,0.6239]
]
"""
"""
parameterValues = [
 [0.5,0.5,0.5],
 [0.6,0.6,0.6],
 [0.4,0.4,0.4],
 [0.7,0.7,0.7],
 [0.5,0.6,0.6]
]
"""

parameterValues = [ [ 0.4944,0.5843,0.5964] ]

f = open('param_school074_all.txt','w')
for trainingSize in [0.05,0.1,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.7,0.8,0.9]:
	for percentageOfGraph in [0]:
		for noOfTimesToRepeat in [0]:
			for percentageOfGraph2 in [0.0]:
				for p in parameterValues:
					f.write(str(trainingSize) + ' ' + str(percentageOfGraph) + ' ' + str(noOfTimesToRepeat) + ' ' + str(percentageOfGraph2) + ' ' + ' '.join(str(i) for i in p) +'\n')
f.close()
