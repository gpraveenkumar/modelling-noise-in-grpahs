# Code to process the data and obtain the histrogram of the labels (using R) amongst 1-6, in order to determine an appopriate cutt off to make it binary

path = "../data/"
schoolNo = 74

input = "school0" + str(schoolNo) + "-parsed.txt"
featureNumbers = [1,2,3]

for featureNumber in featureNumbers:
	output = str(schoolNo) + "-feature"+ str(featureNumber) +".txt"

	f = open(path + input)
	lines = f.readlines()
	f.close()

	l = []

	for i in range(1,len(lines)):
		t = lines[i].strip().split(' ')

		if t[-featureNumber] != "NA":
			l.append(t[-featureNumber])

	f = open(path + output, 'w')
	f.write("ratings"+'\n')
	f.write('\n'.join(l))
	f.close()