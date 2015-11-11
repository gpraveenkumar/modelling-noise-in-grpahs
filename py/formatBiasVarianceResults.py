path = "../results/"
fileName = "school074-label0_biasVariance_flipLabelResults"

f = open(path + fileName + ".txt")
header = f.readline()
lines = f.readlines()
f.close()

output = []

header = header.strip().split('\t')
l = [header[0],header[1],"Label1","Error"]

#header = header[2:]
print header
output.append(l)

for line in lines:
	line = line.strip().split('\t')

	for i in range(2,len(header)):
		l = [ line[0], line[1] ]
		l.append(header[i])
		l.append(str(abs(float(line[i]))))
		output.append(l)
		print l		

fileName = "school_myMPLE"
f = open(path + fileName + ".txt",'w')
f.write("\n".join('\t'.join(i) for i in output))
f.close()