path = "../data/"
schoolNo = 74

input = "school0" + str(schoolNo) + "-parsed.txt"
output = str(schoolNo) + "-feature-all.txt"

f = open(path + input)
lines = f.readlines()
f.close()

l = []
for i in range(1,len(lines)):
	t = lines[i].strip().split(" ")

	if t[-1] != "NA" and t[-2] != "NA" and t[-3] != "NA":
		l1 = []
		l1.append(t[-1])
		l1.append(t[-2])
		l1.append(t[-3])
		l.append(l1)

f = open(path + output, 'w')
f.write("ratings1 ratings2 ratings3"+'\n')
f.write('\n'.join(' '.join(i) for i in l))
f.close()