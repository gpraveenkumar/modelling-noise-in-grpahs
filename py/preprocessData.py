# Pre-process data to make the graph without NA

path = "../data/"
schoolNo = 74

input = "school0" + str(schoolNo) + "-parsed.txt"
output = "preProcessed-school0" + str(schoolNo) + ".txt"

f = open(path + input)
lines = f.readlines()
f.close()

"""
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
"""

l = []
#id_number_mapping = {}
#ctr = 1

for i in range(1,len(lines)):
	t = lines[i].strip().split(" ")

	#If either label in NA, I throw out the whole record as only 5-10 cases have atleast 1 NA label. Hence "or"
	if t[-1] == "NA" or t[-2] == "NA" or t[-3] == "NA":
		continue
	if t[0] == "NA":
		continue
	
	# Throw out nodes that have NA for all the 10 friends
	flag = 1
	for j in range(1,len(t)-3):
		if t[j] != "NA":
			flag = 0;
			break;
	if flag == 1:
		continue

	temp = []
	for j in range(0,len(t)-3):
		temp.append( t[j] )

		"""
		if t[j] not in id_number_mapping:
			#id_number_mapping[ t[j] ] = ctr
			#temp.append(str(ctr))
			#ctr += 1
		else:
			#temp.append(str(id_number_mapping[ t[j] ]))
		"""

	temp.append(t[-3])
	temp.append(t[-2])
	temp.append(t[-1])
	l.append(temp)

f = open(path + output, 'w')
f.write("AID MF1AID MF2AID MF3AID MF4AID MF5AID FF1AID FF2AID FF3AID FF4AID FF5AID label1 label2 label3"+'\n')
f.write('\n'.join(' '.join(i) for i in l))
f.close()
