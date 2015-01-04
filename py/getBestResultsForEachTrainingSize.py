fileName = 'facebook-label0_flipLabelResults'

# which index of the input file is used to compute best score. 
# 2 = Squared Loss
# Note - for certain cases best score would be the lowest value as in case of Squared Loss. It could be the higest value as in case of accuracy.
fieldIndex = 2 



f_in = open('../results/' + fileName +'.txt')

# no need for first line...Skipping the header
junk_ = f_in.readline()

map_trainingSize_bestScore = {}

# This holds the 'Label' corresponding to the best score
map_trainingSize_Label = {}

for line in f_in:

	# Skip Blank lines
	if len(line) < 2:
		continue

	fields = line.strip().split()

	# Skip the lines where the 'Label' is original
	if fields[0] == "original":
		continue

	Label = fields[0]
	trainingSize = float(fields[1])
	score = float(fields[2])

	if trainingSize not in map_trainingSize_bestScore:
		map_trainingSize_bestScore[ trainingSize ] = score
		map_trainingSize_Label[ trainingSize ] = Label
	else:
		if score < map_trainingSize_bestScore[ trainingSize ]:
			map_trainingSize_bestScore[ trainingSize ] = score
			map_trainingSize_Label[ trainingSize ] = Label

f_in.close()



for i in map_trainingSize_bestScore:
	print str(i),map_trainingSize_Label[ i ],str(map_trainingSize_bestScore[ i ])