f = open('../results/school074-label0_moreRepeats_flipLabelResults.txt')

junk = f.readline()
junk = f.readline()
junk = f.readline()

alreadyRan = []

for line in f:
	fields = line.strip().split('\t')
	
	trainingSize = fields[1]
	percentageFlipped,noOfRepeats = fields[0].split('_')
	percentageFlipped = percentageFlipped[:-4]
	noOfRepeats = noOfRepeats[:-6]

	percentageFlipped = str( float(percentageFlipped)/100 )

	t = trainingSize + " " + percentageFlipped + " " + noOfRepeats + " 0.0"
	alreadyRan.append(t) 

f.close()

f = open('./parallel/toRun_variants_school074_new.txt')

toRun = []
for line in f:
	toRun.append( line.strip() )

f.close()

yetToRun = [i for i in toRun if i not in alreadyRan]

f = open('./parallel/yetToRun.txt','w')
f.write('\n'.join(yetToRun))
f.close()