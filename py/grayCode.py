grayCodeCounts = {}

def computeGrayCodes(n):
	if n in grayCodeCounts:
		print "Memoized."
		return grayCodeCounts[n]

	if n == 0:
		counts = set()
		counts.add( (0,0) )
		return list(counts)

	grayCodeList = []
	grayCodeList.append([0])
	grayCodeList.append([1])

	i = 1
	while i<n:
		i += 1

		newGrayCodeList = []
		for l in grayCodeList:
			new_l = [0] + l
			newGrayCodeList.append(new_l)
			new_l = [1] + l
			newGrayCodeList.append(new_l)

		grayCodeList = newGrayCodeList

	counts = set()
	
	size = len(grayCodeList[0])
	for l in grayCodeList:
		ones = sum(l)
		zeros = size - ones
		counts.add( (zeros,ones) )

	print counts
	grayCodeCounts[n] = list(counts)

	return grayCodeCounts[n]

def computeGrayCodeCountings(n):
	if n in grayCodeCounts:
		print "Memoized."
		return grayCodeCounts[n]

	if n == 0:
		l = []
		l.append( (0,0) )
		return list( l )

	l = []

	for i in range(n+1):
		l.append( (i,n-i) )

	grayCodeCounts[n] = l

	return grayCodeCounts[n]


"""
print computeGrayCodes(1)
print computeGrayCodes(2)
print computeGrayCodes(3)
print computeGrayCodes(2)
print computeGrayCodes(4)
print computeGrayCodes(0)
"""


print computeGrayCodeCountings(1)
print computeGrayCodeCountings(2)
print computeGrayCodeCountings(3)
print computeGrayCodeCountings(2)
print computeGrayCodeCountings(4)
print computeGrayCodeCountings(30)