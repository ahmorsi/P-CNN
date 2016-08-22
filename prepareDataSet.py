import os
import re
import sys

if len(sys.argv) < 2:
	sys.exit("Usage : prepareDataSet <DataSetBaseDir>")

baseDir = os.path.join(sys.argv[1],'splits')
if not os.path.exists(baseDir):
	sys.exit("{0} -- Not exist\n".format(baseDir))

rawsplitsDir = os.path.join(sys.argv[1],'raw_splits')
if not os.path.exists(rawsplitsDir):
	os.makedirs(rawsplitsDir)
label = 0
class2Label = dict()
for file in os.listdir(baseDir):
    if file.endswith(".txt"):
	m = re.search('(?P<class>.+)_test_split(?P<split>\d+)', file)
	className = m.group('class')
	splitNum = m.group('split')
	directory = os.path.join(baseDir,className)
	if not os.path.exists(directory):
		os.makedirs(directory)
		label = label + 1
		class2Label[className] = label
	trainPath = os.path.join(directory,"train{0}.txt".format(splitNum))
	testPath = os.path.join(directory,"test{0}.txt".format(splitNum))	
	trainSw = open(trainPath,"a" if os.path.exists(trainPath) else "w")
	testSw = open(testPath,"a" if os.path.exists(testPath) else "w")	 	
	with open(os.path.join(baseDir,file)) as f:	
		for line in f:
			name,type = line.split()
			if name.endswith('.avi'):
				name = name[:-4]
			if type == "1":
				trainSw.write("{0}\t{1}\n".format(name,class2Label[className]))
			else:
				testSw.write("{0}\t{1}\n".format(name,class2Label[className]))		
        print(file)
	os.rename(os.path.join(baseDir,file),os.path.join(rawsplitsDir,file))
	trainSw.close()
	testSw.close()

with open(os.path.join(baseDir,"Label2Class.txt"),"w") as f:
	for key in class2Label:
		f.write("{0}\t{1}\n".format(class2Label[key],key))
