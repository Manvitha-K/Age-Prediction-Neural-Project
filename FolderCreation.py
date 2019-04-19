import os
import shutil
import numpy as np
from random import shuffle
#from numpy.random import seed
#from numpy.random import rand

def createFolders():
	cwd = os.getcwd()
	for i in range(8):
		for folder in [ cwd + '/data/age_train/', cwd + '/data/age_valid/', cwd + '/data/age_test/']:
			folderPath = folder + str(i) + "/"
			if not os.path.isdir(folderPath):
				os.makedirs(folderPath)
	print("folder creation complete")



def parsefiles(filepath):
	myList = {}
	with open(filepath, 'r') as f:
    		x = f.readlines()
	for line in x:
		path, label = line.rstrip().split(' ')
		if myList.get(int(label)) == None:
			myList[int(label)] = [path]
		else:
			myList[int(label)].append(path)
	return myList


def moveImages(ImgList, newImgDir):
	cwd = os.getcwd()
	for ele in ImgList:
		folder, ImgName = ele.split('/')
		oldImgPath = cwd + '/aligned/' + ele
		#when face cropped Images are used
		#oldImgPath = cwd + '/cropped/' + ele
		shutil.copy(oldImgPath, newImgDir)

def splitData(myList):
	validList, trainList = [], []
	for val, path in zip(values, myList):
		if val <= 0.8:
			trainList.append(path)
		else:
			validList.append(path)
	return(trainList, validList)
	

if __name__ == '__main__':
	cwd = os.getcwd()
	createFolders()
	labelsInfo = parsefiles(cwd + "/labels/ageLabels.txt")
	#When using face cropped Images
	#labelsInfo = parsefiles(cwd + "/labels/ageCroppedLabels.txt")
	trainCnt, testCnt, valCnt = 0, 0, 0
	print(labelsInfo.keys())

	for key in labelsInfo.keys():
		myList = labelsInfo[key]
		shuffle(myList)
		N = len(myList)
		trainset, validset, testset = myList[:int(0.6*N)], myList[int(0.6*N):int(0.8*N)], myList[int(0.8*N):]
		trainCnt += len(trainset)
		testCnt += len(testset)
		valCnt += len(validset)
		newImgDirs = [cwd + "/data/age_train/" + str(key), cwd + "/data/age_test/"+str(key) , cwd + "/data/age_valid/" + str(key)]
		dataSets = [trainset, testset, validset]		
		for newImgFolder, dataList in zip(newImgDirs, dataSets):
			moveImages(dataList, newImgFolder)
	
	print("train {}, valid {}, test {}". format(trainCnt, testCnt, valCnt))


		
	
