import csv
import os

ageClasses = {'(0, 2)': 0, '(4, 6)': 1, '(8, 12)': 2, '(15, 20)': 3, '(25, 32)': 4, '(38, 43)': 5, '(48, 53)': 6, '(60, 100)': 7}

with open("original.txt") as tsv:
    next(tsv)
    with open('ageLabels.txt', 'w') as f:
	for line in csv.reader(tsv, dialect="excel-tab"):
	      userId = line[0].strip()
	      imageName = line[1].strip()
	      faceId = line[2].strip()
	      age = line[3].strip()
	      combinedPath =  userId + '/landmark_aligned_face.' + faceId + '.' + imageName
	      if age != 'None' and age in ageClasses.keys():
		f.write(combinedPath + " " + str(ageClasses[age]))
		f.write('\r\n')
       
