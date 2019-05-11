## Instructions for Running:    
**Dependencies:**  
Python 3: Running CNN  
Python 2.7: Running Tkinter  
Tkinter  
numpy  
Keras  
Pillow  
Numpy  
Pathlib  
cv2  
dlib  
gensim  
cmake  
  
**Run:**  
After cloning the repository from github (assuming you maintained same relative folder structure)  

**For running base model:**  
python Base.py   

**For running pretrained model:**  
python preTrained.py  

**For running GUI:**  
GUI.py  

**For creating cropped face images:**  
faceRecog.ipynb  

**For loading the trained model:**    
These models should be downloaded to /hdf folders    
https://drive.google.com/open?id=1M-H2zl1u6GkV8ozDYNb3_GdErs9utqav    


Labels folder has code that does data preprocessing  
hdf folder has all the saved models  
FolderCreation.py does the train, test, valid split and creates folders in a suitable way such that imagedatagenerator of keras can use it  
Base.py is basic CNN with 3 CNN layers followed by 2 dense layers  
faceRecog.ipynb does the face detection algorithm and extracts cropped face images from original images  
preTrainbed.py loads the model trained on original images and then finetunes these weights by running on only facial images  
GUI.py: Application that when inputted with an image does age estimation  

**References:**  
Face Recognition  
https://github.com/krasserm/face-recognition
