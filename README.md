# FaceTag

Organizes pictures according to peoples faces in the pictures. It also writes the peoples names in the usercomment EXIF tag. It uses the amazing [face_recognition](https://github.com/ageitgey/face_recognition) based on dlib. 


## Usage

It can be used in the console using
`python face-rec.py`
or in a jupyter notebook.



### Give a picture directory
`python face-rec.py  --folder demo`
and it will recursively get all jpg files.

![](res/pics.jpg)
and also rotate them according to the Orientation exif tag using [jhead](http://www.sentex.net/~mwandel/jhead/).

### Label first faces 
It asks for the names of all unknown people in a picture and adds them to it's name database.
![](res/demo1.jpg)
![](res/demo2.jpg)
### Exif Comment Tag
The exif user_comment tag is filled with the names from the left to the right :-) 

![](res/exif.jpg)

### Face recognition and labeling already working automatically
Already in the next picture it can recognize Barack and no user action is needed.

![](res/demo3.jpg)
![](res/demo6.jpg)

### Folders with Softlinks  
Additionally to EXIF tags, subfolders with softlinks are created according to the names.

![](res/folders.jpg)




## Installation and dependencies

Getting dlib to work is a little tricky. The following works for me (with an anaconda environment):
`conda install -c menpo dlib 
pip install face_recognition
sudo apt install jhead`
