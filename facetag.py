
# coding: utf-8

# In[5]:

# This script detects faces in picture, rotates the pictures automatically according to the exif tag (jhead must be installed)
# asks for the Names of the people and adds the names as in the Note field of the Exif info.
# It uses the face_recognition library to detect automatically faces. 
# It then writes the names as "Jon Doe, John Smith, ..." to the comment exif tag  (in the order from left to right)

########## Install Instructions
# conda install -c menpo dlib 
# pip install face_recognition

import face_recognition
import os, sys
import numpy as np
import piexif
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle #pickle.dump( data, open( "file.save", "wb" ) ) #data = pickle.load( open( "file.save", "rb" ) )
import subprocess 
from pathlib import Path


plt.rcParams['toolbar'] = 'None'


# ## Functions

# In[6]:

def in_notebook():
    """
    Returns ``True`` if the module is running in IPython kernel,
    ``False`` if in IPython shell or other Python shell.
    """
    return 'ipykernel' in sys.modules

if in_notebook():
    print('Running in Notebook')
else:
    print('Running in Shell')

    

def ExpandDirectories(flist, ending='.jpg', not_conatin=None):
    newplotfiles = []
    if isinstance(flist,str):
        flist = [flist]
    for directory in flist:
        if os.path.isdir(directory):
            for dirpath, dirnames, files in os.walk(directory, followlinks=True):
                for name in files:
                    if (ending.lower() in name.lower()) and (not_conatin==None or not_conatin not in name  ):
                        newplotfiles += [os.path.join(dirpath, name)]
#                         print('Include '+name)
        elif os.path.isfile(directory):
            newplotfiles += [directory]
    return newplotfiles




def ShowImg(pic, trim=None, Timer=1):
    if isinstance(pic,str):
        data = mpimg.imread(pic)
    else:
        data = pic
    if trim!=None:
        fig = plt.imshow(data[trim[0]:trim[2],trim[3]:trim[1]])
    else:
        fig = plt.imshow(data)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    if Timer!=None:
        plt.show(block=False)
        if not in_notebook(): plt.pause(Timer)
        plt.close()
    else:
        plt.show(block=False)
        if not in_notebook(): plt.pause(0.3)
    return data


def arr2str(arr, sep=', ', pre_counter=False, pre_counter_str=' = '):
    output = ''
    for i,item in enumerate(arr):
        if pre_counter:      output += str(i)+pre_counter_str
        output += str(item)
        if i<len(arr)-1:     output += sep
    return output
        

def ExeCmd(cmd, errormessage='Error'):
    try:
        output = subprocess.check_output(cmd, shell=True)
        if output!= "b''":
            print(output) 
        return output
    except: 
        print(errormessage)    
    
def RotateImg(pic):
    ExeCmd("jhead -autorot \'"+pic+"\'", errormessage= 'Could not rotate picture. Is jhead installed?')
    
    
def MultipleChoice(arr, pre='', post ='Please select:'):
    text = pre+arr2str(arr, sep='\n', pre_counter=True)+'\n'+post+'\n'
    return input(text)
    
# MultipleChoice(['a','b','c'])




def Path2Dir(path,   end_sep=True):
    directory = os.path.dirname(path)
    if end_sep:
        directory = directory+sep_char
    return directory


def Path2Filename(path,  RemoveEnding = False ): 
    filename = os.path.basename(path)
    if RemoveEnding:
        filename = '.'.join(filename.split('.')[:-1])
    return filename



# ## Arguments

# In[7]:

args = {
    'folder' : ['demo'],
    'database' : 'Face_encodings.save',
    'shuffle' : True,
    'softlinks' : True,
    'softlink_folder' : 'People folders',
}


if not in_notebook():
    import argparse

    parser = argparse.ArgumentParser(description='Detect Faces and write them in the exif tag')
    parser.add_argument('-f','--folder',nargs='*', help='Picture folders',type=str , default=args['folder'])
    parser.add_argument('--database', help='Database File, storing the face encodings',type=str , default=args['database'])
    parser.add_argument('--shuffle', type=lambda s: s.lower() in ['true', 't', 'yes', '1'], default=args['shuffle'])
 
    for k,v in vars(parser.parse_args()).items():
        args[k] = v
    
    


# ## Load Database

# In[8]:

if  os.path.exists(args['database']): 
    faces = pickle.load( open( args['database'], "rb" ) )
else:
    faces = {
        'encs' : [np.zeros([128])]    ,
        'names' : np.array(['0'])     ,       
    }
    
    


# ## Recognize Faces

# In[62]:

pics = np.array(ExpandDirectories(args['folder']))
if args['shuffle']:
    np.random.shuffle(pics)



for pic_idx, pic in enumerate(pics):
    print("------------------------------"+"{0:.2f}".format(pic_idx/len(pics)*100)+'% ,   '+str(pic_idx)+'/'+str(len(pics)))
    print('Loading: '+pic)
    try:
        RotateImg(pic)
        pic_data = ShowImg(pic, Timer=None)

        print('Detecting faces....')
        image = face_recognition.load_image_file(pic)
        locs = face_recognition.face_locations(image)
        encs = face_recognition.face_encodings(image, known_face_locations=locs)


        # sort according to x coordinate  (left to right)
        x_coors = np.array([l[1] for l in locs])
        sort_idxs = np.argsort(x_coors)
        locs = [locs[idx] for idx in sort_idxs]
        encs = [encs[idx] for idx in sort_idxs]

        plt.close()

        # recognize each face
        if len(encs) ==0: print('No faces found.')
        names = []
        for i in range(len(encs)):
            matches_bool = np.array(face_recognition.compare_faces(faces['encs'], encs[i], tolerance=0.48) )

            
            if matches_bool.any():
                red_faces = faces['names'][matches_bool]
                distances = face_recognition.face_distance(faces['encs'][matches_bool], encs[i])
                print('Multiple possible Faces found:\n'+ 
                      arr2str(["{0:.2f}".format(d)+'  '+name for d,name in zip(distances,red_faces)], sep='\n'))         
                names += [red_faces[np.argmin(distances)]]                        
                print('Choosing the closest match: '+names[-1])                    
                
                
                plt.close()
            else:
                ShowImg(pic, trim=locs[i], Timer=None)
                new_name = input('Please name this face (empty if you want to skip): ')
                if new_name!='':
                    names += [new_name]
                    faces['encs'] = np.vstack([faces['encs'],encs[i]])
                    faces['names'] = np.hstack([faces['names'],new_name])
                else:
                    print('Ok. Skipping.')
                plt.close()


        if len(names)>0:   # only do something if there were faces
            print('Writing names to exif tag: '+arr2str(names))
            output = ExeCmd("jhead -cl \'"+arr2str(names)+"\'   \'" + pic+"\'" , errormessage='Error: Could not write Tags.' )     


            # safe softlink
            if args['softlinks'] and len(pics)>1:
                for name in names:
                    namefolder = os.path.join(args['folder'][0],'..', args['softlink_folder'], name)
                    if not os.path.exists(namefolder):    os.makedirs(namefolder)    
                    relative_from_subfolder = os.path.join('..','..',pic)
                    os.symlink(relative_from_subfolder, os.path.join(namefolder,Path2Filename(pic)))


            # periodically save the database
            pickle.dump( faces, open( args['database'], "wb" ) ) #data = pickle.load( open( "file.save", "rb" ) )
    except KeyboardInterrupt: 
        raise
    #except:
    #    print('Error in processing image. Skipping.')    



# In[ ]:



