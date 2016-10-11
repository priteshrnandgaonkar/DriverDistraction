import PIL
from PIL import Image
import numpy as np
from numpy import*
import os

lookup = [[1,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0]
,[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,1]]
# from PIL import Image

# img = Image.open('somepic.jpg')
# wpercent = (basewidth/float(img.size[0]))
# hsize = int((float(img.size[1])*float(wpercent)))
# img = img.resize((basewidth,hsize), PIL.Image.ANTIALIAS)
# img.save('sompic.jpg')

def load_training_data():
    basewidth = 100
    path = os.getcwd()+'/imgs'
    path = path + '/train'
    classification_folders = np.array([path+'/'+name for name in os.listdir(path) if os.path.isdir(path+'/'+name)])
    image_name_list = []
    image_matrix_list = []
    classification_list = []

    for classification in classification_folders:
    	image_name_sublist = [classification+'/'+image_name for image_name in os.listdir(classification) if image_name.endswith('.jpg')]
    	image_name_list = np.append(image_name_list, image_name_sublist)
    	for image_name in image_name_sublist:
            print image_name
            img = Image.open(image_name)
            wpercent = (basewidth/float(img.size[0]))
            hsize = int((float(img.size[1])*float(wpercent)))
            img = img.resize((basewidth,hsize), PIL.Image.ANTIALIAS)
            # matrix = asarray(Image.open(image_name))
            matrix = asarray(img)
            x = matrix.shape[0]
    	    y = matrix.shape[1]*matrix.shape[2]
            matrix.resize(x*y)
            image_matrix_list.append(matrix)   
            clist = classification.split('/')
            last = clist.pop()
            c_index = int(last[1])
            classification_list.append(lookup[c_index])

    return (image_name_list, image_matrix_list, classification_list)

def load_test_data():
    basewidth = 100
    path = os.getcwd()+'/imgs'
    path = path + '/test'
    path_image_name_list = [path+'/'+image_name for image_name in os.listdir(path) if image_name.endswith('.jpg')]
    image_matrix_list = []
    image_name_list = []
    for image_name in path_image_name_list:
        print image_name
        img = Image.open(image_name)
        wpercent = (basewidth/float(img.size[0]))
        hsize = int((float(img.size[1])*float(wpercent)))
        img = img.resize((basewidth,hsize), PIL.Image.ANTIALIAS)
        # matrix = asarray(Image.open(image_name))
        matrix = asarray(img)
        actual_image_name = image_name.split('/').pop()
        image_name_list.append(actual_image_name)
        x = matrix.shape[0]
        y = matrix.shape[1]*matrix.shape[2]
        matrix.resize(x*y)
        image_matrix_list.append(matrix)

    return (image_name_list, image_matrix_list)   


 




