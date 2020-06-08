! curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
! sudo apt-get install git-lfs
! git lfs install
! git clone https://github.com/neheller/kits19.git
! pip3 install -r /content/kits19/requirements.txt
%cd kits19
! python -m starter_code.get_imaging
! pip3 install -U --user nilearn
! pip3 install -U --user nibabel

from starter_code.utils import load_case
volume, segmentation = load_case("case_00123")
print(volume)
print(segmentation)
print("________________________")
print(volume.dataobj)
print(img.dataobj[0])
print(volume)
im = nib.nifti1.Nifti1Pair((389, 512, 512),affine=None)
img = nib.nifti1.load("/content/kits19/data/case_00000/imaging.nii.gz")

! python -m starter_code.visualize
import starter_code.visualize
%mkdir kits19/data/cs1
starter_code.visualize.visualize(1, "/content/kits19/data")

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import nilearn as nil
from nilearn import plotting
from nilearn import image
plotting.plot_img("/content/kits19/data/case_00000/imaging.nii.gz", cut_coords= 1, display_mode='z')
plotting.plot_stat_map("/content/kits19/data/case_00000/imaging.nii.gz")
plotting.plot_glass_brain("/content/kits19/data/case_00000/imaging.nii.gz")
img = nib.nifti1.load("/content/kits19/data/case_00000/imaging.nii.gz")

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import nilearn as nil
from nilearn import plotting
from nilearn import image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
plotting.plot_img("/content/kits19/data/case_00000/imaging.nii.gz", cut_coords= [-222], display_mode='z', draw_cross = True , black_bg = True , colorbar = True, cmap='gray' )

def convert(file_patch):
    
    file_patch = "/content/kits19/data/case_00000/imaging.nii.gz"
    #Załadowanie danych
    img_nifti = nib.load(file_patch)
    #Wybór obrazów poprzecznych z paczki nifti 
    z_img = image[110, :, :]
    for i in z_img:
        ar_data[i] = z_img.astype(np.float32)

'''
import numpy as np
import pdb
# If you are not using nifti files you can comment this line
import nibabel as nib
import scipy.io as sio
# ----- Loader for nifti files ------ #
def load_nii (imageFileName, printFileNames) :
    if printFileNames == True:
        print ("/content/kits19/data".format(imageFileName))
    img_proxy = nib.load(imageFileName)
    imageData = img_proxy.get_data()
    return (imageData,img_proxy)
def release_nii_proxy(img_proxy) :
    img_proxy.uncache()
'''
'''
def convert(file_patch):
    
    file_patch = "/content/kits19/data/case_00000/imaging.nii.gz"
    #Załadowanie danych
    img_nifti = nib.load(file_patch)
    #Wybór obrazów poprzecznych z paczki nifti 
    z_img = image[110, :, :]
    print(z_img.dtype)
    #Zamiana danych na tablicę float dla każdego zadjęcia
    for i in z_img:
        data[i] = plt.imshow(np.asarray(z_img[i], dtype=np.float16)
    
    img_ni = plotting.plot_img(z_img, cut_coords= [-222], display_mode='z', cmap='gray')
    img = load_img(image_ni)
    img_array = img_to_array(img)

    img_nifti = nib.load("/content/kits19/data/case_00000/imaging.nii.gz")
    img = img_nifti.get_data()

    plt.imshow(np.asarray(img2, dtype=np.uint16)
'''
"DOKUMENTACJA do NiLearn: https://nilearn.github.io/modules/reference.html#module-nilearn.plotting"
"https://www.pluralsight.com/guides/importing-image-data-into-numpy-arrays https://stackoverflow.com/questions/44422248/how-to-convert-nifti-file-to-numpy-array https://vincentblog.xyz/posts/medical-images-in-python-computed-tomography"