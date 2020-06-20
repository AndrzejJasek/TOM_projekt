import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt
import os

cd "/Users/mikolajdobrowolski/kits19"

from starter_code.utils import load_case
#wczytywanie danych do trenowania(narazie tylko 5 pacjentów)
volume0, segmentation0 = load_case("case_00000")
volume0 = volume0.get_fdata() #metoda do wyciągania tablicy danych - u nas obrazów nerek i obrazów segmentacji (metoda z biblioteki nibabel)
segmentation0 = segmentation0.get_fdata()

volume1, segmentation1 = load_case("case_00001")
volume1 = volume1.get_fdata() 
segmentation1 = segmentation1.get_fdata()

volume2, segmentation2 = load_case("case_00002")
volume2 = volume2.get_fdata() 
segmentation2 = segmentation2.get_fdata()

volume3, segmentation3 = load_case("case_00003")
volume3 = volume3.get_fdata() 
segmentation3 = segmentation3.get_fdata()

volume4, segmentation4 = load_case("case_00005")
volume4 = volume4.get_fdata() 
segmentation4 = segmentation4.get_fdata()

#wektory obrazów do treningu
volume = [volume0, volume1, volume2, volume3]
segmentation = [segmentation0,segmentation1,segmentation2,segmentation3]

volume_shape = np.shape(volume0)
segmentation_shape = np.shape(segmentation0)

print(f"shape of volume {volume_shape}")
print(f"shape of segmentation {segmentation_shape}")

from starter_code.visualize import visualize

visualize("case_00000", "CT_case0")

cd "/Users/mikolajdobrowolski/kits19/CT_case0"

#sprawdzenie czy obrazy są w skali szarości
image = plt.imread("00000.png")
image_shape = np.shape(image)
print(f"image shape {image_shape}")
