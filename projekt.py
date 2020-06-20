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

#wektory obrazów do treningu
volume = [volume0, volume1]
segmentation = [segmentation0,segmentation1]

volume_shape = np.shape(volume0)
segmentation_shape = np.shape(segmentation0)

print(f"shape of volume {volume_shape}")
print(f"shape of segmentation {segmentation_shape}")

from starter_code.visualize import visualize

visualize("case_00000", "CT_case0")

cd "/Users/mikolajdobrowolski/kits19/CT_case0"

#sprawdzenie czy obrazy są w skali szarości
image = plt.imread("00000.png", cmap = 'gray')
image_shape = np.shape(image)
print(f"image shape {image_shape}")

#wyświetlenie obrazów CT
plt.imshow(volume0[150, :, :], cmap = 'gray')

plt.imshow(segmentation0[150, :, :])
