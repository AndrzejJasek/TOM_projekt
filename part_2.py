import numpy as np
import matplotlib.pylab as plt
from skimage import color


class Image_segmentation:
    def __init__(self, image):
        self.image = image
        self.hist = np.histogram(image, bins=256)
        self.number_of_detected_kidneys = 0
        # Inicjalizacja zmiennych przechowywujących potencjalne maski.
        self.mask_kidney_1 = None
        self.mask_tumor_1 = None
        self.mask_kidney_2 = None
        self.mask_tumor_2 = None

    def max_range_from_histogram(self):
        """
        Funkcja zwraca największy przedział o niezerowej ilości danych jako krotkę.
        """
        ind = 0
        for i in range(len(self.histogram[0]) - 1, -1, -1):
            if(0 < self.histogram[0, i]):
                ind = i
                break
        return (self.histogram[1, i], self.histogram[1, i+1])

    def index_of_value_in_given_range(self, ran):
        """
        Funkcja zwraca indeks elementu zdjęcia o wartości mieszczącej się w podanym przedziale.
        """
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                if(r[0] <= self.image[i,j] and self.image[i,j] < r[1]):
                    return (i,j)

    def region_growing_local(image, seed, bottom_threshold, upper_threshold):
        """
        Algorytm rozrostu obszaru w wersji lokalnej. Zwraca maskę binarną zdjęcia.
        """
        lower_val = image[seed] - bottom_threshold
        upper_val = image[seed] + upper_threshold

        def clock_iter(row, col):
            """
            Wyjściowy zbiór służy do chodzenia wokół podanej współrzędnej.
            """
            return {(row-1, col),(row-1, col+1),(row, col+1),(row+1, col+1),(row+1, col),(row+1, col-1),(row, col-1),(row-1, col-1)}

        region = {seed}
        region_to_check = [seed]
        while region_to_check != []: # Otrzymujemy tym kodem wszystkie pożądane koordynaty.
            to_check = region_to_check.pop(0)
            for ind in clock_iter(to_check[0], to_check[1])-region:
                try:
                    if lower_val < image[ind] and image[ind] < upper_val:
                        region.add(ind)
                        region_to_check.append(ind)
                except IndexError:
                    None

        result = np.zeros(image.shape, dtype=bool) # Konwertujemy koordynaty wskazujące na jedynki do np.arraya
        for ind in region:
            result[ind] = True
        return result

    def kidney_segmentation(self):
        """
        Funkcja dokonuje segmentacji potencjalnych nerek na podstawie histogramu.
        """
        r = self.max_range_from_histogram # przedział wartości zmiennej szukanej na zdjęciu
        ind = self.index_of_value_in_given_range(r) # indeks początkowy dla algorytmu rozrostu obszaru w wersji lokalnej
        mask = region_growing_local(self.image, ind, 0.95*self.image[ind], 1.05*self.image[ind])

        return self.image[mask], mask # Zwraca zdjęcie potencjalnej nerki do dalszej analizy i maskę tej nerki.

    def tumour_segmentation(mask):
        """
        Funkcja dokonuje segmentacji guza na podstawie dostarczonej maski nerki.
        Zwraca zdjęcie i maskę guza.
        """
        maks = max(image) # Ta wartość wskazuje na obszar wykrytej nerki

        def horizontal_coherence(mask, ind):
            """
            Funkcja sprawdza czy w danym rzędzie maska jest ciągła.
            Jeżeli w rzędzie znajdują się same wartości False, ten rząd uznaje się za ciągły.
            """
            mask_row = list(mask[ind])
            if(not any(mask_row)): # Sprawdzenie, czy w rzędzie znajdują się same wartości false
                return True
            else:
                first_ind = mask_row.index(True)

                try:
                    second_ind = mask_row[first_ind:].index(False)
                except ValueError:
                    return True

                if(0 < mask_row[second_ind:].count(True)):
                    return False
                else:
                    return True

        ind = None
        for i in range(mask.shape[0]): # Znalezienie punktu nieciągłości w wierszu (ind)
            if(not horizontal_coherence(mask, i)):
                row_mask = list(mask[i])
                first_ind = row_mask.index(True)
                ind = row_mask[first_ind:].index(False)
                break

        mask = region_growing_local(self.image, ind, 0.95*self.image[ind], 1.05*self.image[ind])
        return self.image[mask], mask # Zwraca zdjęcie guza i jego maskę
