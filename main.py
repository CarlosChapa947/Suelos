import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(imgpath: str):
    # Use a breakpoint in the code line below to debug your script.
    img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    #cv2.imshow("img2", img2)
    h2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
    #cv2.imshow("img", img)
    #cv2.imshow("img2", h2)
    img3 = img.flatten()  # fusionamos todas las sublistas(RGB, GRAY, BGR) a un vector unidimensional

    plt.hist(img3, 256, (0, 256), color='b')
    plt.xlim([0, 256])
    plt.title('Histograma')
    plt.show()

    res = cv2.equalizeHist(img, img2)
    res2 = cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)
    #cv2.imshow("img3", res)

    plt.hist(res.flatten(), 256, [0, 256], color='g')
    plt.xlim([0, 256])
    plt.title('Histograma Equalizado')
    plt.show()

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(20, 20))
    cl1 = clahe.apply(img)
    imgeq = cv2.cvtColor(cl1, cv2.COLOR_GRAY2RGB)
    #cv2.imshow("img4", imgeq)

    img4 = imgeq.flatten() 

    """plt.hist(img4, 256, [0, 256], color='r')  # (solo funciona con escala de grises para ver la intensidad)
    plt.xlim([0, 256])
    plt.title('Histograma')
    plt.show()"""

    hist, bins = np.histogram(img4, 256, (0, 256))
    value = np.argmax(hist)
    th3, dst3 = cv2.threshold(imgeq, value, 255, cv2.THRESH_BINARY)
    cv2.imshow("Result", dst3)

    #plt.imshow(imgeq)

    cv2.waitKey(0)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("1")
    print_hi("./Images/suelo1.png")
    print("2")
    print_hi("./Images/suelo2.png")
    print("3")
    print_hi("./Images/suelo3.png")
    print("4")
    print_hi("./Images/suelo4.png")
    print("5")
    print_hi("./Images/suelo5.png")



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
