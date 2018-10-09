
import cv2
import numpy as np
import math


def image_pad(img_s):
    row=len(img_s)
    col=len(img_s[0])
    res=[[0 for x in range(col+6)] for y in range(row+6)]
    result=np.asarray(res)
    for i in range(3,row+3):
        for j in range(3,col+3):
            result[i][j]=img_s[i-3][j-3]

    return result

def Create_gaussian_kernel(size, sigma):
    center=(int)(size/2)
    kernel=np.zeros((size,size))
    for i in range(size):
       for j in range(size):
          diff=np.sqrt((i-center)**2+(j-center)**2)
          kernel[i,j]=np.exp(-(diff**2)/2*sigma**2)
    return kernel/np.sum(kernel)

def resize(img):
    rimage=img[0:len(img):2,0:len(img[0]):2]
    return rimage

def Multiplication(img, kernel):
    convolved_image=np.zeros(img.shape)
    for row in range(3, img.shape[0]-2):
        for column in range(3, img.shape[1]-3):
            sum = 0
            sum_pixels = 0
            for i in range(-3,3):
                for j in range(-3,3):
                    sum += img[row+i][column+j] * kernel[3+i][3+j]
                    sum_pixels += img[row+i][column+j]
            convolved_image[i][j] = sum / sum_pixels
    return convolved_image




def normalization(matrix):
    maximum=0
    minimum=matrix[1][1]
    pos_edge_x =[[0 for x in range(len(matrix[0]))] for y in range(len(matrix))] 
    
    for i in range(1,len(matrix)):
        for j in range(1,len(matrix[0])):
            if matrix[i][j]>maximum:
                maximum=matrix[i][j]

    for i in range(1,len(matrix)):
        for j in range(1,len(matrix[0])):
            if matrix[i][j]<minimum:      
                minimum=matrix[i][j]
                
    for i in range(1,len(matrix)):
        for j in range(1,len(matrix[0])):
            pos_edge_x[i][j] = ((matrix[i][j] - minimum)*(255) / (maximum - minimum))
    

    return(pos_edge_x)

def subtract(A,B):

    row_a_size = len(A)
    col_a_size = len(A[0]) 
    row_b_size = len(B)
    col_b_size = len(B[0])
    for i in range(0, row_b_size):
        for j in range(0, col_b_size):
            A[i][j] = A[i][j] - B[i][j]
    return A






root_2=((2)**0.5)

total=0.0
img=cv2.imread('/Users/kamallala/Downloads/proj1_cse573-3/task2.jpg',0)
img0=np.asarray(img)
rows=len(img0)
cols=len(img0[0])
image_matrix=image_pad(img0)
image_matrix=np.asarray(image_matrix)

'''def normalization(A):
    row_a_size = len(A)
    col_a_size = len(A[0]) 
    max_x = 0
    min_x = A[0][0]
    for i in range(0, row_a_size):
        max_x = max(max(A[i]), max_x)
        min_x = min(min(A[i]), min_x)

    for i in range(0, row_a_size):
        for j in range(0, col_a_size):
            A[i][j] = (A[i][j] - min_x )*255/(max_x - min_x)
    return A'''


    







Octave_1=np.asarray(resize(img0))
Result_Octave_1=np.asarray(image_pad(Octave_1))


Octave_2=np.asarray(resize(Octave_1))
Result_Octave_2=np.asarray(image_pad(Octave_2))

Octave_3=np.asarray(resize(Octave_2))
Result_Octave_3=np.asarray(image_pad(Octave_3))

Octave_4=np.asarray(resize(Octave_3))
Result_Octave_4=np.asarray(image_pad(Octave_4))


#1st Gaussian
gaussian_kernel_value_1=Create_gaussian_kernel(7,1/root_2)
gaussian_1_Multiplied_value=np.asarray(Multiplication(Result_Octave_1, gaussian_kernel_value_1))
gaussian_1_normalized=np.asarray(normalization(gaussian_1_Multiplied_value))



#2nd Gaussian
gaussian_kernel_value_2=Create_gaussian_kernel(7,1)
gaussian_2_Multiplied_value=np.asarray(Multiplication(Result_Octave_1, gaussian_kernel_value_2))
gaussian_2_normalized=np.asarray(normalization(gaussian_2_Multiplied_value))
DOG_1=subtract(gaussian_1_normalized,gaussian_2_normalized)

#3rd Gaussian
gaussian_kernel_value_3=Create_gaussian_kernel(7,root_2)
gaussian_3_Multiplied_value=np.asarray(Multiplication(Result_Octave_1, gaussian_kernel_value_3))
gaussian_3_normalized=np.asarray(normalization(gaussian_3_Multiplied_value))
DOG_2=subtract(gaussian_2_normalized,gaussian_3_normalized)


#4th Gaussian
gaussian_kernel_value_4=Create_gaussian_kernel(7,2)
gaussian_4_Multiplied_value=np.asarray(Multiplication(Result_Octave_1, gaussian_kernel_value_4))
gaussian_4_normalized=np.asarray(normalization(gaussian_4_Multiplied_value))

DOG_3=subtract(gaussian_3_normalized,gaussian_4_normalized)



#5th Gaussian
gaussian_kernel_value_5=Create_gaussian_kernel(7,2*root_2)
gaussian_5_Multiplied_value=np.asarray(Multiplication(Result_Octave_1, gaussian_kernel_value_5))
gaussian_5_normalized=np.asarray(normalization(gaussian_5_Multiplied_value))
DOG_4=subtract(gaussian_4_normalized,gaussian_5_normalized)



#6th Gaussiam
gaussian_kernel_value_6=Create_gaussian_kernel(7,root_2)
gaussian_6_Multiplied_value=np.asarray(Multiplication(Result_Octave_2, gaussian_kernel_value_6))
gaussian_6_normalized=np.asarray(normalization(gaussian_6_Multiplied_value))
#DOG_5=subtract(gaussian_5_normalized,gaussian_6_normalized)



#7th Gaussian
gaussian_kernel_value_7=Create_gaussian_kernel(7,2)
gaussian_7_Multiplied_value=np.asarray(Multiplication(Result_Octave_2, gaussian_kernel_value_7))
gaussian_7_normalized=np.asarray(normalization(gaussian_7_Multiplied_value))
DOG_5=subtract(gaussian_6_normalized,gaussian_7_normalized)



#8th Gaussian
gaussian_kernel_value_8=Create_gaussian_kernel(7,2*root_2)
gaussian_8_Multiplied_value=np.asarray(Multiplication(Result_Octave_2, gaussian_kernel_value_8))
gaussian_8_normalized=np.asarray(normalization(gaussian_8_Multiplied_value))
DOG_6=subtract(gaussian_7_normalized,gaussian_8_normalized)



#9th Gaussian

gaussian_kernel_value_9=Create_gaussian_kernel(7,4)
gaussian_9_Multiplied_value=np.asarray(Multiplication(Result_Octave_2, gaussian_kernel_value_9))
gaussian_9_normalized=np.asarray(normalization(gaussian_9_Multiplied_value))
DOG_7=subtract(gaussian_8_normalized,gaussian_9_normalized)


print("idhar 9")

#10th Gaussian

gaussian_kernel_value_10=Create_gaussian_kernel(7,4*root_2)
gaussian_10_Multiplied_value=np.asarray(Multiplication(Result_Octave_2, gaussian_kernel_value_10))
gaussian_10_normalized=np.asarray(normalization(gaussian_10_Multiplied_value))
DOG_8=subtract(gaussian_9_normalized,gaussian_10_normalized)




#11th Gaussian

gaussian_kernel_value_11=Create_gaussian_kernel(7,2*root_2)
gaussian_11_Multiplied_value=np.asarray(Multiplication(Result_Octave_3, gaussian_kernel_value_11))
gaussian_11_normalized=np.asarray(normalization(gaussian_11_Multiplied_value))
#DOG_10=subtract(gaussian_10_normalized,gaussian_11_normalized)




#12th Gaussian

gaussian_kernel_value_12=Create_gaussian_kernel(7,4)
gaussian_12_Multiplied_value=np.asarray(Multiplication(Result_Octave_3, gaussian_kernel_value_12))
gaussian_12_normalized=np.asarray(normalization(gaussian_12_Multiplied_value))
DOG_9=subtract(gaussian_11_normalized,gaussian_12_normalized)




#13th Gaussian

gaussian_kernel_value_13=Create_gaussian_kernel(7,4*root_2)
gaussian_13_Multiplied_value=np.asarray(Multiplication(Result_Octave_3, gaussian_kernel_value_13))
gaussian_13_normalized=np.asarray(normalization(gaussian_13_Multiplied_value))
DOG_10=subtract(gaussian_12_normalized,gaussian_13_normalized)




#14th Gaussian

gaussian_kernel_value_14=Create_gaussian_kernel(7,8)
gaussian_14_Multiplied_value=np.asarray(Multiplication(Result_Octave_3, gaussian_kernel_value_14))
gaussian_14_normalized=np.asarray(normalization(gaussian_14_Multiplied_value))
DOG_11=subtract(gaussian_13_normalized,gaussian_14_normalized)
cv2.imshow('ss',DOG_11)




#15th Gaussian

gaussian_kernel_value_15=Create_gaussian_kernel(7,8*root_2)
gaussian_15_Multiplied_value=np.asarray(Multiplication(Result_Octave_3, gaussian_kernel_value_15))
gaussian_15_normalized=np.asarray(normalization(gaussian_15_Multiplied_value))
DOG_12=subtract(gaussian_14_normalized,gaussian_15_normalized)




#16th Gaussian

gaussian_kernel_value_16=Create_gaussian_kernel(7,4*root_2)
gaussian_16_Multiplied_value=np.asarray(Multiplication(Result_Octave_4, gaussian_kernel_value_16))
gaussian_16_normalized=np.asarray(normalization(gaussian_16_Multiplied_value))
#DOG_15=subtract(gaussian_15_normalized,gaussian_16_normalized)


print("idhar 16")


#17th Gaussian

gaussian_kernel_value_17=Create_gaussian_kernel(7,8)
gaussian_17_Multiplied_value=np.asarray(Multiplication(Result_Octave_4, gaussian_kernel_value_17))
gaussian_17_normalized=np.asarray(normalization(gaussian_17_Multiplied_value))
DOG_13=subtract(gaussian_16_normalized,gaussian_17_normalized)




#18th Gaussian

gaussian_kernel_value_18=Create_gaussian_kernel(7,8*root_2)
gaussian_18_Multiplied_value=np.asarray(Multiplication(Result_Octave_4, gaussian_kernel_value_18))
gaussian_18_normalized=np.asarray(normalization(gaussian_18_Multiplied_value))
DOG_14=subtract(gaussian_17_normalized,gaussian_18_normalized)




#19th Gaussian

gaussian_kernel_value_19=Create_gaussian_kernel(7,16)
gaussian_19_Multiplied_value=np.asarray(Multiplication(Result_Octave_4, gaussian_kernel_value_19))
gaussian_19_normalized=np.asarray(normalization(gaussian_19_Multiplied_value))
DOG_15=subtract(gaussian_18_normalized,gaussian_19_normalized)




#20th Gaussian

gaussian_kernel_value_20=Create_gaussian_kernel(7,16*root_2)
gaussian_20_Multiplied_value=np.asarray(Multiplication(Result_Octave_4, gaussian_kernel_value_20))
gaussian_20_normalized=np.asarray(normalization(gaussian_20_Multiplied_value))
DOG_16=subtract(gaussian_19_normalized,gaussian_20_normalized)




row2=len(DOG_6)
column2=len(DOG_6[0])
KEYPOINTS_1_3=np.array([[0 for i in range(column2)] for j in range(row2)])

for i in range(1,(row2-1)):
    for j in range(1,(column2-1)):
        if DOG_6[i][j] == max(DOG_6[i-1][j-1], DOG_6[i-1][j], DOG_6[i-1][j+1], DOG_6[i][j-1], DOG_6[i][j], DOG_6[i][j+1], DOG_6[i+1][j-1], DOG_6[i+1][j], DOG_6[i+1][j+1]):
            if DOG_6[i,j] > max(DOG_5[i-1,j-1], DOG_5[i-1,j], DOG_5[i-1,j+1], DOG_5[i,j-1], DOG_5[i,j], DOG_5[i,j+1], DOG_5[i+1,j-1], DOG_5[i+1,j], DOG_5[i+1,j+1]):
                if DOG_6[i,j] > max(DOG_7[i-1][j-1], DOG_7[i-1][j], DOG_7[i-1][j+1], DOG_7[i][j-1], DOG_7[i][j], DOG_7[i][j+1], DOG_7[i+1][j-1], DOG_7[i+1][j], DOG_7[i+1][j+1]):
                    KEYPOINTS_1_3[i][j] = 255
                else:
                    continue

        else: 
            if DOG_6[i,j] == min(DOG_6[i-1][j-1], DOG_6[i-1][j], DOG_6[i-1][j+1], DOG_6[i][j-1], DOG_6[i][j], DOG_6[i][j+1], DOG_6[i+1][j-1], DOG_6[i+1][j], DOG_6[i+1][j+1]):
                if DOG_6[i,j] < min(DOG_5[i-1,j-1], DOG_5[i-1,j], DOG_5[i-1,j+1], DOG_5[i,j-1], DOG_5[i,j], DOG_5[i,j+1], DOG_5[i+1,j-1], DOG_5[i+1,j], DOG_5[i+1,j+1]):
                    if DOG_6[i,j] < min(DOG_7[i-1][j-1], DOG_7[i-1][j], DOG_7[i-1][j+1], DOG_7[i][j-1], DOG_7[i][j], DOG_7[i][j+1], DOG_7[i+1][j-1], DOG_7[i+1][j], DOG_7[i+1][j+1]):
                        KEYPOINTS_1_3[i][j] = 0


KEYPOINTS_1_3=np.asarray(normalization(KEYPOINTS_1_3))
cv2.imshow('KEYPOINTS_1_3',KEYPOINTS_1_3)                   

row3=len(DOG_7)
column3=len(DOG_7[0])
KEYPOINTS_2_4= np.array([[0 for i in range(column3)] for j in range(row3)])

for i in range(1,(row3-1)):
    for j in range(1,(column3-1)):
        if DOG_7[i][j] == max(DOG_7[i-1][j-1], DOG_7[i-1][j], DOG_7[i-1][j+1], DOG_7[i][j-1], DOG_7[i][j], DOG_7[i][j+1], DOG_7[i+1][j-1], DOG_7[i+1][j], DOG_7[i+1][j+1]):
            if DOG_7[i][j] > max(DOG_6[i-1][j-1], DOG_6[i-1][j], DOG_6[i-1][j+1], DOG_6[i][j-1], DOG_6[i][j], DOG_6[i][j+1], DOG_6[i+1][j-1], DOG_6[i+1][j], DOG_6[i+1][j+1]):
                if DOG_7[i][j] > max(DOG_8[i-1][j-1], DOG_8[i-1][j], DOG_8[i-1][j+1], DOG_8[i][j-1], DOG_8[i][j], DOG_8[i][j+1], DOG_8[i+1][j-1], DOG_8[i+1][j], DOG_8[i+1][j+1]):
                    KEYPOINTS_2_4[i][j] = 255
                else:
                    continue

        else:
             if DOG_7[i][j] == min(DOG_7[i-1][j-1], DOG_7[i-1][j], DOG_7[i-1][j+1], DOG_7[i][j-1], DOG_7[i][j], DOG_7[i][j+1], DOG_7[i+1][j-1], DOG_7[i+1][j], DOG_7[i+1][j+1]):
                if DOG_7[i][j] < min(DOG_6[i-1][j-1], DOG_6[i-1][j], DOG_6[i-1][j+1],DOG_6[i][j-1],DOG_6[i][j], DOG_6[i][j+1], DOG_6[i+1][j-1],DOG_6[i+1][j], DOG_6[i+1][j+1]):
                    if DOG_7[i][j] < min(DOG_8[i-1][j-1], DOG_8[i-1][j], DOG_8[i-1][j+1],DOG_8[i][j-1], DOG_8[i][j], DOG_8[i][j+1], DOG_8[i+1][j-1], DOG_8[i+1][j], DOG_8[i+1][j+1]):
                        KEYPOINTS_2_4[i][j] = 0

KEYPOINTS_2_4=np.asarray(normalization(KEYPOINTS_2_4))
cv2.imshow('KEYPOINTS_2_4',KEYPOINTS_2_4) 

#octave 3

row4=len(DOG_10)
column4=len(DOG_10[0])
FINAL_KEYPOINTS=np.array([[0 for i in range(column4)] for j in range(row4)])

for i in range(1,(row4-1)):
    for j in range(1,(column4-1)):
        if DOG_10[i][j] == max(DOG_10[i-1][j-1], DOG_10[i-1][j], DOG_10[i-1][j+1], DOG_10[i][j-1], DOG_10[i][j], DOG_10[i][j+1], DOG_10[i+1][j-1], DOG_10[i+1][j], DOG_10[i+1][j+1]):
            if DOG_10[i][j] > max(DOG_9[i-1][j-1], DOG_9[i-1][j], DOG_9[i-1][j+1], DOG_9[i][j-1], DOG_9[i][j], DOG_9[i][j+1], DOG_9[i+1][j-1], DOG_9[i+1][j], DOG_9[i+1][j+1]):
                if DOG_10[i][j] > max(DOG_11[i-1][j-1], DOG_11[i-1][j], DOG_11[i-1][j+1], DOG_11[i][j-1], DOG_11[i][j], DOG_11[i][j+1], DOG_11[i+1][j-1], DOG_11[i+1][j], DOG_11[i+1][j+1]):
                    FINAL_KEYPOINTS[i][j] = 255
                else:
                    continue

        else: 
            if DOG_10[i][j] == min(DOG_10[i-1][j-1], DOG_10[i-1][j], DOG_10[i-1][j+1], DOG_10[i][j-1], DOG_10[i][j], DOG_10[i][j+1], DOG_10[i+1][j-1], DOG_10[i+1][j], DOG_10[i+1][j+1]):
                if DOG_10[i][j] < min(DOG_9[i-1][j-1], DOG_9[i-1][j], DOG_9[i-1][j+1], DOG_9[i][j-1], DOG_9[i][j], DOG_9[i][j+1], DOG_9[i+1][j-1], DOG_9[i+1][j], DOG_9[i+1][j+1]):
                    if DOG_6[i][j] < min(DOG_7[i-1][j-1], DOG_7[i-1][j], DOG_7[i-1][j+1],DOG_7[i][j-1], DOG_7[i][j], DOG_7[i][j+1], DOG_7[i+1][j-1], DOG_7[i+1][j], DOG_7[i+1][j+1]):
                        KEYPOINTS_1_3[i][j] = 0


FINAL_KEYPOINTS=np.asarray(normalization(FINAL_KEYPOINTS))

cv2.namedWindow('FINAL_KEYPOINTS', cv2.WINDOW_NORMAL)
cv2.imshow('FINAL_KEYPOINTS',FINAL_KEYPOINTS)   
cv2.waitKey(0)

row5=len(DOG_11)
column5=len(DOG_11[0])
FINAL_KEYPOINTS_2= np.array([[0 for i in range(column3)] for j in range(row3)])

for i in range(1,(row5-1)):
    for j in range(1,(column5-1)):
        if DOG_11[i][j] == max(DOG_11[i-1][j-1], DOG_11[i-1][j], DOG_11[i-1][j+1], DOG_11[i][j-1], DOG_11[i][j], DOG_11[i][j+1], DOG_11[i+1][j-1], DOG_11[i+1][j], DOG_11[i+1][j+1]):
            if DOG_11[i][j] > max(DOG_10[i-1][j-1], DOG_10[i-1][j], DOG_10[i-1][j+1], DOG_10[i][j-1], DOG_10[i][j], DOG_10[i][j+1], DOG_10[i+1][j-1], DOG_10[i+1][j], DOG_10[i+1][j+1]):
                if DOG_11[i][j] > max(DOG_12[i-1][j-1], DOG_12[i-1][j], DOG_12[i-1][j+1], DOG_12[i][j-1], DOG_12[i][j], DOG_12[i][j+1], DOG_12[i+1][j-1], DOG_12[i+1][j], DOG_12[i+1][j+1]):
                    FINAL_KEYPOINTS_2[i][j] = 255
                else:
                    continue

        else:
             if DOG_11[i][j] == min(DOG_11[i-1][j-1], DOG_11[i-1][j], DOG_11[i-1][j+1], DOG_11[i][j-1], DOG_11[i][j], DOG_11[i][j+1], DOG_11[i+1][j-1], DOG_11[i+1][j], DOG_11[i+1][j+1]):
                if DOG_11[i][j] < min(DOG_10[i-1][j-1], DOG_10[i-1][j], DOG_10[i-1][j+1], DOG_10[i][j-1], DOG_10[i][j], DOG_10[i][j+1], DOG_10[i+1][j-1], DOG_10[i+1][j], DOG_10[i+1][j+1]):
                    if DOG_7[i][j] < min(DOG_12[i-1][j-1], DOG_12[i-1][j], DOG_12[i-1][j+1], DOG_12[i][j-1], DOG_12[i][j], DOG_12[i][j+1], DOG_12[i+1][j-1], DOG_12[i+1][j], DOG_12[i+1][j+1]):
                        FINAL_KEYPOINTS_2[i][j] = 0

FINAL_KEYPOINTS_2=np.asarray(normalization(FINAL_KEYPOINTS_2))
import ipdb
ipdb.set_trace()
cv2.namedWindow('FINAL_KEYPOINTS_2', cv2.WINDOW_NORMAL)
cv2.imshow('FINAL_KEYPOINTS_2',FINAL_KEYPOINTS_2) 
cv2.waitKey(0)
cv2.destroyAllWindows()

