import cv2
import numpy as np

img0= cv2.imread('/Users/kamallala/Downloads/proj1_cse573-3/task2.jpg/task1.png', 0)
img=np.asarray(img0)

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
            pos_edge_x[i][j] = ((matrix[i][j] - minimum) / (maximum - minimum))
    

    return(np.asarray(pos_edge_x))


def soble_x(imgs):
    col=len(imgs[0])
    row=len(imgs)

    

    #return(imgs)

    z=[[0 for x in range(col)] for y in range(row)]
    for i in range(1, img0.shape[0]-1):
        for j in range(1, img0.shape[1]-1):
            z1=imgs[i-1][j-1]* -1 + imgs[i-1][j]* 0+ imgs[i+1][j+1]* 1+imgs[i][j-1]* -2+imgs[i][j]* 0+imgs[i-1][j+1]* 2+imgs[i+1][j-1]* -1+imgs[i+1][j]* 0+imgs[i+1][j+1]* 1
            z[i][j]=z1
    return(np.asarray(z))


soble_x_array=soble_x(img)
print(soble_x_array)
pos_edge_x = normalization(soble_x_array)
cv2.imshow('pos_edge_x_dir', pos_edge_x)


def soble_y(imgs):
    col=len(imgs[0])
    row=len(imgs)

    
    z=[[0 for x in range(col)] for y in range(row)]
    for i in range(1, img0.shape[0]-1):
        for j in range(1, img0.shape[1]-1):
            z1=img0[i-1][j-1]* -1+img0[i-1][j]* -2+img0[i+1][j+1]* -1+img0[i][j-1]* 0+img0[i][j]* 0+img0[i-1][j+1]* 0+img0[i+1][j-1]* 1+img0[i+1][j]* 2+img0[i+1][j+1]* 1

            z[i][j]=z1
            
    return(np.asarray(z))

soble_y_array=soble_y(img0)
print(soble_y_array)
pos_edge_y = np.asarray(normalization(soble_y_array))
cv2.namedWindow('pos_edge_y_dir', cv2.WINDOW_NORMAL)
cv2.imshow('pos_edge_y_dir', pos_edge_y)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''cv2.namedWindow('sobel_y',cv2.WINDOW_NORMAL)
cv2.imshow('final_edge of y',soble_y_array)
cv2.waitKey(0)
cv2.destroyAllWindows()'''





#res=np.asarray(soble_x(img))

#print(res)


