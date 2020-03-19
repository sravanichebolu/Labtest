import os
from cv2 import cv2
import numpy as np    
import math

def salt_and_pepper(img, pro): 
    sampimg = img   
    for k in range(sampimg.shape[0]):    
        for h in range(sampimg.shape[1]):   
            rand = np.random.random()
            if rand < (float)(pro/200):#假設胡椒鹽比例為十，則rand<0.05時產生白點
                sampimg[k,h,0]= 255    
                sampimg[k,h,1]= 255    
                sampimg[k,h,2]= 255    
            elif rand > (1-(float)(pro/200)):#假設胡椒鹽比例為十，則rand>0.95時產生黑點
                sampimg[k,h,0]= 0 
                sampimg[k,h,1]= 0    
                sampimg[k,h,2]= 0   
    return sampimg   

def sort(list):#insertion sort
    for i in range(1, len(list)): 
        tmp = list[i]
        j = i-1 
        while j >= 0 and tmp < list[j]:
            list[j+1] = list[j] 
            j = j-1
        list[j+1] = tmp
    mid = list[4]
    return mid

def median_filter(img):#暫時沒有想到處理邊緣鏡射比較好的寫法，只能用土法煉鋼的方法
    R_data = [0]*9
    G_data = [0]*9
    B_data = [0]*9
    (B,G,R) = cv2.split(img)
    for i in range(img.shape[0]):    
        for j in range(img.shape[1]): 
            if i>0 and i<(img.shape[0]-1) and j>0 and j<(img.shape[1]-1):
                d = 0
                for k in range(-1,2):#取得3*3 data
                    for h in range(-1,2):
                        B_data[d] = B[i+k,j+h]
                        G_data[d] = G[i+k,j+h]
                        R_data[d] = R[i+k,j+h]
                        d += 1 
                B[i,j] = sort(B_data)
                G[i,j] = sort(G_data)
                R[i,j] = sort(R_data)
            elif i==0 and j==0:#左上
                d=0
                for k in range(4):     
                    B_data[d] = B[i,j]
                    G_data[d] = G[i,j]
                    R_data[d] = R[i,j]
                    d += 1
                for k in range(2):     
                    B_data[d] = B[i,j+1]
                    G_data[d] = G[i,j+1]
                    R_data[d] = R[i,j+1]
                    d += 1
                for k in range(2):     
                    B_data[d] = B[i+1,j]
                    G_data[d] = G[i+1,j]
                    R_data[d] = R[i+1,j]
                    d += 1   
                B_data[d] = B[i+1,j+1]
                G_data[d] = G[i+1,j+1]
                R_data[d] = R[i+1,j+1] 
                B[i,j] = sort(B_data)
                G[i,j] = sort(G_data)
                R[i,j] = sort(R_data) 
            elif i==(img.shape[0]-1) and j==0: #左下
                d=0
                for k in range(4):     
                    B_data[d] = B[i,j]
                    G_data[d] = G[i,j]
                    R_data[d] = R[i,j]
                    d += 1
                for k in range(2):     
                    B_data[d] = B[i,j+1]
                    G_data[d] = G[i,j+1]
                    R_data[d] = R[i,j+1]
                    d += 1
                for k in range(2):     
                    B_data[d] = B[i-1,j]
                    G_data[d] = G[i-1,j]
                    R_data[d] = R[i-1,j]
                    d += 1   
                B_data[d] = B[i-1,j+1]
                G_data[d] = G[i-1,j+1]
                R_data[d] = R[i-1,j+1] 
                B[i,j] = sort(B_data)
                G[i,j] = sort(G_data)
                R[i,j] = sort(R_data)
            elif i==0 and j==(img.shape[1]-1): #右上
                d=0
                for k in range(4):     
                    B_data[d] = B[i,j]
                    G_data[d] = G[i,j]
                    R_data[d] = R[i,j]
                    d += 1
                for k in range(2):     
                    B_data[d] = B[i,j-1]
                    G_data[d] = G[i,j-1]
                    R_data[d] = R[i,j-1]
                    d += 1
                for k in range(2):     
                    B_data[d] = B[i+1,j]
                    G_data[d] = G[i+1,j]
                    R_data[d] = R[i+1,j]
                    d += 1   
                B_data[d] = B[i+1,j-1]
                G_data[d] = G[i+1,j-1]
                R_data[d] = R[i+1,j-1] 
                B[i,j] = sort(B_data)
                G[i,j] = sort(G_data)
                R[i,j] = sort(R_data) 
            elif i==(img.shape[0]-1) and j==(img.shape[1]-1): #右下
                d=0
                for k in range(4):     
                    B_data[d] = B[i,j]
                    G_data[d] = G[i,j]
                    R_data[d] = R[i,j]
                    d += 1
                for k in range(2):     
                    B_data[d] = B[i,j-1]
                    G_data[d] = G[i,j-1]
                    R_data[d] = R[i,j-1]
                    d += 1
                for k in range(2):     
                    B_data[d] = B[i-1,j]
                    G_data[d] = G[i-1,j]
                    R_data[d] = R[i-1,j]
                    d += 1   
                B_data[d] = B[i-1,j-1]
                G_data[d] = G[i-1,j-1]
                R_data[d] = R[i-1,j-1] 
                B[i,j] = sort(B_data)
                G[i,j] = sort(G_data)
                R[i,j] = sort(R_data) 
            elif i==0 and j<(img.shape[1]-1) and j>0:  #上邊
                d=0
                for k in range(2):     
                    B_data[d] = B[i,j]
                    G_data[d] = G[i,j]
                    R_data[d] = R[i,j]
                    d += 1
                for k in range(2):     
                    B_data[d] = B[i,j+1]
                    G_data[d] = G[i,j+1]
                    R_data[d] = R[i,j+1]
                    d += 1
                for k in range(2):     
                    B_data[d] = B[i,j-1]
                    G_data[d] = G[i,j-1]
                    R_data[d] = R[i,j-1]
                    d += 1   
                B_data[d] = B[i+1,j+1]
                G_data[d] = G[i+1,j+1]
                R_data[d] = R[i+1,j+1] 
                d += 1
                B_data[d] = B[i+1,j]
                G_data[d] = G[i+1,j]
                R_data[d] = R[i+1,j] 
                d += 1
                B_data[d] = B[i+1,j-1]
                G_data[d] = G[i+1,j-1]
                R_data[d] = R[i+1,j-1] 
                B[i,j] = sort(B_data)
                G[i,j] = sort(G_data)
                R[i,j] = sort(R_data)
            elif i==(img.shape[0]-1) and j<(img.shape[1]-1) and j>0:  #下邊
                d=0
                for k in range(2):     
                    B_data[d] = B[i,j]
                    G_data[d] = G[i,j]
                    R_data[d] = R[i,j]
                    d += 1
                for k in range(2):     
                    B_data[d] = B[i,j+1]
                    G_data[d] = G[i,j+1]
                    R_data[d] = R[i,j+1]
                    d += 1
                for k in range(2):     
                    B_data[d] = B[i,j-1]
                    G_data[d] = G[i,j-1]
                    R_data[d] = R[i,j-1]
                    d += 1   
                B_data[d] = B[i-1,j-1]
                G_data[d] = G[i-1,j-1]
                R_data[d] = R[i-1,j-1] 
                d += 1
                B_data[d] = B[i-1,j]
                G_data[d] = G[i-1,j]
                R_data[d] = R[i-1,j] 
                d += 1
                B_data[d] = B[i-1,j+1]
                G_data[d] = G[i-1,j+1]
                R_data[d] = R[i-1,j+1] 
                B[i,j] = sort(B_data)
                G[i,j] = sort(G_data)
                R[i,j] = sort(R_data)
            elif i>0 and i<(img.shape[0]-1) and j==0:  #左邊
                d=0
                for k in range(2):     
                    B_data[d] = B[i,j]
                    G_data[d] = G[i,j]
                    R_data[d] = R[i,j]
                    d += 1
                for k in range(2):     
                    B_data[d] = B[i+1,j]
                    G_data[d] = G[i+1,j]
                    R_data[d] = R[i+1,j]
                    d += 1
                for k in range(2):     
                    B_data[d] = B[i-1,j]
                    G_data[d] = G[i-1,j]
                    R_data[d] = R[i-1,j]
                    d += 1   
                B_data[d] = B[i+1,j+1]
                G_data[d] = G[i+1,j+1]
                R_data[d] = R[i+1,j+1] 
                d += 1
                B_data[d] = B[i,j+1]
                G_data[d] = G[i,j+1]
                R_data[d] = R[i,j+1] 
                d += 1
                B_data[d] = B[i-1,j+1]
                G_data[d] = G[i-1,j+1]
                R_data[d] = R[i-1,j+1] 
                B[i,j] = sort(B_data)
                G[i,j] = sort(G_data)
                R[i,j] = sort(R_data)   
            elif i<(img.shape[0]-1) and i>0 and j==(img.shape[1]-1) :  #右邊
                d=0
                for k in range(2):     
                    B_data[d] = B[i,j]
                    G_data[d] = G[i,j]
                    R_data[d] = R[i,j]
                    d += 1
                for k in range(2):     
                    B_data[d] = B[i+1,j]
                    G_data[d] = G[i+1,j]
                    R_data[d] = R[i+1,j]
                    d += 1
                for k in range(2):     
                    B_data[d] = B[i-1,j]
                    G_data[d] = G[i-1,j]
                    R_data[d] = R[i-1,j]
                    d += 1   
                B_data[d] = B[i-1,j-1]
                G_data[d] = G[i-1,j-1]
                R_data[d] = R[i-1,j-1] 
                d += 1
                B_data[d] = B[i,j-1]
                G_data[d] = G[i,j-1]
                R_data[d] = R[i,j-1] 
                d += 1
                B_data[d] = B[i+1,j-1]
                G_data[d] = G[i+1,j-1]
                R_data[d] = R[i+1,j-1] 
                B[i,j] = sort(B_data)
                G[i,j] = sort(G_data)
                R[i,j] = sort(R_data)                            
    merged = cv2.merge([B,G,R])            
    return merged

def mse(img_reg, img_noise):
    mse_ans = 0
    for i in range(img_reg.shape[0]):    
        for j in range(img_reg.shape[1]): 
            gray_reg = img_reg[i,j,0]*0.114 + img_reg[i,j,1]*0.587 + img_reg[i,j,2]*0.299#將RGB轉換為gray以便做PSNR計算
            gray_noise = img_noise[i,j,0]*0.114 + img_noise[i,j,1]*0.587 + img_noise[i,j,2]*0.299
            mse_ans += (gray_reg - gray_noise)**2
    return mse_ans/(img_reg.shape[0]*img_reg.shape[1])

def psnr(img_reg, img_noise):
    mse_ans = mse(img_reg, img_noise)
    psnr_ans = 10 * math.log10((255**2)/mse_ans)
    return psnr_ans

compare_image = cv2.imread("lena.jpg")#對image做胡椒雜訊後，即使是assign給新的變數，原本的image還是會被改道，故重新抓一次圖for PSNR
#10%胡椒鹽雜訊、median-filter、PSNR
image = cv2.imread("lena.jpg")
result_10 = salt_and_pepper(image, 10)
#cv2.imwrite("Salt&Pepper10%.jpg", result_10)
cv2.imshow("Salt&Pepper10%", result_10)
after_filter_10 = median_filter(result_10)
#cv2.imwrite("after_median_filter10%.jpg",after_filter_10)
cv2.imshow("after_median_filter10%",after_filter_10)
print('PSNR of 10% Salt&Pepper noise is : ',psnr(compare_image,result_10))
#20%胡椒鹽雜訊、median-filter、PSNR
image = cv2.imread("lena.jpg")
result_20 = salt_and_pepper(image, 20)
#cv2.imwrite("Salt&Pepper20%.jpg", result_20)
cv2.imshow("Salt&Pepper20%", result_20)
after_filter_20 = median_filter(result_20)
#cv2.imwrite("after_median_filter20%.jpg",after_filter_20)
cv2.imshow("after_median_filter20%",after_filter_20)
print('PSNR of 20% Salt&Pepper noise is : ',psnr(compare_image,result_20))
#30%胡椒鹽雜訊、median-filter、PSNR
image = cv2.imread("lena.jpg")
result_30 = salt_and_pepper(image, 30)
#cv2.imwrite("Salt&Pepper30%.jpg", result_30)
cv2.imshow("Salt&Pepper30%", result_30) 
after_filter_30 = median_filter(result_30)
#cv2.imwrite("after_median_filter30%.jpg",after_filter_30)
cv2.imshow("after_median_filter30%",after_filter_30)
print('PSNR of 30% Salt&Pepper noise is : ',psnr(compare_image,result_30))

cv2.waitKey(0) 
cv2.destroyAllWindows()