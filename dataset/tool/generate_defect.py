import cv2  
import numpy as np 
import random,math,os,shutil


def random_rect(num,img_size,rect_size_h,rect_size_w):
    height,width = img_size
    ratio_lb_h,ratio_hb_h = rect_size_h
    ratio_lb_w,ratio_hb_w = rect_size_w

    # print height,width

    rect_list = []
    for i in range(num):

        ratio_h = random.uniform(ratio_lb_h,ratio_hb_h)
        ratio_w = random.uniform(ratio_lb_w,ratio_hb_w)

        rect_h = int(height * ratio_h)
        rect_w = int(width * ratio_w)

        x_start = random.randint(0,width-rect_w-1)
        y_start = random.randint(0,height-rect_h-1)

        rect_list.append([x_start,y_start,rect_h,rect_w])

    return rect_list

def convex_hull_generate(img,rect,points_num):
    x_start,y_start,height,width = rect


    points_list = []

    for i in range(points_num):

        ratio_h = random.random()
        ratio_w = random.random()

        rect_h = int(height * ratio_h)
        rect_w = int(width * ratio_w)

        points_list.append((x_start+rect_w,y_start+rect_h))

    # for i in range(len(points_list)):
    #     cv2.circle(img,points_list[i],10,255)

    hull = cv2.convexHull(np.array(points_list),returnPoints =True)
    
    mask = np.zeros_like(img)

    cv2.drawContours(mask,[hull],0,255,-1)

    pixelpoints = np.nonzero(mask)

    # cv2.imshow('hull',mask)

    return pixelpoints,hull



# img = np.zeros_like(img)
# rect_list = random_rect(10,img.shape,(0.1,0.12),(0.2,0.3))
# print rect_list,img.shape
# cv2.rectangle(img,(rect_list[0][0],rect_list[0][1]),(rect_list[0][0]+rect_list[0][3],rect_list[0][1]+rect_list[0][2]),255)
# convex_hull_generate(img,rect_list[0],5)

def generate_blur(img,label_img,num,ratio_h,ratio_w):
    rect_list = random_rect(num,img.shape,ratio_h,ratio_w)
    ksize = 20
    handle_img = np.copy(img)
    rect = rect_list[0]
    x,y,h,w = rect
    roi = handle_img[y:y+h,x:x+w]
    handle_img[y:y+h,x:x+w] = cv2.blur(roi,(ksize,ksize))

    pixelpoints,_ = convex_hull_generate(img,rect,10)

    label_img[pixelpoints] = 255
    
    # print pixelpoints
    img[pixelpoints] = handle_img[pixelpoints] 

def generate_crack(img,label_img,num,ratio_area,aspect,length_ratio):
    ratio_area_l,ratio_area_h = ratio_area
    length_ratio_l,length_ratio_h = length_ratio
    area_l = img.shape[0]*img.shape[1]*ratio_area_l
    area_h = img.shape[0]*img.shape[1]*ratio_area_h
    length_l = min( img.shape[0],img.shape[1]) * length_ratio_l
    length_h = min( img.shape[0],img.shape[1]) * length_ratio_h

    count = 0
    while True:
        rect_list = random_rect(1,img.shape,(0,1),(0,1))

        handle_img = np.copy(img)

        rect = rect_list[0]
        x,y,h,w = rect
        # roi = handle_img[y:y+h,x:x+w]
        handle_img[y:y+h,x:x+w] = 0

        pixelpoints,hull = convex_hull_generate(img,rect,10)
        w_,h_ = cv2.minAreaRect(hull)[1]
        # print w_,h_
        if w_ != 0 and h_ != 0:
            if (cv2.contourArea(hull) > area_l and  cv2.contourArea(hull) < area_h) and \
              (w_/h_ > aspect or h_/w_ >aspect) and  max(w_,h_) < length_h and min(w_,h_) > length_l:
                img[pixelpoints] = handle_img[pixelpoints] 
                label_img[pixelpoints] = 255
                count += 1
                if count >= num:
                    break


def generate_scratch(img,label_img,num,ratio_area,aspect,length_ratio):

    ratio_area_l,ratio_area_h = ratio_area
    length_ratio_l,length_ratio_h = length_ratio
    area_l = img.shape[0]*img.shape[1]*ratio_area_l
    area_h = img.shape[0]*img.shape[1]*ratio_area_h
    length_l = min( img.shape[0],img.shape[1]) * length_ratio_l
    length_h = min( img.shape[0],img.shape[1]) * length_ratio_h


    count = 0
    while True:
        rect_list = random_rect(1,img.shape,(0,1),(0,1))

        handle_img = np.copy(img)

        rect = rect_list[0]
        x,y,h,w = rect
        # roi = handle_img[y:y+h,x:x+w]
        handle_img[y:y+h,x:x+w] = 50

        pixelpoints,hull = convex_hull_generate(img,rect,10)
        w_,h_ = cv2.minAreaRect(hull)[1]

        # print w_,h_,length_h,length_l,cv2.contourArea(hull),area_h,area_l

        if w_ != 0 and h_ != 0:
            if (cv2.contourArea(hull) > area_l and  cv2.contourArea(hull) < area_h ) and  \
                    (w_/h_ > aspect or h_/w_ >aspect) and max(w_,h_) < length_h and min(w_,h_) > length_l:
                img[pixelpoints] = handle_img[pixelpoints] 
                label_img[pixelpoints] = 255
                count += 1
                if count >= num:
                    break

def generate_spot(img,label_img,num,ratio_area,aspect):

    ratio_area_l,ratio_area_h = ratio_area
    area_l = img.shape[0]*img.shape[1]*ratio_area_l
    area_h = img.shape[0]*img.shape[1]*ratio_area_h

    count = 0
    while True:
        rect_list = random_rect(1,img.shape,(0,1),(0,1))

        handle_img = np.copy(img)

        rect = rect_list[0]
        x,y,h,w = rect
        # roi = handle_img[y:y+h,x:x+w]
        handle_img[y:y+h,x:x+w] = 255

        pixelpoints,hull = convex_hull_generate(img,rect,10)
        w_,h_ = cv2.minAreaRect(hull)[1]
        if w_ != 0 and h_ != 0:
            if (cv2.contourArea(hull) > area_l and  cv2.contourArea(hull) < area_h) and  max(w_/h_,h_/w_) < aspect: #and max(w_,h_)>length:
                img[pixelpoints] = handle_img[pixelpoints]
                label_img[pixelpoints] = 255 
                count += 1
                if count >= num:
                    break

def blur(img,label_img):
    generate_blur(img,label_img,1,(0.05,0.1),(0.05,0.1))
def scratch(img,label_img):
    generate_scratch(img,label_img,1,(0.0001,0.002),5,(0.005,0.4))
def spot(img,label_img):
    generate_spot(img,label_img,1,(0.0001,0.0005),2)

def generate_defect_img(img,min_num,max_num,label_img):

    # label_img = np.zeros_like(img)

    # if random.random > 0.9:
    #     generate_crack(img,label_img,1,(0.01,0.05),6,(0.1,0.8))

    #method_list = [blur,scratch,spot]
    method_list = [blur,scratch,spot]


    num = random.randint(min_num,max_num)
    print num


    for i in range(num):
        fun_index = random.randint(0,len(method_list)-1)

        method_list[fun_index](img,label_img)

        # generate_blur(img,1,(0.05,0.3),(0.05,0.3))
        
        # generate_scratch(img,1,(0.001,0.05),20,(0.01,0.4))
        # generate_spot(img,1,(0.001,0.008),1.5)
    #return label_img

def dataset_generate(root,save_dir):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    
    os.makedirs(save_dir)


    files = os.listdir(root)
    for file in files:
        img = cv2.imread(os.path.join(root,file),cv2.cv.CV_LOAD_IMAGE_GRAYSCALE)
        print img.shape
        label = np.zeros((img.shape[0],img.shape[1]),dtype=np.uint8)
        print "label,",label.shape
        #cv2.imshow('img',img)
        for i in range(16):
            img_final = img[512*(i//8):512*(i//8+1),512*(i%8):512*(i%8+1)]
            label_img = label[512*(i//8):512*(i//8+1),512*(i%8):512*(i%8+1)]
            print img_final.shape,label_img.shape

            generate_defect_img(img_final,1,5,label_img)

        if not os.path.exists(save_dir+'/img'):
            os.makedirs(save_dir+'/img')
        if not os.path.exists(save_dir+'/label'):
            os.makedirs(save_dir+'/label')

        cv2.imwrite(save_dir + '/img/'+file,img)
        cv2.imwrite(save_dir+ '/label/'+file,label)


    

def main():
    root = './resize_4096'
    save_dir = './generate_defect1_again'



    dataset_generate(root,save_dir)

    # cv2.imshow("img", img)

    # cv2.waitKey(0)

# def main():
#     img = cv2.imread('./def_1.png',cv2.cv.CV_LOAD_IMAGE_GRAYSCALE)
#     print img.shape

#     cv2.namedWindow('label',cv2.WINDOW_NORMAL)
#     cv2.namedWindow('img',cv2.WINDOW_NORMAL)

#     # label = generate_defect_img(img,1,5)
#     label_img = np.zeros_like(img)

#     #generate_blur(img,label_img,1,(0.05,0.1),(0.05,0.1))

#     #generate_scratch(img,label_img,1,(0.0001,0.002),5,(0.005,0.4))

#     #generate_spot(img,label_img,1,(0.0001,0.0005),2)


#     cv2.imshow("label", label_img)
#     cv2.imshow("img", img)
#     cv2.waitKey(0)


if __name__ == '__main__':
    main()





 
