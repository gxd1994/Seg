import cv2,os

def resize_img(img_size):
    path = './class1_def_img'
    dst  = './resize_4096'
    if not os.path.exists(dst):
        os.makedirs(dst)
    files = os.listdir(path)
    for file in files:
    	img = cv2.imread(os.path.join(path,file))
        print img.shape
    	img = cv2.resize(img,img_size)
        for i in range(4):
            name = os.path.splitext(file)[0]+'_%d'%i+os.path.splitext(file)[1]
            cv2.imwrite(os.path.join(dst,name),img[1024*i:1024*(i+1),:])



def main():
    resize_img((4096,4096))

if __name__ == '__main__':
    main()