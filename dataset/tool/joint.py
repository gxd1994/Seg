import cv2,os
import numpy as  np

root = './generate_defect1'
save_root = './joint_defect1'

files = os.listdir(root+'/img')
if files is None:
    print 'files is None'

count= 0
for i in range(0,len(files),16):
    result_img = np.zeros((1024,4096,3),dtype = np.uint8)
    result_label = np.zeros((1024,4096),dtype= np.uint8)

    for j in range(16):

        img = cv2.imread(os.path.join(root,'img',files[i+j]))
        label = cv2.imread(os.path.join(root,'label',files[i+j]),cv2.CV_LOAD_IMAGE_UNCHANGED)


        result_img[512*(j//8):512*(j//8+1),512*(j%8):512*(j%8+1),:] = img

        result_label[512*(j//8):512*(j//8+1),512*(j%8):512*(j%8+1)] = label

    count += 1

    cv2.imwrite(os.path.join(save_root,'img',"%d.png"%count),result_img)
    cv2.imwrite(os.path.join(save_root,'label',"%d.png"%count),result_label)




