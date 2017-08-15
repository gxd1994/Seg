import cv2,os
import numpy as np


def generate_detecton_label(label_root,stride,save_root):
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    files = os.listdir(label_root)
    for file in files:
        label = cv2.imread(os.path.join(label_root,file),cv2.CV_LOAD_IMAGE_UNCHANGED)
        target_shape = label.shape[0]//stride,label.shape[1]//stride
        feature_label = np.zeros(target_shape,dtype = np.uint8)
        if label is None:
            print 'plese check label root path'
        h,w = label.shape
        for y in range(0,h-stride,stride):
            for x in range(0,w-stride,stride):
                patch = label[y:(y+stride),x:(x+stride)]
                num_pixels = np.sum((patch != 0))
                if num_pixels >= 1:
                    feature_label[y//stride][x//stride] = 1

        cv2.imwrite(os.path.join(save_root,file),feature_label)

def main():
    label_root = './generate_defect1/label'
    stride = 128
    save_root = './generate_defect1/feature_label'
    generate_detecton_label(label_root,stride,save_root)

if __name__ == '__main__':
    main()





    
