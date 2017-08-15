import cv2,os
import numpy as np


class Data_generate():
    def __init__(self,patch_size = (128,128),stride_radio = (0.5,0.5),pos_threshold = 0.1,lab_ext = '.PNG'):
        self.patch_size = patch_size
        self.stride = (int(patch_size[0] * stride_radio[0]),int(patch_size[1] * stride_radio[1]))
        self.pos_threshold = pos_threshold
        self.lab_ext = lab_ext
        # self.data_root = data_root

    def generate(self,img_root,label_root,save_root):
        files = os.listdir(img_root)
        for file in files:
            file_path = os.path.join(img_root,file)
            label_path = os.path.join(label_root,os.path.splitext(file)[0]+'_label'+self.lab_ext)


            img = cv2.imread(file_path)
            if img is None:
                print 'please check img file path'
                exit()
            label = cv2.imread(label_path,cv2.CV_LOAD_IMAGE_UNCHANGED)
            if label is None:
                print 'please check label file ext'
                exit()
            self._generate_patches(img,label,save_root,file)

    def _generate_patches(self,img,label,save_root,file):

        img_h,img_w = img.shape[0],img.shape[1]
        patch_h,patch_w = self.patch_size[0],self.patch_size[1]
        str_h,str_w = self.stride[0],self.stride[0]
        save_path_img = os.path.join(save_root,'img',os.path.splitext(file)[0])
        img_ext = os.path.splitext(file)[1]
        save_path_label = os.path.join(save_root,'label',os.path.splitext(file)[0])
        lab_ext = self.lab_ext
        if not os.path.exists(save_root+'/img'):
            os.makedirs(save_root+'/img')
        if not os.path.exists(save_root+'/label'):
            os.makedirs(save_root+'/label')

        for y in range(0,img_h-patch_h,str_h):
            for x in range(0,img_w-patch_w,str_w):
                img_patch = img[y:(y+patch_h),x:(x+patch_w)]
                label_patch = label[y:(y+patch_h),x:(x+patch_w)]
                label_patch_b = (label_patch != 0)
                num_pixels = np.sum(label_patch_b)
                if num_pixels > self.pos_threshold * label_patch.shape[0] * label_patch.shape[1]:

                    cv2.imwrite(save_path_img+img_ext,img_patch)
                    cv2.imwrite(save_path_label+lab_ext, label_patch)


                    # img_patch_flipV = cv2.flip(img_patch, 0)
                    # label_patch_flipV = cv2.flip(label_patch, 0)
                    #
                    # cv2.imwrite(save_path_img+'_v'+img_ext,img_patch_flipV)
                    # cv2.imwrite(save_path_label+'_v'+lab_ext, label_patch_flipV)
                    #
                    # img_patch_flipH = cv2.flip(img_patch, 1)
                    # label_patch_flipH = cv2.flip(label_patch, 1)
                    #
                    # cv2.imwrite(save_path_img+'_h'+img_ext,img_patch_flipH)
                    # cv2.imwrite(save_path_label+'_h'+lab_ext, label_patch_flipH)
                    #
                    # img_patch_flipR = cv2.flip(img_patch, -1)
                    # label_patch_flipR = cv2.flip(label_patch, -1)
                    #
                    # cv2.imwrite(save_path_img+'_r'+img_ext,img_patch_flipR)
                    # cv2.imwrite(save_path_label+'_r'+lab_ext, label_patch_flipR)


def main():
    patch_size = (128,128)
    img_root =  './joint_defect1/img'
    label_root = './joint_defect1/label'
    save_root = './gen'
    Generation = Data_generate(patch_size=patch_size)
    Generation.generate(img_root,label_root,save_root)

if __name__ == "__main__":
    main()
