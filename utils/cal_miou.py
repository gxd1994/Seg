from py_img_seg_eval import mean_IU
import os,cv2

label_dir = '../test_result/label'
pred_dir = '../test_result/img'

files = os.listdir(pred_dir)

miou = 0.0
for file in files:
    img = cv2.imread(os.path.join(pred_dir,file),cv2.CV_LOAD_IMAGE_UNCHANGED)
    label = cv2.imread(os.path.join(label_dir,file),cv2.CV_LOAD_IMAGE_UNCHANGED)
    miou += mean_IU(img,label)

print 'miou:%f'%(miou/len(files))
