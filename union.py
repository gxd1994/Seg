import cv2,os,math,shutil
from detection import Detection_Net
from seg import Seg_Net
import tensorflow as tf
import numpy as np
from utils import decode_labels

SEG_MODEL_PATH='./seg_snapshots/model.ckpt-13000'
DET_MODEL_PATH='./det_snapshots/model.ckpt-7500'

SEG_BATCH_SIZE = 32
DET_BATCH_SIZE = 8

SAVE_SEG_DIR = './test_result/img'
IMAGE_DIR = './test_data/img'

DET_FEATSRIDE = 32

IMG_MEAN = np.array((69.73,69.73,69.73), dtype=np.float32)

DET_LOG_DIR = './union_det_log'

class Union_Test_Net():
    # def __init__(self):
    #     pass
    def get_test_data(self,image_dir):
        files = os.listdir(image_dir)
        imgname_list = []
        imgs = None
        img_num = len(files)
        img_size = cv2.imread(os.path.join(image_dir,files[0])).shape
        imgs = np.zeros((img_num,img_size[0],img_size[1],img_size[2]),dtype=np.float32)
        for i,file in enumerate(files):
            imgname_list.append(file)
            img = cv2.imread(os.path.join(image_dir,file))
            if img is None:
                print 'please check img path'
                return
            #img = np.transpose(img,axes=(2,0,1))

            img = img.astype(np.float32)
            img = img - IMG_MEAN
            imgs[i] = img
            # img = img[np.newaxis,:,:,:]
            #
            # if i == 0:
            #     imgs = img
            # else:
            #     imgs = np.concatenate((imgs,img),axis=0)

        print 'read test imags. shape:',imgs.shape

        return imgs,imgname_list

    def get_detection_result(self,imgs):

        with tf.Graph().as_default():
            image_batch = tf.placeholder(dtype=tf.float32,shape=[None,None,None,3],name = 'image_batch_det')
            is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
            with tf.variable_scope('Detection_Net') as scope:
                det_net = Detection_Net()
                logits = det_net.inference(image_batch, is_training)
                det_pred = det_net.eval(logits=logits)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                # summary
                if os.path.exists(DET_LOG_DIR):
                    shutil.rmtree(DET_LOG_DIR)
                summary_writer = tf.summary.FileWriter(DET_LOG_DIR, graph=tf.get_default_graph())

                saver = tf.train.Saver()
                saver.restore(sess,DET_MODEL_PATH)
                print "Restored model parameters from {}".format(DET_MODEL_PATH)

                num = int(math.ceil(float(imgs.shape[0]) / DET_BATCH_SIZE))
                preds_final = None
                for i in range(num):
                    start = i * DET_BATCH_SIZE
                    end = min((i + 1) * DET_BATCH_SIZE, imgs.shape[0])
                    input_batch = imgs[start:end]
                    det_preds, = sess.run([det_pred],feed_dict={image_batch:input_batch,is_training:False})
                    #print 'det_preds',det_preds.shape
                    det_preds = np.squeeze(det_preds,axis=3)
                    #det_preds = det_preds[np.newaxis,:,:,:]
                    if i == 0:
                        preds_final = det_preds
                    else:
                        preds_final = np.concatenate((preds_final,det_preds),axis=0)
                print 'det_preds_final:',preds_final.shape
                summary_writer.close()
            return preds_final

    def _get_seg_data(self,preds,img,stride):
        print img.shape,preds.shape
        all_patch_list = []
        all_patch_cood_list = []
        for i in range(preds.shape[0]):
            cood = np.transpose(np.nonzero(preds[i]))
            # print cood.shape
            # print cood
            all_patch_cood_list.append(cood)
            per_patch_list = []
            image_patch_list = []
            for j in range(cood.shape[0]):
                y_start = cood[j][0]
                x_start = cood[j][1]
                patch = img[i,y_start * stride:((y_start + 1) * stride), x_start*stride:((x_start + 1) * stride), :]
                # cv2.imshow('patch', patch)
                # cv2.waitKey(0)
                #print 'patch.shape',patch.shape
                per_patch_list.append(patch)
            all_patch_list.append(per_patch_list)

        all_patch_np_list = [np.array(e) for e in all_patch_list]

        return all_patch_np_list,all_patch_cood_list

    def get_seg_result(self,det_preds,imgs,imgname_list,stride=128):

        with tf.Graph().as_default():
            image_batch = tf.placeholder(dtype=tf.float32,shape=[None,stride,stride,3],name = 'image_batch_seg')
            is_training = tf.placeholder(tf.bool, shape=[], name='is_training')

            with tf.variable_scope('Seg_Net'):
                seg_net = Seg_Net()
                logits = seg_net.inference(image_batch, is_training)
                seg_pred = seg_net.eval(logits=logits)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver()
                saver.restore(sess,SEG_MODEL_PATH)
                print "Restored model parameters from {}".format(SEG_MODEL_PATH)

                all_patch_np_list,all_patch_cood_list = self._get_seg_data(det_preds,imgs,stride)

                for i,(patchs,coods) in enumerate(zip(all_patch_np_list,all_patch_cood_list)):
                    num = int(math.ceil(float(patchs.shape[0])/SEG_BATCH_SIZE))
                    mask = np.zeros(shape=(imgs.shape[1],imgs.shape[2]),dtype=np.uint8)
                    for j in range(num):
                        start = j*SEG_BATCH_SIZE
                        end = min((j+1)*SEG_BATCH_SIZE,patchs.shape[0])
                        input_batch = patchs[start:end]
                        input_coods = coods[start:end]
                        seg_preds, = sess.run([seg_pred],feed_dict={image_batch:input_batch,is_training:False})
                        print 'seg_preds.shape', seg_preds.shape
                        seg_preds = np.squeeze(seg_preds,axis=3)
                        for k in range(seg_preds.shape[0]):
                            y_start = input_coods[k][0]
                            x_start = input_coods[k][1]
                            mask[y_start*stride:(y_start+1)*stride,x_start*stride:(x_start+1)*stride] = seg_preds[k]

                    mask[np.where(mask==1)] = 255
                    # cv2.namedWindow('mask',cv2.WINDOW_NORMAL)
                    # cv2.imshow('mask',mask)
                    # cv2.waitKey(0)
                    if not os.path.exists(SAVE_SEG_DIR):
                        os.makedirs(SAVE_SEG_DIR)
                    cv2.imwrite(os.path.join(SAVE_SEG_DIR,imgname_list[i]),mask)



def main():
    net = Union_Test_Net()
    imgs,imgname_list = net.get_test_data(IMAGE_DIR)
    det_results = net.get_detection_result(imgs)
    net.get_seg_result(det_preds=det_results,imgs=imgs,imgname_list=imgname_list,stride=DET_FEATSRIDE)


if __name__ == '__main__':
    main()






