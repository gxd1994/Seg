import tensorflow as tf
import sys,os,math,time,shutil,cv2
import argparse
import matplotlib.pyplot as plt
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR,'utils'))
from utils import conv2d,max_pool2d,fully_connected,atrous_conv2d
from utils import ImageReader,decode_labels

N_CLASSES = 2
BATCH_SIZE = 16  #1582
DATA_DIRECTORY = './dataset/seg/generate_defect1' #'/home/VOCdevkit'

DATA_TRAIN_LIST_PATH = DATA_DIRECTORY+'/train.txt'
DATA_VAL_LIST_PATH = DATA_DIRECTORY+'/test.txt'

INPUT_SIZE = '128,128'
FEATSTRIDE = 1
LEARNING_RATE = 1e-3
NUM_STEPS = 20000
RANDOM_SCALE = False  #True
RESTORE_FROM = './seg_snapshots/model.ckpt-19000'
SAVE_DIR = './seg_images/'
SAVE_NUM_IMAGES = 1
SAVE_PRED_EVERY = 200
VAL_PRED = 1000
SNAPSHOT_DIR = './seg_snapshots/'
WEIGHTS_PATH   = './util/net_weights.ckpt'
LOG_DIR = './seg_log'

VAL_LOOP = int(math.ceil(float(395)/BATCH_SIZE))

IMG_MEAN = np.array((69.73,69.73,69.73), dtype=np.float32)

SEG_COLLECTION = 'Seg_Net'
SEG_VAL_COLLECTION = 'Seg_Net_Val'

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data_dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data_train_list", type=str, default=DATA_TRAIN_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--data_val_list", type=str, default=DATA_VAL_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--input_size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE,
                        help="Learning rate for training.")
    parser.add_argument("--num_steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--restore_from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR,
                        help="Where to save figures with predictions.")
    parser.add_argument("--save_num_images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save_pred_every", type=int, default=SAVE_PRED_EVERY,
                        help="Save figure with predictions and ground truth every often.")
    parser.add_argument("--snapshot_dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weights_path", type=str, default=WEIGHTS_PATH,
                        help="Path to the file with caffemodel weights. "
                             "If not set, all the variables are initialised randomly.")

    parser.add_argument("--log_dir", type=str, default=LOG_DIR,
                        help="where to save log file")

    return parser.parse_args()

def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')

def save_train_result(args,step,images,labels,preds):

    fig, axes = plt.subplots(args.save_num_images, 2, figsize=(16, 12))
    for i in xrange(args.save_num_images):
        cv2.imwrite(args.save_dir + str(step) + "_%d.png" % i, (images[i] + IMG_MEAN)[:, :, ::-1].astype(np.uint8))

        axes.flat[i * 2 + 0].set_title('mask')
        axes.flat[i * 2 + 0].imshow(decode_labels(labels[i, :, :, 0]))

        axes.flat[i * 2 + 1].set_title('pred')
        axes.flat[i * 2 + 1].imshow(decode_labels(preds[i, :, :, 0]))

    plt.savefig(args.save_dir + str(step) + ".png")
    plt.close(fig)

def save_val_result(args,step,images,labels,preds,i):
    for j in range(BATCH_SIZE):
        fig, axes = plt.subplots(1, 2, figsize=(16, 12))
        if j < 1:
            cv2.imwrite(args.save_dir + str(step) + '_' + str(i * BATCH_SIZE + j) + "test_img.png",
                        (images[j] + IMG_MEAN)[:, :, ::-1].astype(np.uint8))

        axes.flat[0].set_title('mask')
        axes.flat[0].imshow(decode_labels(labels[j, :, :, 0]))

        axes.flat[1].set_title('pred')
        axes.flat[1].imshow(decode_labels(preds[j, :, :, 0]))

        plt.savefig(args.save_dir + str(step) + '_' + str(i * BATCH_SIZE + j) + "test.png")
        plt.close(fig)

def get_data_queue(args, coord, is_training=True):
    h, w = map(int, args.input_size.split(','))
    input_size_img = (h, w)
    input_size_label = (h / FEATSTRIDE, w / FEATSTRIDE)

    # Load reader.
    if is_training:
        with tf.name_scope("create_train_inputs"):
            reader_train = ImageReader(
                args.data_dir,
                args.data_train_list,
                input_size_img,
                input_size_label,
                RANDOM_SCALE,
                IMG_MEAN,
                coord)
            image_batch_train, label_batch_train = reader_train.dequeue(args.batch_size)
            return image_batch_train, label_batch_train
    else:
        with tf.name_scope("create_val_inputs"):
            reader_val = ImageReader(
                args.data_dir,
                args.data_val_list,
                input_size_img,
                input_size_label,
                False,
                IMG_MEAN,
                coord)
            image_batch_val, label_batch_val = reader_val.dequeue(args.batch_size, is_training=False)
            return image_batch_val, label_batch_val

class Seg_Net():
    def __init__(self,n_classes = 2):
        #self.lr = lr
        self.n_classes = n_classes
        # pass
    def inference(self,images,is_training,weight_decay=1e-5):
        self.image_size = images.get_shape()[1:3]

        conv1 = conv2d(images,48,[3,3],'conv1',[1,1],weight_decay=weight_decay,use_xavier =True,
                       stddev=1e-3,is_training= is_training,bn=True,activation_fn=tf.nn.relu)
        pool1 = max_pool2d(conv1,[3,3],'pool1',[1,1],padding='SAME')

        conv2 = conv2d(pool1,48,[3,3],'conv2',[1,1],weight_decay=weight_decay,use_xavier=True,
                        stddev=1e-3,is_training=is_training,bn=True,activation_fn=tf.nn.relu)
        pool2 = max_pool2d(conv2,[3,3],'pool12',[1,1],padding='SAME')


        conv3 = conv2d(pool2,96, [3,3], 'conv3', [1, 1], weight_decay=weight_decay, use_xavier=True,
                       stddev=1e-3, is_training=is_training, bn=True, activation_fn=tf.nn.relu)
        pool3 = max_pool2d(conv3, [3,3], 'pool13', [1, 1], padding='SAME')


        conv4 = conv2d(pool3,128,[3,3],'conv4',[1,1],weight_decay=weight_decay,use_xavier=True,
                       stddev=1e-3,is_training=is_training,bn=True,activation_fn= tf.nn.relu)
        pool4 = max_pool2d(conv4, [3,3], 'pool14', [1, 1], padding='SAME')


        conv5 = conv2d(pool4,N_CLASSES,[3,3],'conv5',[1,1],weight_decay=weight_decay,use_xavier=True,
                       stddev=1e-3,is_training=is_training,bn=False,activation_fn= None)

        logits = conv5

        return logits
    def _prepare_label(self,label_batch,new_size):
        # label_batch_tmp1 = tf.cast(tf.squeeze(label_batch,axis=-1),tf.int32)
        label_batch_tmp0 = tf.image.resize_nearest_neighbor(label_batch,new_size,name='reisze_label_op')
        tf.summary.image('resize_label', tf.cast(label_batch_tmp0,tf.float32), collections=[SEG_COLLECTION])
        label_batch_tmp1 = tf.squeeze(label_batch_tmp0, axis=-1)
        label_batch_tmp2 = tf.one_hot(label_batch_tmp1,depth=N_CLASSES)
        label_batch_final = tf.reshape(label_batch_tmp2,[-1,N_CLASSES])

        return label_batch_final

    def loss(self,logits,label_batch):
        
        tf.summary.image('train_raw_label', tf.cast(label_batch,tf.float32), collections=[SEG_COLLECTION])
        tf.summary.image('val_label', tf.cast(label_batch,tf.float32), collections=[SEG_VAL_COLLECTION])

        label_batch_final = self._prepare_label(label_batch,tf.stack(logits.get_shape()[1:3]))

        logits_final = tf.reshape(logits,[-1,N_CLASSES])

        print 'final_shape',label_batch_final,logits_final

        cross_entroy = tf.nn.softmax_cross_entropy_with_logits(labels=label_batch_final,logits=logits_final,name='cross_entroy')

        cross_entroy_mean = tf.reduce_mean(cross_entroy)

        tf.add_to_collection('losses',cross_entroy_mean)

        #[regularity_loss,_] = tf.get_collection('losses')

        #print "collection",tf.get_collection('losses')
        loss = tf.add_n(tf.get_collection('losses'),name='total_loss')

        tf.summary.scalar('total_loss', loss, collections=[SEG_COLLECTION])
        tf.summary.scalar('cross_entroy_loss', cross_entroy_mean, collections=[SEG_COLLECTION])

        return loss

    def train(self,total_loss,global_step):
        with tf.control_dependencies([total_loss]):
            optimiser = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
            train_op = optimiser.minimize(total_loss,global_step=global_step)

            return train_op
    def _calculate_miou(self,logits,label_batch):
        with tf.variable_scope('MIOU_CAL'):
            confusion_matrix = tf.confusion_matrix(labels=tf.reshape(label_batch,[-1]),predictions=tf.reshape(logits,[-1]),num_classes=N_CLASSES,dtype=tf.float32)
            def cal_miou(matrix):
                sum_col = np.zeros(shape = [N_CLASSES],dtype=np.float32)
                sum_row = np.zeros(shape = [N_CLASSES],dtype=np.float32)
                miou = np.zeros(shape = [],dtype=np.float32)
                for i in range(N_CLASSES):
                    for j in range(N_CLASSES):
                        sum_row[i] += matrix[i][j]
                        sum_col[j] += matrix[i][j]
                for i in range(N_CLASSES):
                    miou += matrix[i][i]/(sum_col[i]+sum_row[i]-matrix[i][i])
                miou = (miou/N_CLASSES).astype(np.float32)
                return miou

            miou = tf.py_func(cal_miou,[confusion_matrix],tf.float32)
        return miou

    def metrcis(self,logits,label_batch):
        label_batch_final = tf.image.resize_nearest_neighbor(label_batch,tf.stack(logits.get_shape()[1:3]))

        logits_tmp = tf.cast(tf.argmax(logits, axis=3), tf.uint8)
        label_batch_tmp = tf.squeeze(label_batch_final, axis=3)

        miou = self._calculate_miou(logits_tmp,label_batch_tmp)

        tf.summary.scalar('miou_value_train',miou,collections=[SEG_COLLECTION])
        tf.summary.scalar('miou_value_val', miou, collections=[SEG_VAL_COLLECTION])

        acc = tf.reduce_mean(tf.cast(tf.equal(logits_tmp,label_batch_tmp),tf.float32))

        tf.summary.scalar('accuracy_train', acc, collections=[SEG_COLLECTION])
        tf.summary.scalar('accuracy_val', acc, collections=[SEG_VAL_COLLECTION])

        return acc,miou

    def eval(self,logits):
        result = tf.image.resize_bilinear(logits,tf.stack(self.image_size),name='resize_pred_op')
        #result = tf.nn.softmax(logits=result)
        result = tf.cast(tf.argmax(result,axis=3),tf.uint8)
        result = tf.expand_dims(result,axis=3,name='pred')
        tf.summary.image('train_pred', tf.cast(result,tf.float32), collections=[SEG_COLLECTION])
        tf.summary.image('val_pred', tf.cast(result,tf.float32), collections=[SEG_VAL_COLLECTION])
        return result

    def construct_graph(self,image_batch,label_batch,is_training,global_step):
        # construct_graph
        logits = self.inference(image_batch,is_training=is_training)
        pred = self.eval(logits=logits)
        loss = self.loss(logits=logits,label_batch=label_batch)
        accuracy,miou = self.metrcis(logits=logits,label_batch=label_batch)
        train_op = self.train(total_loss=loss,global_step=global_step)

        return loss,train_op,pred,accuracy,miou


# def construct_and_test(args):
#     # Create queue coordinator.
#     coord = tf.train.Coordinator()
#     with tf.Graph().as_default():
#         # get data.
#         image_batch,label_batch = get_data_queue(args,coord,is_training=False)


def construct_and_val(args):

    # Create queue coordinator.
    coord = tf.train.Coordinator()

    with tf.Graph().as_default():

        is_training = tf.placeholder(tf.bool, shape=[], name='is_training')

        # get data.
        image_batch,label_batch = get_data_queue(args,coord,is_training=False)

        # construct_graph
        with tf.variable_scope('Seg_Net') as scope:
            seg_net = Seg_Net()
            logits = seg_net.inference(image_batch,is_training)
            pred = seg_net.eval(logits=logits)
            accuracy, miou = seg_net.metrcis(logits=logits, label_batch=label_batch)

        # session
        # Set up tf session and initialize variables.
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.8
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # Saver
        # Saver for storing checkpoints of the model.
        #print tf.all_variables()
        # print type(tf.trainable_variables()[0]),tf.trainable_variables()[0]
        saver = tf.train.Saver(max_to_keep=40)
        if args.restore_from is not None:
            saver.restore(sess, args.restore_from)
            print("Restored model parameters from {}".format(args.restore_from))
        else:
            print 'must restore model!'
            return
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        # summary
        if os.path.exists(args.log_dir):
            shutil.rmtree(args.log_dir)
        summary_writer = tf.summary.FileWriter(args.log_dir, graph=tf.get_default_graph())

        merged_val = tf.summary.merge_all(SEG_VAL_COLLECTION)

        # Start queue threads.
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        for i in range(VAL_LOOP):
            print "total val step:%d   cur step:%d" % (VAL_LOOP, i)
            start_time = time.time()
            summary_val, images, labels, preds, acc, miou_value = sess.run(
                [merged_val, image_batch, label_batch, pred, accuracy, miou], feed_dict={is_training: False})
            duration = time.time() - start_time
            summary_writer.add_summary(summary_val, i)

            save_val_result(args,0,images,labels, preds, i)

            print 'step {:<6d}, val: acc = {:.5f}, miou={:.5f}, {:.5f} sec/step'.format(i, acc, miou_value, duration)

        summary_writer.close()
    coord.request_stop()
    coord.join(threads)

def construct_and_train(args):

    # Create queue coordinator.
    coord = tf.train.Coordinator()

    with tf.Graph().as_default():

        global_step = tf.Variable(0,trainable=False,dtype=tf.int64,name='global_step')

        tf.summary.scalar('global_step_value', global_step, collections=[SEG_COLLECTION])

        is_training = tf.placeholder(tf.bool, shape=[], name='is_training')

        # get data.
        image_batch_train,label_batch_train = get_data_queue(args,coord,is_training=True)
        image_batch_val,label_batch_val = get_data_queue(args,coord,is_training=False)

        image_batch, label_batch = tf.cond(is_training, lambda: (image_batch_train, label_batch_train),
                                           lambda: (image_batch_val, label_batch_val))
        # construct_graph
        with tf.variable_scope('Seg_Net') as scope:
            seg_net = Seg_Net()
            loss, train_op, pred, accuracy, miou = seg_net.construct_graph(
                                    image_batch,label_batch,is_training,global_step)

        # session
        # Set up tf session and initialize variables.
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.8
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # Saver
        # Saver for storing checkpoints of the model.
        #print tf.all_variables()
        # print type(tf.trainable_variables()[0]),tf.trainable_variables()[0]
        saver = tf.train.Saver(max_to_keep=40)
        if args.restore_from is not None:
            saver.restore(sess, args.restore_from)
            print("Restored model parameters from {}".format(args.restore_from))

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        # summary
        if os.path.exists(args.log_dir):
            shutil.rmtree(args.log_dir)
        summary_writer = tf.summary.FileWriter(args.log_dir, graph=tf.get_default_graph())
        
        merged_train = tf.summary.merge_all(SEG_COLLECTION)
        merged_val = tf.summary.merge_all(SEG_VAL_COLLECTION)

        # Start queue threads.
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        # Iterate over training steps.
        start_step, = sess.run([global_step])
        print "start_step",start_step
        for step in range(start_step,args.num_steps):

            # train and fetch date
            start_time = time.time()
            summary_train, loss_value, _, acc,miou_value = sess.run(
                [merged_train,loss, train_op, accuracy,miou],feed_dict={is_training: True})
            duration = time.time() - start_time
            summary_writer.add_summary(summary_train, step)

            print 'step {:<6d}, loss = {:.5f}, acc = {:.5f},miou = {:.5f}, {:.5f} sec/step'.format(
                step, loss_value, acc, miou_value, duration)

            # train intermediate result
            if step % args.save_pred_every == 0 and step != 0:
                start_time = time.time()
                loss_value, images, labels, preds, _ ,acc, miou_value = sess.run(
                    [loss, image_batch, label_batch, pred, train_op,accuracy,miou],
                        feed_dict={is_training: True})
                duration = time.time() - start_time

                save_train_result(args,step,images,labels,preds)
                print 'step {:<6d}, loss = {:.5f}, acc = {:.5f}, miou={:.5f}, {:.5f} sec/step'.format(
                    step, loss_value,acc,miou_value,duration)

            # Val result
            if step % VAL_PRED == 0 and step != 0:
                for i in range(VAL_LOOP):
                    print "total val step:%d   cur step:%d"%(VAL_LOOP,i)
                    start_time = time.time()
                    summary_val, images, labels, preds, acc, miou_value = sess.run(
                        [merged_val,image_batch, label_batch, pred, accuracy, miou],feed_dict={is_training: False})
                    duration = time.time() - start_time
                    summary_writer.add_summary(summary_val,step)

                    save_val_result(args,step,images,labels,preds,i)
                    print 'step {:<6d}, val: acc = {:.5f}, miou={:.5f}, {:.5f} sec/step'.format(step,acc,miou_value,duration)

                save(saver, sess, args.snapshot_dir,step)

        summary_writer.close()
    coord.request_stop()
    coord.join(threads)



def main():
    args = get_arguments()
    construct_and_train(args)
    #construct_and_val(args)

if __name__ == '__main__':
    main()
