import tensorflow as tf
import sys,os,math,time,shutil,cv2
import argparse
import matplotlib.pyplot as plt
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR,'utils'))
from utils import conv2d,max_pool2d,fully_connected
from utils import ImageReader,decode_labels

N_CLASSES = 2
BATCH_SIZE = 10    #1588 396
DATA_DIRECTORY = './dataset/detection/generate_defect1' #'/home/VOCdevkit'

DATA_TRAIN_LIST_PATH = DATA_DIRECTORY+'/train.txt'
DATA_VAL_LIST_PATH = DATA_DIRECTORY+'/test.txt'

INPUT_SIZE = '1024,4096'
FEATSTRIDE = 32
LEARNING_RATE = 1e-3
NUM_STEPS = 20000+1
RANDOM_SCALE = False  #True
RESTORE_FROM = './det_snapshots/model.ckpt-16500'
SAVE_DIR = './det_images/'
SAVE_NUM_IMAGES = 1
SAVE_PRED_EVERY = 200
VAL_PRED = 1500
SNAPSHOT_DIR = './det_snapshots/'
WEIGHTS_PATH   = './util/net_weights.ckpt'
LOG_DIR = './det_log'
DELETE_LOG = False

VAL_LOOP = int(math.ceil(float(396)/BATCH_SIZE))

IMG_MEAN = np.array((68.73,68.73,68.73), dtype=np.float32)

DET_COLLECTION = 'Detection_Net'
DET_VAL_COLLECTION = 'Detection_Net_Val'

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


def save_train_result(args, step, images, labels, preds):
    fig, axes = plt.subplots(args.save_num_images, 2, figsize=(16, 12))
    for i in xrange(args.save_num_images):
        cv2.imwrite(args.save_dir + str(step) + "_%d.png" % i, (images[i] + IMG_MEAN)[:, :, ::-1].astype(np.uint8))

        axes.flat[i * 2 + 0].set_title('mask')
        axes.flat[i * 2 + 0].imshow(decode_labels(labels[i, :, :, 0]))

        axes.flat[i * 2 + 1].set_title('pred')
        axes.flat[i * 2 + 1].imshow(decode_labels(preds[i, :, :, 0]))

    plt.savefig(args.save_dir + str(step) + ".png")
    plt.close(fig)


def save_val_result(args, step, images, labels, preds, i):
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



def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')


class Detection_Net():
    def __init__(self,n_classes = 2):
        # self.lr = lr
        self.n_classes = n_classes
        # pass
    def inference(self,images,is_training,weight_decay=1e-5):


        conv1 = conv2d(images, 12, [3, 3], 'conv1', [2, 2], weight_decay=weight_decay, use_xavier=True,
                       stddev=1e-3, is_training=is_training, bn=True, activation_fn=tf.nn.relu)
        pool1 = max_pool2d(conv1, [3, 3], 'pool1', [2, 2], padding='SAME')


        conv2 = conv2d(pool1, 12, [3, 3], 'conv2', [1, 1], weight_decay=weight_decay, use_xavier=True,
                       stddev=1e-3, is_training=is_training, bn=True, activation_fn=tf.nn.relu)
        pool2 = max_pool2d(conv2, [3, 3], 'pool12', [2, 2], padding='SAME')


        conv3 = conv2d(pool2, 24, [3, 3], 'conv3', [1, 1], weight_decay=weight_decay, use_xavier=True,
                       stddev=1e-3, is_training=is_training, bn=True, activation_fn=tf.nn.relu)
        pool3 = max_pool2d(conv3, [3, 3], 'pool13', [2, 2], padding='SAME')


        conv4 = conv2d(pool3, 48, [3, 3], 'conv4', [1, 1], weight_decay=weight_decay, use_xavier=True,
                       stddev=1e-3, is_training=is_training, bn=True, activation_fn=tf.nn.relu)
        pool4 = max_pool2d(conv4, [3, 3], 'pool14', [2, 2], padding='SAME')

        conv5 = conv2d(pool4, 64, [3, 3], 'conv5', [1, 1], weight_decay=weight_decay, use_xavier=True,
                       stddev=1e-3, is_training=is_training, bn=True, activation_fn=tf.nn.relu)
        pool5 = max_pool2d(conv5, [3, 3], 'pool15', [1, 1], padding='SAME')

        conv6 = conv2d(pool5, 64, [3, 3], 'conv6', [1, 1], weight_decay=weight_decay, use_xavier=True,
                       stddev=1e-3, is_training=is_training, bn=True, activation_fn=tf.nn.relu)
        pool6 = max_pool2d(conv6, [3, 3], 'pool16', [1, 1], padding='SAME')

        conv7 = conv2d(pool6, N_CLASSES, [1, 1], 'conv7', [1, 1], weight_decay=weight_decay, use_xavier=True,
                       stddev=1e-3, is_training=is_training, bn=False, activation_fn=None)

        logits = conv7

        return logits

    def _prepare_label(self,label_batch):
        # label_batch_tmp1 = tf.cast(tf.squeeze(label_batch,axis=-1),tf.int32)
        label_batch_tmp1 = tf.squeeze(label_batch, axis=-1)
        label_batch_tmp2 = tf.one_hot(label_batch_tmp1,depth=N_CLASSES)
        label_batch_final = tf.reshape(label_batch_tmp2,[-1,N_CLASSES])

        return label_batch_final

    def loss(self,logits,label_batch):
        
        tf.summary.image('train_label',tf.cast(label_batch,tf.float32),collections=[DET_COLLECTION])
        tf.summary.image('val_label',tf.cast(label_batch,tf.float32),collections=[DET_VAL_COLLECTION])

        label_batch_final = self._prepare_label(label_batch)

        logits_final = tf.reshape(logits,[-1,N_CLASSES])

        print 'final_shape',label_batch_final,logits_final

        cross_entroy = tf.nn.softmax_cross_entropy_with_logits(labels=label_batch_final,logits=logits_final,name='cross_entroy')

        cross_entroy_mean = tf.reduce_mean(cross_entroy)

        tf.add_to_collection('losses',cross_entroy_mean)

        #[regularity_loss,_] = tf.get_collection('losses')


        loss = tf.add_n(tf.get_collection('losses'),name='total_loss')

        tf.summary.scalar('total_loss',loss,collections=[DET_COLLECTION])
        tf.summary.scalar('cross_entroy_loss', cross_entroy_mean, collections=[DET_COLLECTION])

        return loss

    def train(self,total_loss,global_step):
        with tf.control_dependencies([total_loss]):
            optimiser = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
            train_op = optimiser.minimize(total_loss,global_step=global_step)

            return train_op

    def metrcis(self,logits,label_batch):

        acc = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(logits,axis=3),tf.uint8),tf.squeeze(label_batch,axis=3)),tf.float32))
        #acc1,acc_op = tf.metrics.accuracy(labels=tf.squeeze(label_batch,axis=-1),predictions = tf.cast(tf.argmax(logits,axis=3),tf.uint8))
        tf.summary.scalar('accuracy_train',acc,collections=[DET_COLLECTION])
        tf.summary.scalar('accuracy_val',acc,collections=[DET_VAL_COLLECTION])
        return acc   #,acc1,acc_op

    def eval(self,logits):

        result = tf.nn.softmax(logits=logits)
        result = tf.cast(tf.argmax(result,axis=3),tf.uint8)
        result = tf.expand_dims(result,axis=3,name='pred')
        tf.summary.image('train_pred',tf.cast(result,tf.float32),collections=[DET_COLLECTION])
        tf.summary.image('val_pred',tf.cast(result,tf.float32),collections=[DET_VAL_COLLECTION])
        return result

    def construct_graph(self,image_batch,label_batch,is_training,global_step):
        # construct_graph
        logits = self.inference(image_batch,is_training=is_training)
        pred = self.eval(logits=logits)
        loss = self.loss(logits=logits,label_batch=label_batch)
        accuracy = self.metrcis(logits=logits,label_batch=label_batch)
        train_op = self.train(total_loss=loss,global_step=global_step)

        return loss,train_op,pred,accuracy



def construct_and_val(args):

    # Create queue coordinator.
    coord = tf.train.Coordinator()

    with tf.Graph().as_default():

        is_training = tf.placeholder(tf.bool, shape=[], name='is_training')

        # get data.
        image_batch,label_batch = get_data_queue(args,coord,is_training=False)

        # construct_graph
        with tf.variable_scope('Detection_Net') as scope:
            det_net = Detection_Net()
            logits = det_net.inference(image_batch,is_training)
            pred = det_net.eval(logits=logits)
            accuracy = det_net.metrcis(logits=logits, label_batch=label_batch)

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
        if DELETE_LOG:
            if os.path.exists(args.log_dir):
                shutil.rmtree(args.log_dir)
        summary_writer = tf.summary.FileWriter(args.log_dir, graph=tf.get_default_graph())

        merged_val = tf.summary.merge_all(DET_VAL_COLLECTION)

        # Start queue threads.
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        for i in range(VAL_LOOP):
            print "total val step:%d   cur step:%d" % (VAL_LOOP, i)
            start_time = time.time()
            summary_val, images, labels, preds, acc = sess.run(
                [merged_val, image_batch, label_batch, pred, accuracy], feed_dict={is_training: False})
            duration = time.time() - start_time
            summary_writer.add_summary(summary_val, i)

            save_val_result(args,0,images,labels, preds, i)

            print 'step {:<6d}, val: acc = {:.5f}, {:.5f} sec/step'.format(i, acc, duration)

        summary_writer.close()
    coord.request_stop()
    coord.join(threads)


def construct_and_train(args):

    # Create queue coordinator.
    coord = tf.train.Coordinator()

    with tf.Graph().as_default():
        global_step = tf.Variable(0,trainable=True,dtype=tf.int64,name='global_step')
        tf.summary.scalar('global_step_value',global_step,collections=[DET_COLLECTION])

        is_training = tf.placeholder(tf.bool, shape=[], name='is_training')

        # get data.
        image_batch_train,label_batch_train = get_data_queue(args,coord,is_training=True)
        image_batch_val,label_batch_val = get_data_queue(args,coord,is_training=False)

        image_batch, label_batch = tf.cond(is_training, lambda: (image_batch_train, label_batch_train),
                                           lambda: (image_batch_val, label_batch_val))
        # construct_graph
        with tf.variable_scope('Detection_Net') as scope:
            det_net = Detection_Net()
            loss, train_op, pred, accuracy = det_net.construct_graph(
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
            saver = tf.train.Saver(var_list = None, max_to_keep=40)
            if args.restore_from is not None:
                saver.restore(sess, args.restore_from)
                print("Restored model parameters from {}".format(args.restore_from))

            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)

            # summary
            if DELETE_LOG:
                if os.path.exists(args.log_dir):
                    shutil.rmtree(args.log_dir)
            summary_writer = tf.summary.FileWriter(args.log_dir, graph=tf.get_default_graph())

            merged_train = tf.summary.merge_all(DET_COLLECTION)
            merged_val = tf.summary.merge_all(DET_VAL_COLLECTION)


            # Start queue threads.
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)

            # Iterate over training steps.
            start_step, = sess.run([global_step])
            print "start_step",start_step
            for step in range(start_step,args.num_steps):
                # train step
                start_time = time.time()

                summary_train, loss_value, _, acc = sess.run(
                    [merged_train,loss, train_op, accuracy],feed_dict={is_training: True})

                duration = time.time() - start_time

                summary_writer.add_summary(summary_train, step)
                print 'step {:<6d}, loss = {:.5f}, acc = {:.5f}, {:.5f} sec/step'.format(
                    step, loss_value, acc, duration)

                #train intermediate result
                if step % args.save_pred_every == 0 and step != 0:
                    start_time = time.time()
                    loss_value, images, labels, preds, _ ,acc = sess.run(
                        [loss, image_batch, label_batch, pred, train_op,accuracy],
                            feed_dict={is_training: True})
                    duration = time.time() - start_time

                    save_train_result(args, step, images, labels, preds)

                    print 'step {:<6d}, loss = {:.5f}, acc = {:.5f},  {:.5f} sec/step'.format(
                        step, loss_value,acc,duration)

                # val result
                if step % VAL_PRED == 0 and step != 0:
                    save(saver, sess, args.snapshot_dir,step)
                    for i in range(VAL_LOOP):
                        print "total val step:%d   cur step:%d" % (VAL_LOOP, i)
                        start_time = time.time()

                        summary_val, images, labels, preds, acc = sess.run(
                            [merged_val,image_batch, label_batch, pred, accuracy],feed_dict={is_training: False})

                        duration = time.time() - start_time
                        summary_writer.add_summary(summary_val, step)

                        save_val_result(args, step, images, labels, preds, i)

                        print 'step {:<6d}, val: acc = {:.5f}, {:.5f} sec/step'.format(step,acc,duration)


        summary_writer.close()
    coord.request_stop()
    coord.join(threads)



def main():
    args = get_arguments()
    construct_and_train(args)
    #construct_and_val(args)

if __name__ == '__main__':
    main()

# conv1 = conv2d(images, 12, [3, 3], 'conv1', [2, 2], weight_decay=weight_decay, use_xavier=True,
#                stddev=1e-3, is_training=is_training, bn=True, activation_fn=tf.nn.relu)
# pool1 = max_pool2d(conv1, [3, 3], 'pool1', [2, 2], padding='SAME')
#
# conv2 = conv2d(pool1, 12, [3, 3], 'conv2', [1, 1], weight_decay=weight_decay, use_xavier=True,
#                stddev=1e-3, is_training=is_training, bn=True, activation_fn=tf.nn.relu)
# pool2 = max_pool2d(conv2, [3, 3], 'pool12', [2, 2], padding='SAME')
#
# conv3 = conv2d(pool2, 24, [3, 3], 'conv3', [1, 1], weight_decay=weight_decay, use_xavier=True,
#                stddev=1e-3, is_training=is_training, bn=True, activation_fn=tf.nn.relu)
# pool3 = max_pool2d(conv3, [3, 3], 'pool13', [2, 2], padding='SAME')
#
# conv4 = conv2d(pool3, 48, [3, 3], 'conv4', [1, 1], weight_decay=weight_decay, use_xavier=True,
#                stddev=1e-3, is_training=is_training, bn=True, activation_fn=tf.nn.relu)
# pool4 = max_pool2d(conv4, [3, 3], 'pool14', [2, 2], padding='SAME')
#
# conv5 = conv2d(pool4, 64, [3, 3], 'conv5', [1, 1], weight_decay=weight_decay, use_xavier=True,
#                stddev=1e-3, is_training=is_training, bn=True, activation_fn=tf.nn.relu)
# pool5 = max_pool2d(conv5, [3, 3], 'pool15', [1, 1], padding='SAME')
#
# conv6 = conv2d(pool5, 64, [3, 3], 'conv6', [1, 1], weight_decay=weight_decay, use_xavier=True,
#                stddev=1e-3, is_training=is_training, bn=True, activation_fn=tf.nn.relu)
# pool6 = max_pool2d(conv6, [3, 3], 'pool16', [1, 1], padding='SAME')
#
# conv7 = conv2d(pool6, N_CLASSES, [1, 1], 'conv7', [1, 1], weight_decay=weight_decay, use_xavier=True,
#                stddev=1e-3, is_training=is_training, bn=False, activation_fn=None)
#
# logits = conv7
#
# return logits