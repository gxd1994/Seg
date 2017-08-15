Data=./Train_Data_Done/

lmdb=./lmdb
mean_file=./mean.binaryproto



caffe_root_tool=/home/gxd/caffe-master/build/tools/

compute_image_mean=$caffe_root_tool/compute_image_mean

convert_imageset=$caffe_root_tool/convert_imageset

python creat_txt.py

$convert_imageset $Data ./data.txt $lmdb

GLOG_logtostderr=1 $compute_image_mean $lmdb $mean_file

 



