 # -*- coding: utf-8 -*-
"""
tf存储的图片工具类
"""
import numpy as np
from PIL import Image
import os

#******************** 将不同目录下的图片文件转变成数组 ***********************####

filedir = "C:\\Users\\lenovo\\Documents\\tfstudy\\pic"
categorys = ["one","two"]

#获取该目录下特定文件夹中的文件路径
def file_paths(category, filedir):
    filepaths=[]
    for root,dirs, files in os.walk(os.path.join(filedir, category)):  
        for file in files:
            filepaths.append(os.path.join(filedir, category, file))
    return filepaths

#根据路径打开一张图片，转换成数组
def open_image(filepath, grey):  #grey表示是否需要灰度化图片
    image = Image.open(filepath)
    imagearr = np.array(image, dtype=np.float64)
    shapelist = list(imagearr.shape)
    shapelist.insert(0,1)
    imagearr = np.reshape(imagearr, newshape=shapelist)
    if grey:
        shapelist[3] = 1
        imagearr=np.reshape(np.average(imagearr,axis=3),shapelist)
    return imagearr

#从各个文件夹中读取多张图片文件，并转化为array类型
def image_arrays(categorys, filedir, grey):
    imagearrays = []
    labels = []
    categorydict = {}
    categorynum = 0
    for category in categorys:
        #构建列表dict
        categorynum = categorynum + 1
        categorydict[category] = categorynum

        #获取文件路径
        filepaths = file_paths(category, filedir)
        #读取文件
        for filepath in filepaths:
            imagearrays.append(open_image(filepath, grey))
            labels.append(categorynum)  
        
    return np.row_stack(imagearrays), labels, categorydict


#******************** tf 储存 ***********************####
import tensorflow as tf

#Tf Record 写入文件的小工具
def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def writeTF(imagearrays, labels, savepath, tfname):
#通过检查label的数量是否与特征数组的第一维数是否一样，判读传入数据是否有误  
    num_examples = len(labels)
    if len(labels) != num_examples:
        raise ValueError("图片数量 %d 不匹配标签数量 %d." % 
                        (len(labels), num_examples))
    
    rows = imagearrays.shape[1]
    cols = imagearrays.shape[2]
    depth = imagearrays.shape[3]
    
    filename = os.path.join(savepath, tfname + ".tfrecords")
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        image_raw  = imagearrays[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={ 
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'label': _int64_feature(labels[index]),
            'image_raw': _bytes_feature(image_raw)}))
            
        #序列化输出到磁盘对应文件中
        writer.write(example.SerializeToString())
        
#使用示例
#labels为list name为保存的文件名
#file_path = "C:\\Users\\lenovo\\Documents\\tfstudy\\pic\\tf"
#writeTF(imagearrays, labels, file_path, "numbers2")

#******************** tf 读取 ***********************####
#读取tfRecord工具
#定义存放到tfrecord中的数据字典格式
feat={
      'image_raw': tf.FixedLenFeature([], tf.string),
      'label': tf.FixedLenFeature([], tf.int64),
      'height': tf.FixedLenFeature([], tf.int64),
      'width': tf.FixedLenFeature([], tf.int64),
      'depth': tf.FixedLenFeature([], tf.int64)}
    
#调用tf.TFRecordReader类的read函数，传人文件处理队列返回序列化对象
def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    
    _, serialized_example = reader.read(filename_queue)
    
    #解析序列化对象
    features = tf.parse_single_example(serialized_example,  features=feat)
    #解码序列化对象
    image = tf.decode_raw(features['image_raw'], tf.float64)
    
    #可选把我们原来的tf.int64数据类型转化为tf.int32
    label = tf.cast(features['label'], tf.int32)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    depth = tf.cast(features['depth'], tf.int32)
    
    return image, label ,height,width,depth

def get_all_records(filepath, shape, samplenum):
     with tf.Session() as sess:
            #tf.train.string_input_producer 可以传入多个文件名列表
            #返回一个用于文件处理的队列
            filename_queue = tf.train.string_input_producer([ filepath ])
            initop = (tf.global_variables_initializer())
            
            #调用前面的read_and_decode函数
            image, label, height, width, depth   = read_and_decode(filename_queue)
            print(image)
            
            #生成一个shape=shape的image tensor
            image = tf.reshape(image, tf.stack(shape))
            
            sess.run(initop)
            
            #tf.train.Coordinator生成一个读取文件线程的协调进程对象
            coord = tf.train.Coordinator()
            
            #开启读取文件线程
            threads = tf.train.start_queue_runners(coord=coord)
            
            featuredata=np.reshape(np.zeros(shape[0] * shape[1] * shape[2] * shape[3]),shape)
            
            labeldata=np.reshape(np.zeros(1),(1))
            
            #在当前的session下读取samplenum次，每次读取一个图片
            for i in range(samplenum):
                example, l = sess.run([image, label]) 
                featuredata=np.append(featuredata,example,axis=0)
                l=np.reshape(np.array(l),1)
                labeldata=np.append(labeldata,l,axis=0)
                
            #请求线程终止
            coord.request_stop()
            
            #等待线程完全终止
            coord.join(threads)
            
            #返回特征
            return featuredata[1:][:][:][:] , labeldata[1:]
        
#使用示例
#file_path = "C:\\Users\\lenovo\\Documents\\tfstudy\\pic\\tf"
#file_name = "numbers.tfrecords"
#shape = [1,10,12,1] #一张图片的shape
#data , label=get_all_records(os.path.join(file_path, file_name), shape, 9)
