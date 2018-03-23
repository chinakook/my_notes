# 自定义rec文件
```python
# 写自定义rec文件
train_record = mx.recordio.MXIndexedRecordIO('./train.idx', './train.rec', 'w')
np_mat = np.array([1,2,3,4], dtype=np.uint8)        
label = np.array([11,22,33,44], dtype=np.int32)        
idx = 0        
p = mx.recordio.pack_img((0,label,idx,0), np_mat)        
train_record.write_idx(idx,p)
train_record.close()        

# 读自定义rec文件
train_record = mx.recordio.MXIndexedRecordIO('./train.idx', './train.rec', 'r')
p = train_record.read_idx(idx)        
header, np_mat = mx.recordio.unpack_img(p)        
label = header.label        
train_record.close()

# 写压缩图像和分割标记图像
val_record = mx.recordio.MXIndexedRecordIO('val.idx', 'val.rec', 'w')
img = np.ones((4,5))
mask = np.zeros((4,5))
lbl = np.ones((4,5)) * 3
_,img_png = cv2.imencode('.png',img)
_,lbl_png = cv2.imencode('.png',lbl + 255-mask)
iv = 0
data=img_png.tostring() + lbl_png.tostring()
labelinfo = np.array([len(img_png), len(lbl_png)])
p = mx.recordio.pack((0,labelinfo,iv,0), data)
val_record.write_idx(iv,p)
iv = iv + 1
val_record.close()

# 读压缩图像和分割标记图像
val_record = mx.recordio.MXIndexedRecordIO('val.idx', 'val.rec', 'r')
p = val_record.read_idx(0)
header, data = mx.recordio.unpack(p)
img_enc = data[:int(header.label[0])]
lbl_enc = data[int(header.label[0]):]
img = cv2.imdecode(np.fromstring(img_enc,dtype=np.uint8), cv2.IMREAD_UNCHANGED)
lbl = cv2.imdecode(np.fromstring(lbl_enc,dtype=np.uint8), cv2.IMREAD_UNCHANGED)
val_record.close()
```

# 用numpy数组初始化常量
```python
@mx.init.register
class MyConstant(mx.init.Initializer):
    def __init__(self, value):
        super(MyConstant, self).__init__(value=value)
        self.value = value

    def _init_weight(self, _, arr):
        arr[:] = mx.nd.array(self.value)
w = mx.sym.Variable('w',shape=(1,2), init=MyConstant([class_weights]))
w = mx.sym.BlockGrad(w)
```

# Stanford_car数据集
```python
# Matlab数据格式转json文件
import scipy.io as sio
import json
annos = 'cars_annos.mat'
annomat = sio.loadmat(annos)
annotations_mat = annomat['annotations']
class_names_mat = annomat['class_names']
annotations = []
class_names = []
for c in class_names_mat[0]:
    class_names.append(c[0])
for a in annotations_mat[0]:
    rip = a[0][0]
    bbox_x1 = a[1][0][0]
    bbox_y1 = a[2][0][0]
    bbox_x2 = a[3][0][0]
    bbox_y2 = a[4][0][0]
    vclass = a[5][0][0]
    istest = a[6][0][0]
    annotations.append({"path":rip, "xmin": int(bbox_x1-1)
                        , "ymin":int(bbox_y1-1), "xmax":int(bbox_x2-1), "ymax":int(bbox_y2-1), "class": int(vclass-1), "test": int(istest)})
                        
with open('annotations.json', 'w') as f: 
    f.write(json.dumps(annotations))
with open('class_names.json', 'w') as f: 
    f.write(json.dumps(class_names))

```

# opencv处理中文路径问题
```python
img = cv2.imdecode(np.fromfile(fn, dtype=np.uint8),-1)

```

# VOC格式xml数据集
```python
# 读取和保存
from xml.etree import ElementTree as ET
anno = ET.parse('test.xml')
obj_node=anno.getiterator("object")
for obj in obj_node:
    bndbox = obj.find('bndbox')
    xmin = bndbox.find('xmin')
    ymin = bndbox.find('ymin')
    xmax = bndbox.find('xmax')
    ymax = bndbox.find('ymax')
    cls_name = obj.find('name')
    # 可改变xmin.text
# 保存xml，可覆盖原xml
anno.write('test.xml')

# 复制ElementTree
import copy
from xml.etree import ElementTree as ET
anno = ET.parse('test.xml')
anno2 = copy.deepcopy(anno)

# 写VOC xml
from xml.etree import ElementTree as ET
import codecs
def writeVOCXML(fn, fn_path, w , h, bboxes, outf):
    folder = ET.Element('folder')
    folder.text = '.'
    filename = ET.Element('filename')
    # 图像文件名
    filename.text='imagefile.jpg'
    path = ET.Element('path')
    path.text = fn_path
    source = ET.Element('source')
    database = ET.Element('database')
    source.append(database)
    database.text = 'Unknown'
    size_tag = ET.Element('size')
    width = ET.Element('width')
    # 图像宽
    width.text = str(w)
    height = ET.Element('height')
    # 图像高
    height.text = str(h)
    depth = ET.Element('depth')
    depth.text = '3'
    size_tag.append(width)
    size_tag.append(height)
    size_tag.append(depth)
    segmented = ET.Element('segmented')
    segmented.text = '0'

    root = ET.Element('annotation')
    tree = ET.ElementTree(root)
    root.append(folder)
    root.append(filename)
    root.append(path)
    root.append(source)
    root.append(size_tag)
    root.append(segmented)

    for b in bboxes:
        name = ET.Element('name')
        name.text = 'plate'
        pose = ET.Element('pose')
        pose.text = 'Unspecified'
        truncated = ET.Element('truncated')
        truncated.text = '0'
        difficult = ET.Element('difficult')
        difficult.text = '0'
        bndbox = ET.Element('bndbox')
        object_tag = ET.Element('object')
        root.append(object_tag)
        object_tag.append(name)
        object_tag.append(pose)
        object_tag.append(truncated)
        object_tag.append(difficult)
        object_tag.append(bndbox)
        xmin=ET.Element('xmin')
        ymin=ET.Element('ymin')
        xmax=ET.Element('xmax')
        ymax=ET.Element('ymax')
        # 图像bounding box
        xmin.text=str(int(b[0]))
        ymin.text=str(int(b[1]))
        xmax.text=str(int(b[2]))
        ymax.text=str(int(b[3]))
        
        bndbox.append(xmin)
        bndbox.append(ymin)
        bndbox.append(xmax)
        bndbox.append(ymax)

    def prettify(elem):
        """Return a pretty-printed XML string for the Element.
        """
        rough_string = ET.tostring(elem, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="\t")

    xml_text = prettify(root)

	# outf为输出的xml文件名
    f=codecs.open(outf,'w','utf-8')
    f.write(xml_text)
    f.close()

```

# Python NMS实现
```python
def nms(dets, prob_thresh):
    x1 = dets[:, 0]  
    y1 = dets[:, 1]  
    x2 = dets[:, 2]  
    y2 = dets[:, 3]  
    scores = dets[:, 4]  
    
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  

    order = scores.argsort()[::-1]  

    keep = []  
    while order.size > 0: 
        i = order[0]  
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])  
        yy1 = np.maximum(y1[i], y1[order[1:]])  
        xx2 = np.minimum(x2[i], x2[order[1:]])  
        yy2 = np.minimum(y2[i], y2[order[1:]])  
        w = np.maximum(0.0, xx2 - xx1 + 1)  
        h = np.maximum(0.0, yy2 - yy1 + 1)  
        inter = w * h
        
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= prob_thresh)[0]  

        order = order[inds + 1]  
    return keep
```

# PSD文件读取
```python
import os
import numpy as np
import psd_tools
import PIL

psd_img = psd_tools.PSDImage.load('test.psd')
psd_img = psd_tools.PSDImage.load(psd_file)
fg = psd_img.layers[0]
bg = psd_img.layers[1]
fg_img = np.array(fg.as_PIL())
mask = np.zeros((bg.bbox.y2, bg.bbox.x2), dtype=np.uint8)
mask[fg.bbox.y1:fg.bbox.y2, fg.bbox.x1:fg.bbox.x2][:] = fg_img[:,:,3]
```

# Conv与BN合并
```python
import numpy as np
import mxnet as mx
import json
sym, args, auxs = mx.model.load_checkpoint('mbn',0)
graph=json.loads(sym.tojson())
args_deploy = {}
auxs_deploy = {}

merge_dict = {}
for i, n in enumerate(graph['nodes']):
    if n['op'] == 'BatchNorm':
        pre_layer =  graph['nodes'][n['inputs'][0][0]]
        if pre_layer['op'] == 'Convolution':
            merge_dict[pre_layer['name']] = i

for i, n in enumerate(graph['nodes']):
    if n['op'] == 'Convolution':
        if merge_dict.has_key(n['name']):
            bn = graph['nodes'][merge_dict[n['name']]]
            gamma = args[bn['name']+'_gamma']
            beta = args[bn['name']+'_beta']
            moving_mean = auxs[bn['name']+'_moving_mean']
            moving_var = auxs[bn['name']+'_moving_var']
            eps = float(bn['attrs']['eps'])

            weight = args[n['name']+'_weight']
            if not n['attrs'].has_key('no_bias') or n['attrs']['no_bias'] == 'False':
                bias = args[n['name']+'_bias']
            else:
                bias = mx.nd.zeros((weight.shape[0],))              
            a = gamma / mx.nd.sqrt(moving_var + eps)
            b = beta - a * moving_mean
            a = mx.nd.reshape(a,(-1,1,1,1))
            weight = weight * a
            bias = bias + b
            args_deploy[n['name'] + '_weight'] = weight
            args_deploy[n['name'] + '_bias'] = bias
        else:
            args_deploy[n['name'] + '_weight'] = args[n['name']+'_weight']
            if not n['attrs'].has_key('no_bias') or n['attrs']['no_bias'] == 'False':
                args_deploy[n['name'] + '_bias'] = args[n['name']+'_bias']
    elif n['op'] == 'FullyConnected':
        args_deploy[n['name']+'_weight'] = args[n['name']+'_weight']
        if not n['attrs'].has_key('no_bias') or n['attrs']['no_bias'] == 'False':
            args_deploy[n['name']+'_bias'] = args[n['name']+'_bias']
```

# MX获取各层参数形状
```python
arg_shapes, _, aux_shapes = sym.infer_shape(data=(1,3,224,224))
arg_names = fusex.list_arguments()
aux_names = fusex.list_auxiliary_states()
arg_shape_dic = dict(zip(arg_names, arg_shapes))
aux_shape_dic = dict(zip(aux_names, aux_shapes))
```

# MX构造反卷积二次线性核
```python
def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1.0
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
sz = arg_shape_dic['dconv0_weight']
deconv_weight = np.zeros(sz)
deconv_weight[range(sz[0]), range(sz[1]),:,:] = upsample_filt(sz[3])
```

# Matconvnet模型转MXNet
```pyton
import scipy.io as sio

# tinyface的人脸检测模型举例
matpath='./imagenet-resnet-101-dag.mat'
net = sio.loadmat(matpath)
layers = net['layers'][0]

layers_hyper_params_dict = {}
mat_params = net['params'][0]
mat_params_dict = {}
for l in layers:
    if l[1][0] == 'dagnn.BatchNorm':
        layers_hyper_params_dict[l[0][0]] = l[5][0][0][1][0][0]

for p in mat_params:
    mat_params_dict[p[0][0]] = p[1]

# res4b22_relu需要事先定义好，这里通过caffe_converter得到
graph=json.loads(res4b22_relu.tojson())

for i, n in enumerate(graph['nodes']):
    if n['op'] == 'Convolution':
        wmat = mat_params_dict[n['name']+'_filter'].copy()
        wmat = np.transpose(wmat, [3,2,0,1]) # matlab array is (h w c n) so need to swap axes
        arg_params[n['name']+'_weight'] = mx.nd.array(wmat)
        if not n['attrs'].has_key('no_bias') or n['attrs']['no_bias'] == 'False':
            bias = mat_params_dict[layer_name+'_bias'][0]
            arg_params[n['name']+'_bias'] = mx.nd.array(bias)
    elif n['op'] == 'BatchNorm':
        gamma = mat_params_dict[n['name']+'_mult'][:,0]
        beta = mat_params_dict[n['name']+'_bias'][:,0]
        moments = mat_params_dict[n['name']+'_moments']
        moving_mean = moments[:,0]
        epslion = layers_hyper_params_dict[n['name']]
        moving_var = moments[:,1] * moments[:,1] - epslion
        arg_params[n['name']+'_gamma'] = mx.nd.array(gamma)
        arg_params[n['name']+'_beta'] = mx.nd.array(beta)
        aux_params[n['name']+'_moving_mean'] = mx.nd.array(moving_mean)
        aux_params[n['name']+'_moving_var'] = mx.nd.array(moving_var)
    else:
        pass
```

# MX快速做出分割数据集
```pyton
import os
import cv2
import numpy as np
train_path = './data/train'
gt_path = './data/train_label'

proc_src = lambda fn: np.transpose(cv2.cvtColor(cv2.imread(fn),cv2.COLOR_BGR2RGB), (2, 0, 1))
proc_gt = lambda fn: cv2.cvtColor(cv2.imread(fn),cv2.COLOR_BGR2GRAY)[np.newaxis,:]
imgs = np.array([proc_src(train_path+'/'+i) for i in os.listdir(train_path) if i.endswith('.png')],dtype=np.float32)
gts = np.array([proc_gt(gt_path+'/'+i) for i in os.listdir(gt_path) if i.endswith('.png')],dtype=np.float32)

nd_iter = mx.io.NDArrayIter(data={'data':imgs}, label={'y': gts}, batch_size=16)
```

# MX Forward
```python
import mxnet as mx
import numpy as np
from PIL import Image
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])
# 以U-Net举例
sym, arg_params, aux_params = mx.model.load_checkpoint('u_net', 59)

context=mx.gpu()
mod = mx.mod.Module(symbol=all_layers['pred_output'], context=context, data_names=['data'], label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (1, 3, 500, 500))], label_shapes=None, force_rebind=False)
mod.set_params(arg_params=arg_params, aux_params=aux_params, force_init=False)

im_raw = Image.open('test.png')
im = np.array(im_raw, dtype=np.float32)
im = im[np.newaxis,np.newaxis,:,:]

# reshape to process variant image size
mod.reshape(data_shapes=[('data', (1, 1, im.shape[2], im.shape[3]))])

# 当有多个输入时，batch构造方式
# batch = mx.io.DataBatch([mx.nd.array(im_tensor), mx.nd.array(im_info)])
mod.forward(Batch([mx.nd.array(im)]), is_train=False)

f = mod.get_outputs()[0].asnumpy()
fore_ground = f[0,1,:,:]
pred_label = (f1>0.5)*255
pred_label = res.astype(np.uint8)
```

# SSD检测时的前处理和后处理
```python
# 前处理
data_shape = 300

h = raw_img.shape[0]
w = raw_img.shape[1]

if w > h:
    short_dim = int(h * data_shape / float(w))
    raw_img2 = cv2.resize(raw_img, (data_shape, short_dim))
    pad = data_shape - short_dim
    raw_img2 = cv2.copyMakeBorder(raw_img2, pad//2, (pad+1)//2, 0, 0, cv2.BORDER_CONSTANT)
    scale = data_shape / float(w)
else:
    short_dim = int(w * data_shape / float(h))
    raw_img2 = cv2.resize(raw_img, (short_dim, data_shape))
    pad = data_shape - short_dim
    raw_img2 = cv2.copyMakeBorder(raw_img2, 0, 0, pad//2, (pad+1)//2, cv2.BORDER_CONSTANT)
    scale = data_shape / float(h)

raw_img2 = cv2.cvtColor(raw_img2, cv2.COLOR_BGR2RGB)
raw_img2 = cv2.resize(raw_img2, (data_shape,data_shape))

img = np.transpose(raw_img2, (2,0,1))
img = img[np.newaxis, :]
img = cls_std_scale * (img.astype(np.float32) - cls_mean_val)

# 后处理
detections = mod.get_outputs()[0].asnumpy()

res = None
for i in range(detections.shape[0]):
    det = detections[i, :, :]
    res = det[np.where(det[:, 0] >= 0)[0]]

final_dets = np.empty(shape=(0, 6))

for i in range(res.shape[0]):
    cls_id = int(res[i, 0])
    if cls_id >= 0:
        score = res[i, 1]
        if score > 0.61:
            xmin = int(res[i, 2] * data_shape)
            ymin = int(res[i, 3] * data_shape)
            xmax = int(res[i, 4] * data_shape)
            ymax = int(res[i, 5] * data_shape)

            if w > h:
                pad = w - h
                # scale即前处理时的scale
                xmin2 = xmin / scale
                xmax2 = xmax / scale
                ymin2 = ymin / scale - (pad // 2)
                ymax2 = ymax / scale - (pad // 2)
            else:
                pad = h - w
                xmin2 = xmin / scale - (pad // 2)
                xmax2 = xmax / scale - (pad // 2)
                ymin2 = ymin / scale
                ymax2 = ymax / scale


            final_dets = np.vstack((final_dets, [xmin2, ymin2, xmax2, ymax2, score,cls_id]))
```

# 随机仿射变换图像扩增处理
```python
x0 = header.label[1]
y0 = header.label[2]
x1 = header.label[3]
y1 = header.label[4]

dst_w = self.image_shape
dst_h = self.image_shape
obj_w = x1 - x0
obj_h = y1 - y0
xc = (x0 + x1) / 2.
yc = (y0 + y1) / 2.

scale_w = dst_w / float(obj_w)
scale_h = dst_h / float(obj_h)

if self.aug:
    sj = random.uniform(0.9, 1.2)
    scale_jitter_w = sj * random.choice([-1,1])
    scale_jitter_h = sj * random.choice([-1,1])

    scale_w *= scale_jitter_w
    scale_h *= scale_jitter_h

    dst_w_jitter = random.uniform(-15,15)
    dst_h_jitter = random.uniform(-10,10)
    theta = random.uniform(-np.pi/45, np.pi/45)

    T0 = np.array([[1,0,-xc],[0,1,-yc],[0,0,1]])
    S = np.array([[scale_w,0,0],[0, scale_h,0],[0,0,1]])
    R = np.array([[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0],[0,0,1]])
    T1 = np.array([[1,0,dst_w/2. +dst_w_jitter],[0,1,dst_h/2. +dst_h_jitter],[0,0,1]])
    M = np.dot(S, T0)
    M = np.dot(R, M)
    M = np.dot(T1, M)
    M_warp = M[0:2,:]

    dst_img = cv2.warpAffine(img, M_warp, dsize=(int(dst_w), int(dst_h)))
else:
    dst_img = img[int(y0):int(y1)+1, int(x0):int(x1)+1, :]
    dst_img = cv2.resize(dst_img, (int(dst_w), int(dst_h)))

dst_img = cv2.cvtColor(dst_img, cv2.COLOR_BGR2RGB)
dst_img = np.transpose(dst_img,(2,0,1))
dst_img = std_scale * (dst_img.astype(np.float32) - mean_val)
```

# 为SSD制作数据IMDB
```python
import os
import cv2
import json
import numpy as np
from imdb import Imdb

# 以压力表数据为例
class ylbdb(Imdb):
    def __init__(self, name, root_path, is_train = False):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, 'zp')
        self.extension = '.jpg'
        self.is_train = is_train
        self.classes = self._load_class_names('names.txt',root_path)
        self.num_classes = len(self.classes)

        if self.is_train:
            self.image_set_index, self.labels = self._load_ylb_dat()
        else:
            self.image_set_index, _ = self._load_ylb_dat()
        self.num_images = len(self.image_set_index)

    def _load_ylb_dat(self):
        with open(self.data_path + '/bj2.json') as f:
            db = json.load(f)

        imgfiles = [i for i in os.listdir(self.data_path) if i.endswith('.jpg')]

        db_dict = {}

        for i,j in enumerate(db):
            fn = os.path.split(j['filename'])[1]
            db_dict[fn] = i

        labels = []

        for i in imgfiles:
            img = cv2.imread(self.data_path + '/' + i)
            
            h = img.shape[0]
            w = img.shape[1]

            idx = db_dict[i]
            info = db[idx]
            anno = info['annotations']

            label = []

            x0 = info['rect']['x0']
            y0 = info['rect']['y0']
            x1 = info['rect']['x1']
            y1 = info['rect']['y1']

            x0 = x0 / float(w)
            y0 = y0 / float(h)
            x1 = x1 / float(w)
            y1 = y1 / float(h)

			# 只有1个类，故label第一个元素为0
            label.append([0, x0, y0, x1, y1])
            labels.append(np.array(label))

        imgfiles = [self.data_path + '/' + i for i in imgfiles]

        return imgfiles, labels

    def image_path_from_index(self, index):
        return self.image_set_index[index]

    def label_from_index(self, index):
        return self.labels[index]
```

# 利用pickle读写numpy数据
```python
# 写
import numpy as np
import pickle
meta_file = open('meta.pkl', 'wb')
# cluseters和averageImage均为numpy数组
pickle.dump(clusters, meta_file, 1)
pickle.dump(averageImage, meta_file, 1)
meta_file.close()

# 读
meta_file = open('meta.pkl', 'rb')
clusters = pickle.load(meta_file)
averageImage = pickle.load(meta_file)
meta_file.close()

```

# 原图与分割图像拼接并合成视频
```python
import os
import cv2
import numpy as np

inputs_ori = os.listdir('origin')
inputs_seg = os.listdir('color')
video = cv2.VideoWriter('ad.avi',cv2.cv.CV_FOURCC('I','4','2','0'), 24, (1469,614))
for i, fn in enumerate(inputs_ori):
    img_ori = cv2.imread('origin/'+inputs_ori[i])
    img_seg = cv2.imread('color/'+inputs_seg[i])
    
    img_res = np.zeros((img.shape[0],img.shape[1]*2,3), dtype=np.uint8)
    img_res[:,:img.shape[1],:]=img_ori.copy()
    img_res[:,img.shape[1]:,:]=img_seg.copy()

    img_res = cv2.resize(img_res, (0,0), fx=0.3, fy=0.3)

    video.write(img_res)
video.release()
```

# FCN从前面引出的层初始化方式
```python
# 以tinyface的FCN为例，全零初始化
arg_params['score_res4_weight'] = mx.nd.zeros(shape=(arg_shape_dic['score_res4_weight']))
arg_params['score_res4_bias'] = mx.nd.zeros(shape=(arg_shape_dic['score_res4_bias']))

arg_params['score_res3_weight'] = mx.nd.zeros(shape=(arg_shape_dic['score_res3_weight']))
arg_params['score_res3_bias'] = mx.nd.zeros(shape=(arg_shape_dic['score_res3_bias']))
```

# MX Xavier初始化
```python
def my_xavier(shape):
    hw_scale = np.prod(shape[2:])
    fan_in, fan_out = shape[1] * hw_scale, shape[0] * hw_scale
    return np.sqrt(6/float(fan_in+fan_out))
```

# MX Symbol反向传播小实验
```python
# softmax
x = mx.nd.array([[[[1,2,3],[4,5,6],[7,8,9]],[[2,3,4],[5,2,3],[3,4,2]]]])
l = mx.nd.array([[[[1,0,1],[0,0,1],[1,1,0]]]])
vm=mx.sym.Variable('m')
vn=mx.sym.Variable('n')
dx = mx.nd.empty(x.shape)
out=mx.sym.SoftmaxOutput(data=vm,label=vn,grad_scale=1, ignore_label=225, multi_output=True, use_ignore=True)
exec_ = out.bind(mx.cpu(),{'m':x,'n':l}, args_grad={'m': dx})
exec_.forward()
print(exec_.outputs[0].asnumpy())
exec_.backward(out_grads=mx.nd.ones_like(dx))
print(exec_.grad_arrays)

# smooth_l1
x = mx.nd.array([[[[1,2,3],[4,5,6],[7,8,9]],[[2,3,4],[5,2,3],[3,4,2]]]])
l = mx.nd.array([[[[1,0,2],[1,2,3],[5,4,3]],[[1,1,1],[6,8,9],[13,14,12]]]])
vm=mx.sym.Variable('m')
vn=mx.sym.Variable('n')
dx = mx.nd.empty(x.shape)
out=mx.symbol.smooth_l1(vm-vn, scalar=1.0)
out = mx.symbol.MakeLoss(data=out,grad_scale=1. ,normalization='valid')
exec_ = out.bind(ctx=mx.cpu(), args={'m':x, 'n':l}, args_grad={'m': dx})
exec_.forward()
print(exec_.outputs[0].asnumpy())
exec_.backward(out_grads=mx.nd.ones_like(dx))
print(exec_.grad_arrays)
```

# KMeans聚类二维点并显示
```python
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin

# 以聚类矩形框为例
def cluster_rects(total_rects, N, minsz=(10,10), maxsz=(np.inf, np.inf)):

    hs = total_rects[:,3] - total_rects[:,1]+1
    ws = total_rects[:,2] - total_rects[:,0]+1

    total_rects = np.hstack((np.expand_dims(ws,1), np.expand_dims(hs,1)))

    idx = (ws >= minsz[0]) & (hs >= minsz[1]) & (ws <= maxsz[0]) & (hs <=maxsz[1])

    total_rects = total_rects[idx,:]

    total_rects = np.array(random.sample(total_rects, np.minimum(total_rects.shape[0], 100000L)))

    k_means = KMeans(init='k-means++', n_clusters=N, n_init=10)

    k_means.fit(total_rects)

    k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
    k_means_labels = pairwise_distances_argmin(total_rects, k_means_cluster_centers)

    fig = plt.figure(figsize=(8, 3))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
    colors = ['#%06X' % int(np.floor(random.random()*0xFFFFFF)) for i in range(N)]
    #colors = ['#4EACC5', '#FF9C34', '#4E9A06', '#218868', '#FF4500']
    ax = fig.add_subplot(1, 3, 1)
    for k, col in zip(range(N), colors):
        my_members = k_means_labels == k
        cluster_center = k_means_cluster_centers[k]
        ax.plot(total_rects[my_members, 0], total_rects[my_members, 1], 'w',
                markerfacecolor=col, marker='.')
        ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                markeredgecolor='k', markersize=6)
    ax.set_title('KMeans')
    ax.set_xticks(())
    ax.set_yticks(())
    plt.show()
    return k_means_cluster_centers[::-1,:]
```

# 感受野计算
```python
import json

def filter_rf(k, s, p):
    return k, s, (k-1)/2. - p

def deconv_rf(k, u, c):
    beta = (2 * c - k + 1) / (2. * u)
    return (k - 1)/float(u) + 1, 1 / float(u), beta

def compose_rfs(rfb, sb, ob, rft, st, ot):
    s = sb * st
    offset = sb * ot + ob
    rfsize = sb * (rft - 1) + rfb
    return rfsize, s, offset

def overlay_rfs(rf0, s0, o0, rf1, s1, o1):
    assert s0 == s1, "two strides must be the same."
    s = s0
    a = min(o0 - (rf0 - 1)/2., o1 - (rf1 - 1)/2.)
    b = max(o0 + (rf0 - 1)/2., o1 + (rf1 - 1)/2.)
    return b - a + 1, s, (a + b) / 2.

def looks_like_weight(name):
    """Internal helper to figure out if node should be hidden with `hide_weights`.
    """
    if name.endswith("_weight"):
        return True
    if name.endswith("_bias"):
        return True
    if name.endswith("_beta") or name.endswith("_gamma") or name.endswith("_moving_var") or name.endswith("_moving_mean"):
        return True
    return False


def rf_summery(smb):
    conf = json.loads(smb.tojson())
    nodes = conf["nodes"]

    hidden_nodes = set()
    for node in nodes:
        op = node["op"]
        name = node["name"]
        if op == "null" and looks_like_weight(node["name"]):
            hidden_nodes.add(node["name"])
        elif op == "null" and name == "data":
            node["meta"] = {}
            node["meta"]["rf"] = 1
            node["meta"]["stride"] = 1
            node["meta"]["offset"] = 0
        elif op == "Convolution" or op == "Pooling":
            k = int(node["attrs"]["kernel"][1])
            if node["attrs"].has_key("stride"):
                s = int(node["attrs"]["stride"][1])
            else:
                s = 1
            if node["attrs"].has_key("pad"):
                p = int(node["attrs"]["pad"][1])
            else:
                p = 0
            rf, stride, offset = filter_rf(k,s,p)
            node["meta"] = {}
            node["meta"]["rf"] = rf
            node["meta"]["stride"] = stride
            node["meta"]["offset"] = offset
        elif op == "Activation" or op == "BatchNorm":
            node["meta"] = {}
            node["meta"]["rf"] = 1
            node["meta"]["stride"] = 1
            node["meta"]["offset"] = 0
        elif op == "Deconvolution":
            k = int(node["attrs"]["kernel"][1])
            u = int(node["attrs"]["stride"][1])
            c = int(node["attrs"]["pad"][1])
            rf, stride, offset = deconv_rf(k,u,c)
            print(rf, stride, offset)
            node["meta"] = {}
            node["meta"]["rf"] = rf
            node["meta"]["stride"] = stride
            node["meta"]["offset"] = offset
        else:
            node["meta"] = {}
            node["meta"]["rf"] = 1
            node["meta"]["stride"] = 1
            node["meta"]["offset"] = 0

    for node in nodes:
        op = node["op"]
        name = node["name"]
        if name in hidden_nodes:
            continue
        else:
            inputs = node["inputs"]
            input_nodes = []
            for item in inputs:
                input_node = nodes[item[0]]
                if input_node["name"] in hidden_nodes:
                    continue
                input_nodes.append(input_node)
        
            if op == "Convolution" or op == "Pooling" or op == "Deconvolution" 
                    or op == "Activation" or op == "BatchNorm" or op == "slice":
                assert len(input_nodes) == 1, "Filter layer inputs count should be 1."
                rf0 = input_nodes[0]["meta"]["rf"]
                stride0 = input_nodes[0]["meta"]["stride"]
                offset0 = input_nodes[0]["meta"]["offset"]

                rf = node["meta"]["rf"]
                stride = node["meta"]["stride"]
                offset = node["meta"]["offset"]

                rf_c, stride_c, offset_c = compose_rfs(rf0, stride0, offset0, rf, stride, offset)
                node["meta"]["rf"] = rf_c
                node["meta"]["stride"] = stride_c
                node["meta"]["offset"] = offset_c

                print(name, rf_c, stride_c, offset_c)
            elif op == "broadcast_add" or op == "_Plus" or op == "elemwise_add" or op == "Crop":
                assert len(input_nodes) == 2, "ElemetWise layer inputs count should be 2."
                rf0 = input_nodes[0]["meta"]["rf"]
                stride0 = input_nodes[0]["meta"]["stride"]
                offset0 = input_nodes[0]["meta"]["offset"]
                rf1 = input_nodes[1]["meta"]["rf"]
                stride1 = input_nodes[1]["meta"]["stride"]
                offset1 = input_nodes[1]["meta"]["offset"]
                
                rf_over, stride_over, offset_over = overlay_rfs(rf0, stride0, offset0, rf1, stride1, offset1)
                node["meta"]["rf"] = rf_over
                node["meta"]["stride"] = stride_over
                node["meta"]["offset"] = offset_over
                print(name, rf_over, stride_over, offset_over)
```

# Flask模型预测服务
```python
# 服务端
from flask import Flask, request
from werkzeug.utils import secure_filename
import hashlib
import cv2
import numpy as np
import datetime
from test_one import *

app = Flask(__name__)

def mx_predict(file_location):
    fname = mx.test_utils.download(file_location, dirname="static/img_pool")
    raw_img = cv2.imread(fname)

    # ......

    retstr = "%d, %f, (%d, %d, %d, %d)" % (ylb_type, pred[0, ylb_type], xmin, ymin, xmax, ymax)

    return retstr
    

ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg', 'bmp']
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/upload_image", methods = ['POST'])
def FUN_upload_image():

    if request.method == 'POST':
        print("img uploaded")
        # check if the post request has the file part
        if 'file' not in request.files:
            return ""
        file = request.files['file']

        # if user does not select file, browser also submit a empty part without filename
        if file.filename == '':
            return ""

        if file and allowed_file(file.filename):
            filename = os.path.join("static/img_pool", hashlib.sha256(str(datetime.datetime.now())).hexdigest() 
                + secure_filename(file.filename).lower())
            filename=filename.replace('\\','/')
            file.save(filename)
            prediction_result = mx_predict(filename)
            return prediction_result

        return ""

################################################
# Start the service
################################################
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
    
# 客户端，以压力表项目为例
import requests
url = "http://192.168.20.219:5000/upload_image"
files = {'file': open("../ylb_det/222.jpg", 'rb')}
r = requests.post(url, files = files)
```

# Symbol 动态利用shape
```python
vm=mx.sym.Variable('m',shape=(1,2,3,3))
out=vm+vm
_,out_shape,_=out.infer_shape_partial()
kk = mx.sym.arange(out_shape[0][3])

x = mx.nd.array([[[[1,2,3],[4,5,6],[7,8,9]],[[2,3,4],[5,2,3],[3,4,2]]]])
l = mx.nd.array([[[[1,0,1],[0,0,1],[1,1,0]]]])

exec_ = kk.bind(mx.cpu(),{'m':x})
exec_.forward()
```