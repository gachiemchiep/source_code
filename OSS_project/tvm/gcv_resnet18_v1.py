import argparse

import mxnet as mx
from mxnet import nd, image
import time
import gluoncv as gcv
from tvm.contrib.download import download_testdata
from gluoncv.data import ImageNet1kAttr
from gluoncv.data.transforms.presets.imagenet import transform_eval
from gluoncv.model_zoo import get_model

ctx = mx.gpu(0)

net = get_model("resnet18_v1", pretrained=True)
net.collect_params().reset_ctx(ctx)

block = get_model('resnet18_v1', pretrained=True)
img_url = 'https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true'
img_name = 'cat.png'
synset_url = ''.join(['https://gist.githubusercontent.com/zhreshold/',
                      '4d0b62f3d01426887599d4f7ede23ee5/raw/',
                      '596b27d23537e5a1b5751d2b0481ef172f58b539/',
                      'imagenet1000_clsid_to_human.txt'])
synset_name = 'imagenet1000_clsid_to_human.txt'
img_path = download_testdata(img_url, 'cat.png', module='data')
synset_path = download_testdata(synset_url, synset_name, module='data')
with open(synset_path) as f:
    synset = eval(f.read())

# Load Images
img = image.imread(img_path)

# Transform
img = transform_eval(img)
img = img.as_in_context(ctx)

run_cnt = 1000
start_time = time.time()
for i in range(run_cnt):
    pred = net(img)

end_time = time.time()
inference_time = (end_time - start_time) / float(run_cnt) * 1000 # ms
print("Inference time: {:6.1f} ms".format(inference_time))

topK = 5
ind = nd.topk(pred, k=topK)[0].astype('int')
print('The input picture is classified to be')
for i in range(topK):
    print('\t[%s], with probability %.3f.'%
          (synset[ind[i].asscalar()], nd.softmax(pred)[0][ind[i]].asscalar()))
