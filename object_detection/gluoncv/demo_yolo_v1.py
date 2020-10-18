from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt
import mxnet as mx

# profiler
from mxnet import profiler
ctx = mx.gpu(0)

profiler.set_config(profile_all=True,
                    aggregate_stats=True,
                    continuous_dump=True,
                    filename='demo_yolo_v1.json')

net = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True)
net.collect_params().reset_ctx([ctx])



im_fname = utils.download('https://raw.githubusercontent.com/zhreshold/' +
                          'mxnet-ssd/master/data/demo/dog.jpg',
                          path='dog.jpg')

x, img = data.transforms.presets.yolo.load_test(im_fname, short=512)
print('Shape of pre-processed image:', x.shape)

x = x.as_in_context(ctx)
class_IDs, scores, bounding_boxs = net(x)

ax = utils.viz.plot_bbox(img, bounding_boxs[0], scores[0],
                         class_IDs[0], class_names=net.classes)
plt.show()