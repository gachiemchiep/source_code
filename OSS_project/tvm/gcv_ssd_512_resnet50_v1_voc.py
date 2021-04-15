import mxnet as mx
from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt
import time

ctx = mx.gpu()

net = model_zoo.get_model('ssd_512_resnet50_v1_voc', pretrained=True)
net.collect_params().reset_ctx(ctx)


im_fname = utils.download('https://github.com/dmlc/web-data/blob/master/' +
                          'gluoncv/detection/street_small.jpg?raw=true',
                          path='street_small.jpg')

x, img = data.transforms.presets.ssd.load_test(im_fname, short=512)
# print('Shape of pre-processed image:', x.shape)
x = x.as_in_context(ctx)

# warming up
net(x)

run_cnt = 1000
start_time = time.time()
for i in range(run_cnt):
    class_IDs, scores, bounding_boxes = net(x)

end_time = time.time()
inference_time = (end_time - start_time) / float(run_cnt) * 1000 # ms
print("Inference time: {:6.1f} ms".format(inference_time))

# ax = utils.viz.plot_bbox(img, bounding_boxes[0], scores[0],
#                          class_IDs[0], class_names=net.classes)
# plt.show()
