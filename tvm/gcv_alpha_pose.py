
from matplotlib import pyplot as plt
from gluoncv import model_zoo, data, utils
from gluoncv.data.transforms.pose import detector_to_alpha_pose, get_max_pred
# , heatmap_to_coord_alpha_pose
import mxnet as mx
from mxnet import nd, image
import time
import numpy  as np

def transformBoxInvert(pt, ul, br, resH, resW):
    # type: (Tensor, Tensor, Tensor, float, float, float, float) -> Tensor

    center = mx.nd.zeros(2, ctx=ctx)
    center[0] = (br[0] - 1 - ul[0]) / 2
    center[1] = (br[1] - 1 - ul[1]) / 2

    lenH = max(br[1] - ul[1], (br[0] - ul[0]) * resH / resW)
    lenW = lenH * resW / resH

    _pt = (pt * lenH) / resH

    if bool(((lenW - 1) / 2 - center[0]) > 0):
        _pt[0] = _pt[0] - ((lenW - 1) / 2 - center[0]).asscalar()
    if bool(((lenH - 1) / 2 - center[1]) > 0):
        _pt[1] = _pt[1] - ((lenH - 1) / 2 - center[1]).asscalar()

    new_point = mx.nd.zeros(2)
    new_point[0] = _pt[0] + ul[0]
    new_point[1] = _pt[1] + ul[1]
    return new_point

def heatmap_to_coord_alpha_pose(hms, boxes):

    t0 = time.time()
    hm_h = hms.shape[2]
    hm_w = hms.shape[3]
    coords, maxvals = get_max_pred(hms)
    t1 = time.time()
    print("A: {}".format(t1 - t0))

    if boxes.shape[1] == 1:
        pt1 = mx.nd.array(boxes[:, 0, (0, 1)], dtype=hms.dtype, ctx=ctx)
        pt2 = mx.nd.array(boxes[:, 0, (2, 3)], dtype=hms.dtype, ctx=ctx)
    else:
        assert boxes.shape[1] == 4
        pt1 = mx.nd.array(boxes[:, (0, 1)], dtype=hms.dtype, ctx=ctx)
        pt2 = mx.nd.array(boxes[:, (2, 3)], dtype=hms.dtype, ctx=ctx)

    # post-processing
    print(hms.shape)
    print(coords.shape)
    for n in range(coords.shape[0]):
        for p in range(coords.shape[1]):
            hm = hms[n][p]
            px = int(nd.floor(coords[n][p][0] + 0.5).asscalar())
            py = int(nd.floor(coords[n][p][1] + 0.5).asscalar())
            if 1 < px < hm_w - 1 and 1 < py < hm_h - 1:
                diff = nd.concat(hm[py][px + 1] - hm[py][px - 1],
                                 hm[py + 1][px] - hm[py - 1][px],
                                 dim=0)
                coords[n][p] += nd.sign(diff) * .25
    t2 = time.time()
    print("B: {}".format(t2 - t1))

    preds = nd.zeros_like(coords)
    for i in range(hms.shape[0]):
        for j in range(hms.shape[1]):
            preds[i][j] = transformBoxInvert(coords[i][j], pt1[i], pt2[i], hm_h, hm_w)
    t3 = time.time()
    print("C: {}".format(t3 - t2))

    print(preds.shape)
    print(maxvals.shape)
    print(maxvals)
    return preds, maxvals

ctx = mx.gpu()

detector = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True)
pose_net = model_zoo.get_model('alpha_pose_resnet101_v1b_coco', pretrained=True)

detector.reset_class(["person"], reuse_weights=['person'])
detector.collect_params().reset_ctx(ctx)
pose_net.collect_params().reset_ctx(ctx)

im_fname = utils.download('https://github.com/dmlc/web-data/blob/master/' +
                          'gluoncv/pose/soccer.png?raw=true',
                          path='soccer.png')
x, img = data.transforms.presets.yolo.load_test(im_fname, short=512)
print('Shape of pre-processed image:', x.shape)
x_ctx = x.as_in_context(ctx)

# warming up
class_IDs, scores, bounding_boxs = detector(x_ctx)
pose_input, upscale_bbox = detector_to_alpha_pose(img, class_IDs, scores, bounding_boxs, ctx=ctx)

pose_input = pose_input.as_in_context(ctx)
predicted_heatmap = pose_net(pose_input)
pred_coords, confidence = heatmap_to_coord_alpha_pose(predicted_heatmap, upscale_bbox)

# Benchmark
run_cnt = 100
detector_time = []
pose_time = []
for i in range(run_cnt):
    t1 = time.time()
    x_ctx = x.as_in_context(ctx)
    t2 = time.time()
    class_IDs, scores, bounding_boxs = detector(x_ctx)
    pose_input, upscale_bbox = detector_to_alpha_pose(img, class_IDs, scores, bounding_boxs, ctx=ctx)
    t3 = time.time()
    pose_input = pose_input.as_in_context(ctx)
    t4 = time.time()

    predicted_heatmap = pose_net(pose_input)
    pred_coords, confidence = heatmap_to_coord_alpha_pose(predicted_heatmap, upscale_bbox)

    pose_time.append(t4 - t3)
    detector_time.append(t2 - t1)

detector_inference_time = np.sum(np.array(detector_time)) / run_cnt * 1000 #ms
pose_inference_time = np.sum(np.array(pose_time)) / run_cnt * 1000         #ms

print("Detector: {:4.2f} ms".format(detector_inference_time))
print("Pose: {:4.2f} ms".format(pose_inference_time))
######################################################################
# Display the pose estimation results
# ---------------------
#
# We can use :py:func:`gluoncv.utils.viz.plot_keypoints` to visualize the
# results.

ax = utils.viz.plot_keypoints(img, pred_coords, confidence,
                              class_IDs, bounding_boxs, scores,
                              box_thresh=0.5, keypoint_thresh=0.2)
plt.show()
