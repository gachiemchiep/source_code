###############################################
##      Yolov4 + Open CV and Numpy integration        ##
###############################################

import numpy as np
import cv2
import argparse
import time

def get_args():
    parser = argparse.ArgumentParser(description='Benchmark yolov4.')
    parser.add_argument('--batch_size', help='Batch size.', type=int, default=1)
    parser.add_argument('--mode', help='fp32 or fp16', choices=['fp32', 'fp16'], default='fp32')
    parser.add_argument('--cfg', help='yolov4 cfg', required=True)
    parser.add_argument('--classes', help='class names', required=True)
    parser.add_argument('--weight', help='yolo weight file', required=True)
    parser.add_argument('--shape', help='yolo input shape', choices=[320, 416, 512, 608], default=608, type=int)
    parser.add_argument('--conf', help='confidence threshold', default=0.5)
    parser.add_argument('--nms', help='nms threshold', default=0.6)
    args = parser.parse_args()

    return args

def init_net(args):

    # Load a network
    net = cv2.dnn.readNet(args.cfg, args.weight, 'darknet')
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    if args.mode == 'fp32':
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    else:
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    outNames = net.getUnconnectedOutLayersNames()

    with open(args.classes, 'rt') as f:
        class_names = f.read().rstrip('\n').split('\n')

    return net, outNames, class_names

def warm_up_net(net, outNames, args):

    frame = (np.random.standard_normal([args.shape, args.shape, 3]) * 255).astype(np.uint8)
    images = []
    for i in range(args.batch_size):
        images.append(frame)
    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImages(images, size=(args.shape, args.shape), swapRB=True, ddepth=cv2.CV_8U)

    # Run a model
    net.setInput(blob, scalefactor=1/255.0)

    outs = net.forward(outNames)

def drawPred(frame, class_name, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0))

        label = '%s: %.2f' % (class_name, conf)

        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv2.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

def postprocess(frame, out, confThreshold, nmsThreshold, class_names):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    
    classIds = []
    confidences = []
    boxes = []
    
    out_bboxes = out[:, 0:4]
    out_bbox_confs = out[:, 4]
    out_class_confs = out[:, 5:]
            
    out_class_confs_max_ind = np.argmax(out_class_confs, axis=1)
    out_class_confs_max = out_class_confs[np.arange(len(out_class_confs)), out_class_confs_max_ind]
    
    # Get conf_max mask
    out_class_confs_max_mask = out_class_confs_max[:] >= confThreshold
    
    # Remove un-masking bboxes, conf_max (class score), classIds
    out_bboxes_cut = out_bboxes[out_class_confs_max_mask, :]
    out_class_confs_max_cut = out_class_confs_max[out_class_confs_max_mask]
    out_class_confs_max_ind_cut = out_class_confs_max_ind[out_class_confs_max_mask]
    
    for idx in range(out_bboxes_cut.shape[0]):
        center_x = int(out_bboxes_cut[idx, 0] * frameWidth)
        center_y = int(out_bboxes_cut[idx, 1] * frameHeight)
        width = int(out_bboxes_cut[idx, 2] * frameWidth)
        height = int(out_bboxes_cut[idx, 3] * frameHeight)
        left = int(center_x - width / 2)
        top = int(center_y - height / 2)
        boxes.append([left, top, width, height])
    
    classIds += out_class_confs_max_ind_cut.tolist()
    confidences += out_class_confs_max_cut.tolist()

    # NMS is used inside Region layer only on DNN_BACKEND_OPENCV for another backends we need NMS in sample
    # or NMS is required if number of outputs > 1
    indices = []
    classIds = np.array(classIds)
    boxes = np.array(boxes)
    confidences = np.array(confidences)
    unique_classes = set(classIds)
    for cl in unique_classes:
        class_indices = np.where(classIds == cl)[0]
        conf = confidences[class_indices]
        box  = boxes[class_indices].tolist()
        nms_indices = cv2.dnn.NMSBoxes(box, conf, confThreshold, nmsThreshold)
        nms_indices = nms_indices[:, 0] if len(nms_indices) else []
        indices.extend(class_indices[nms_indices])
    
    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(frame, class_names[classIds[i]], confidences[i], left, top, left + width, top + height)
    

def main(): 

    args = get_args()

    print(args)

    # Init network
    t0 = time.time()
    net, outNames, class_names = init_net(args)
    print("Init network in: {:04.2f} ms".format( (time.time() - t0) * 1000 ))

    # warming up
    t0 = time.time()
    warm_up_net(net, outNames, args)
    print("Warmup network in: {:04.1f} ms".format( (time.time() - t0) * 1000))

    cap = cv2.VideoCapture(0)

    try:
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        while True:

            # Wait for a coherent pair of frames: depth and color
            total_0 = time.time()

            ret, color_image = cap.read()
            if not ret:
                continue

            # Duplicate frame so we can benchmark for batch_size > 1
            images = []
            for i in range(args.batch_size):
                images.append(color_image)
            blob = cv2.dnn.blobFromImages(images, size=(args.shape, args.shape), swapRB=True, crop=False, ddepth=cv2.CV_8U)

            # Run a model
            inference_0 = time.time()
            net.setInput(blob, scalefactor=1/255.0)
            #       Output shape
            #       Yolo use pyramid feature (1100%, 50%, 25%) so we will have 3 follow blobs
            #       batch_size x number_of_detected x detection_size
            #       detection_size = center_x, center_y, height, width, box_score, class_0_score, .... , class_n_score
            outs_ = net.forward(outNames)
            inference_1 = time.time()

            # only use the first frame for displaying
            if args.batch_size == 1:
                outs = np.concatenate(outs_, axis=0)
                postprocess(color_image, outs, args.conf, args.nms, class_names)
            else:
                outs = np.concatenate(outs_, axis=1)
                for i in range(args.batch_size):
                    postprocess(color_image, outs[i, :, :], args.conf, args.nms, class_names)

            total_1 = time.time()

            # Draw the inference time
            label = '{:04.1f} / {:04.1f} (ms)'.format(
                (inference_1 - inference_0) * 1000, 
                (total_1 - total_0) * 1000
            ) 

            cv2.putText(color_image, label, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Show images
            cv2.imshow('RealSense', color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:

        # Stop streaming
        pipeline.stop()

if __name__ == "__main__":
    main()