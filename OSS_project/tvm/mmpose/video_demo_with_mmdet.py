import os
from argparse import ArgumentParser

import cv2
from mmdet.apis import inference_detector, init_detector

from mmpose.apis import inference_pose_model, init_pose_model, vis_pose_result
import time
import numpy as np

def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--video-path', type=str, help='Video path')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')
    parser.add_argument(
        '--out-video-root',
        default='',
        help='Root of the output video file. '
        'Default not saving the visualization video.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')

    args = parser.parse_args()

    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    assert args.show or (args.out_video_root != '')
    assert 'cuda' in args.device
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device)
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device)

    cap = cv2.VideoCapture(args.video_path)

    if args.out_video_root != '':
        save_out_video = True
    else:
        save_out_video = False

    if save_out_video:
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(
            os.path.join(args.out_video_root,
                         f'vis_{os.path.basename(args.video_path)}'), fourcc,
            fps, size)

    detection_times = []
    pose_times = []
    vis_times = []
    while (cap.isOpened()):
        flag, img = cap.read()
        if not flag:
            break
        # test a single image, the resulting box is (x1, y1, x2, y2)
        t1 = time.time()
        det_results = inference_detector(det_model, img)
        # keep the person class bounding boxes.
        person_bboxes = det_results[0].copy()
        t2 = time.time()
        detection_times.append(t2 - t1)

        # test a single image, with a list of bboxes.
        pose_results = inference_pose_model(
            pose_model,
            img,
            person_bboxes,
            bbox_thr=args.bbox_thr,
            format='xyxy')
        t3 = time.time()
        pose_times.append(t3 - t2)

        # show the results
        vis_img = vis_pose_result(
            pose_model,
            img,
            pose_results,
            skeleton=skeleton,
            kpt_score_thr=args.kpt_thr,
            show=False)
        t4 = time.time()
        vis_times.append(t4 - t3)

        if args.show:
            cv2.imshow('Image', vis_img)

        if save_out_video:
            videoWriter.write(vis_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    detection_time = np.mean(np.asarray(detection_times)) * 1000
    pose_time = np.mean(np.asarray(pose_times)) * 1000
    vis_time = np.mean(np.asarray(vis_times)) * 1000
    print("Detection: {:4.2f} ms".format(detection_time ))
    print("Pose: {:4.2f} ms".format(pose_time ))
    print("Pose vis: {:4.2f} ms".format(vis_time))


    cap.release()
    if save_out_video:
        videoWriter.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
