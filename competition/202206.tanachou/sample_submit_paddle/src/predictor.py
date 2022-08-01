from enum import unique
import os
import sys 

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))
sys.path.append(os.path.abspath(os.path.join(__dir__, './')))

import copy
import cv2
import numpy as np
import faiss
import pickle

from src.predict_rec import RecPredictor
from src.predict_det import DetPredictor
from src.build_gallery import GalleryBuilder

from utils import logger
from utils import config
from utils.get_image_list import get_image_list
from utils.draw_bbox import draw_bbox_results

from src.predict_system import SystemPredictor


class ScoringService(object):
    @classmethod
    def get_model(cls, model_path="", reference_path="", reference_meta_path=""):
        """Get model method

        Args:
            model_path (str): Path to the trained model directory.

        Returns:
            bool: The return value. True for success, False otherwise.
        """

        try:
            args = config.parse_args()
            args.config = "./inference_product.yaml"
            load_config = config.get_config(args.config, overrides=args.override, show=True)
            cls.config = load_config
            cls.system_predictor = SystemPredictor(load_config)
            return True
        except:
            return False


    @classmethod
    def predict(cls, input):
        """Predict method

        Args:
            input (str): path to the image you want to make inference from

        Returns:
            dict: Inference for the given input.
        """
        # load an image and get the file name
        sample_name = os.path.basename(input).split('.')[0]
        img = cv2.imread(input)[:, :, ::-1]
        results = cls.system_predictor.predict_topk(img)
        # print(results)

        results_sorted = sorted(results, key=lambda d: d['rec_scores'], reverse=True) 
        # check new type
        # if results_sorted[0]["rec_scores"] < 0.85:
        #     batch_img = []
        #     batch_img.append(img)
        #     rec_feat = cls.system_predictor.rec_predictor.predict(batch_img)

        #     start_id = max(cls.system_predictor.id_map.keys()) + 1 
        #     ids_now = (np.arange(0, len(rec_feat)) + start_id).astype(np.int64)
            
        #     cls.system_predictor.Searcher.add_with_ids(rec_feat, ids_now)
        #     current_classes = set( [int(tmp.split()[1]) for tmp in cls.system_predictor.id_map.values()] )
        #     current_classes_sort = sorted(current_classes, reverse=True)
            
        #     # is this a new class
        #     if results_sorted[0]["rec_scores"] < 0.75:
        #         new_class = current_classes_sort[0] + 1
        #     else:
        #         new_class = current_classes_sort[0]

        #     cls.system_predictor.id_map[ids_now[0]] = "tmp.jpg\t{:03d}".format(new_class)

        #     # Update current file
        #     faiss.write_index(cls.system_predictor.Searcher, os.path.join(cls.config["IndexProcess"]["index_dir"], "vector.index"))

        #     with open(os.path.join(cls.config["IndexProcess"]["index_dir"], "id_map.pkl"), 'wb') as fd:
        #         pickle.dump(cls.system_predictor.id_map, fd)


        # make prediction
        prediction = []
        for result in results_sorted:
            prediction.append(result["rec_docs"])

        # # make output
        output = {sample_name: prediction[:10]}

        return output

if __name__ == '__main__':
    # args = config.parse_args()
    # args.config = "./configs/inference_product.yaml"
    # config = config.get_config(args.config, overrides=args.override, show=True)
    scoringService = ScoringService()
    print(scoringService.get_model())
    # print(scoringService.predict("/datadrive/workspace/others/202206.tanachou/data/train/000/0.jpg"))

    print(scoringService.predict("/datadrive/workspace/others/202206.tanachou/sample_submit_paddle/model/product/gallery/gannidress_05.jpg"))
