import torch
import numpy as np
from mmcv.parallel import collate, scatter
from mmcls.apis import init_model, show_result_pyplot
from mmcls.datasets.pipelines import Compose

import os
import json
from skimage import io


def inference_model(model, img, topk=10):
    """Inference image(s) with the classifier.
    Args:
        model (nn.Module): The loaded classifier.
        img (str/ndarray): The image filename or loaded image.
        topk             : top k result
    Returns:
        result (list(dict)): The classification results that contains
            `class_name`, `pred_label` and `pred_score`.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    if isinstance(img, str):
        if cfg.data.test.pipeline[0]['type'] != 'LoadImageFromFile':
            cfg.data.test.pipeline.insert(0, dict(type='LoadImageFromFile'))
        data = dict(img_info=dict(filename=img), img_prefix=None)
    else:
        if cfg.data.test.pipeline[0]['type'] == 'LoadImageFromFile':
            cfg.data.test.pipeline.pop(0)
        data = dict(img=img)
    test_pipeline = Compose(cfg.data.test.pipeline)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)

    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]

    # forward the model
    with torch.no_grad():
        # inference only 1 image
        scores = model(return_loss=False, **data)[0] 
        # pred_scores = np.max(scores, axis=1)[:topk]
        # pred_labels = np.argmax(scores, axis=1)[:topk]
        # topk maximum
        pred_labels = (-scores).argsort()[:topk]
        pred_scores = scores[pred_labels]

    results = []
    for pred_score, pred_label in zip(pred_scores, pred_labels):
        result = {'pred_label': pred_label, 'pred_score': float(pred_score), 'pred_class': model.CLASSES[pred_label]}
        results.append(result)

        # print("{} : {}".format(pred_score, pred_label))
    # result['pred_class'] = model.CLASSES[result['pred_label']]
    return results

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
            config = "../model/config.py"
            checkpoints = "../model/best_accuracy.pth"
            device = "cuda:0"
            cls.model = init_model(config, checkpoints, device=device)
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
        image = io.imread(input)
        sample_name = os.path.basename(input).split('.')[0]

        # make prediction
        results = inference_model(cls.model, input, topk=10)
        prediction = []
        for result in results:
            prediction.append(result["pred_class"])

        # # make output
        output = {sample_name: prediction}

        return output

if __name__ == '__main__':
    scoringService = ScoringService()
    print(scoringService.get_model())
    print(scoringService.predict("/datadrive/workspace/others/202206.tanachou/data/train/000/0.jpg"))
