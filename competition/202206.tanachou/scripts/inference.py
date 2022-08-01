# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

import mmcv

from mmcls.apis import inference_model, init_model, show_result_pyplot


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Whether to show the predict results by matplotlib.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_model(model, args.img)
    # show the results
    print(mmcv.dump(result, file_format='json', indent=4))
    if args.show:
        show_result_pyplot(model, args.img, result)


class ScoringService(object):
    @classmethod
    def get_model(cls, model_path, reference_path, reference_meta_path):
        """Get model method

        Args:
            model_path (str): Path to the trained model directory.
            reference_path (str): Path to the reference data.
            reference_meta_path (str): Path to the meta data.

        Returns:
            bool: The return value. True for success, False otherwise.
        """
        #try:
        config = "config.py"
        checkpoints = "checkpoint.pth"
        device = "cuda:0"
        cls.model = init_model(config, checkpoints, device=device)

        with open(reference_meta_path) as f:
            reference_meta = json.load(f)
        embeddings, ids = make_reference(reference_path, reference_meta, cls.model)
        cls.embeddings = embeddings
        cls.ids = ids
        #    return True
        #except:
        #    return False

    @classmethod
    def predict(cls, input):
        """Predict method

        Args:
            input (str): path to the image you want to make inference from

        Returns:
            dict: Inference for the given input.
        """
        # load an image and get the file name
        image = read_image(input)
        sample_name = os.path.basename(input).split('.')[0]

        # make prediction
        with torch.no_grad():
            prediction = cls.model.predict(image.unsqueeze(0).cuda()).cpu().numpy()

        result = inference_model(cls.model, input)


        # make output
        prediction = postprocess(prediction, cls.embeddings, cls.ids)
        output = {sample_name: prediction}

        return output


if __name__ == '__main__':
    main()