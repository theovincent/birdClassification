def get_model(model_name, use_cuda):
    model_ft = None

    if model_name == "detectron2":
        """detectron2"""
        from detectron2.config import get_cfg
        from detectron2 import model_zoo
        from detectron2.engine import DefaultPredictor

        cfg = get_cfg()
        # Load model
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        if use_cuda:
            cfg.MODEL.DEVICE = "gpu"
        else:
            cfg.MODEL.DEVICE = "cpu"

        model_ft = DefaultPredictor(cfg)
        input_height, input_width = (480, 640)
    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, (input_height, input_width)
