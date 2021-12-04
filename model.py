import torch
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class Model():
    def __init__(self, input_channels, num_classes):
        self.model = Model.get_fasterrcnn_model2(input_channels, num_classes)
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        # move model to the right device
        self.model.to(self.device)

    @staticmethod
    def get_fasterrcnn_model2(input_channels, num_classes):

        # Default is ((32,), (64,), (128,), (256,), (512,))
        anchor_sizes = ((2), (4), (8), (16), (32))

        # Default: ((0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0))
        #aspect_ratios = ((0.5), (1.0), (1.5), (2.0))
        aspect_ratios = ((0.5, 1.0, 1.5), (0.5, 1.0, 1.5),
                         (0.5, 1.0, 1.5), (0.5, 1.0, 1.5),
                         (0.5, 1.0, 1.5))

        anchor_generator = AnchorGenerator(
            sizes=anchor_sizes, aspect_ratios=aspect_ratios)
        print(f"sizes: {anchor_sizes}")
        print(f"ratios: {aspect_ratios}")

        if input_channels == 3:
            image_mean = [0.485, 0.456, 0.406]
            image_std = [0.229, 0.224, 0.225]
        elif input_channels == 6:
            # optical stats, mean: tensor([0.3407, 0.3538, 0.3668]), std: tensor([0.2424, 0.2450, 0.2402])
            # optical_flow stats, mean: tensor([0.1372, 0.2718, 0.0078]), std: tensor([0.2668, 0.4377, 0.0441])
            image_mean = [0.3407, 0.3538, 0.3668, 0.1372, 0.2718, 0.0078]
            image_std = [0.2424, 0.2450, 0.2402, 0.2668, 0.4377, 0.0441]
    # Defaults
    #        image_mean = [0.485, 0.456, 0.406, 0.485, 0.456, 0.406]
    #        image_std = [0.229, 0.224, 0.225, 0.229, 0.224, 0.225]

        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=False,
            image_mean=image_mean,
            image_std=image_std,
            rpn_anchor_generator=anchor_generator,
            min_size=1080,
            max_size=2048)

        # get number of input features for the classifier

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes)

        model.backbone.body.conv1 = torch.nn.Conv2d(input_channels, 64, kernel_size=7,
                                                    stride=2, padding=3, bias=False)

        return model

    def get_model(self):
        return self.model

    def resume(self, params):
        self.model.load_state_dict(params)
