# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

from torchvision.models.detection.rpn import AnchorGenerator
import os
import numpy as np
import torch
from PIL import Image, ImageDraw

import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from engine import train_one_epoch, evaluate
import utils
import transforms as T

import xml.etree.ElementTree as ET  # elementpath

import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter


class PennFudanDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images and masks
        print(f"Loading item {idx}")
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        print(f"obj_ids: {obj_ids}")
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]
        print(f"masks: {masks}")

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        print(f"target: {target}")

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class PesmodOpticalFlowDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.optical_flow_imgs = list(
            sorted(os.listdir(os.path.join(root, "optical_flow"))))
        self.xmls = list(sorted(os.listdir(os.path.join(root, "annotations"))))
        self.categories = PesmodOpticalFlowDataset._get_categories(
            self.root, self.xmls)
        self.cache = {}

    @staticmethod
    def _get_categories(path, xml_files):
        """Generate category name to id mapping from a list of xml files.

        Arguments:
            xml_files {list} -- A list of xml file paths.

        Returns:
            dict -- category name to id mapping.
        """
        classes_names = []
        for xml_file in xml_files:
            filename = os.path.join(path, "annotations", xml_file)
            tree = ET.parse(filename)
            root = tree.getroot()
            for member in root.findall("object"):
                classes_names.append(member[0].text)
        classes_names = list(set(classes_names))
        classes_names.sort()
        return {name: i for i, name in enumerate(classes_names)}

    @staticmethod
    def _get_and_check(root, name, length):
        vars = root.findall(name)
        if len(vars) == 0:
            raise ValueError("Can not find %s in %s." % (name, root.tag))
        if length > 0 and len(vars) != length:
            raise ValueError(
                "The size of %s is supposed to be %d, but is %d."
                % (name, length, len(vars))
            )
        if length == 1:
            vars = vars[0]
        return vars

    def __getitem__(self, idx):
        # load images and masks
        #print(f"Loading item {idx}")
        #cached_item = self.cache.get(idx, None)
        # if cached_item:
        #    return cached_item

        img_path = os.path.join(self.root, "images", self.imgs[idx])
        optical_flow_path = os.path.join(
            self.root, "optical_flow", self.optical_flow_imgs[idx])
        optical_img = Image.open(img_path)
        optical_flow_img = Image.open(optical_flow_path).convert('HSV')

        boxes = []
        xml_file = os.path.join(self.root, "annotations", f"{idx:06}.xml")
        tree = ET.parse(xml_file)
        root = tree.getroot()
        boxes = []
        for obj in root.findall("object"):
            category = PesmodOpticalFlowDataset._get_and_check(
                obj, "name", 1).text
            if category not in self.categories:
                new_id = len(self.categories)
                self.categories[category] = new_id
            category_id = self.categories[category]
            bndbox = PesmodOpticalFlowDataset._get_and_check(obj, "bndbox", 1)
            xmin = int(float(PesmodOpticalFlowDataset._get_and_check(
                bndbox, "xmin", 1).text)) - 1
            ymin = int(float(PesmodOpticalFlowDataset._get_and_check(
                bndbox, "ymin", 1).text)) - 1
            xmax = int(float(PesmodOpticalFlowDataset._get_and_check(
                bndbox, "xmax", 1).text))
            ymax = int(float(PesmodOpticalFlowDataset._get_and_check(
                bndbox, "ymax", 1).text))
            assert xmax > xmin
            assert ymax > ymin
            boxes.append([xmin, ymin, xmax, ymax])
        num_objs = len(boxes)
        #print(f"loaded boxes {idx}:{num_objs}")
# 'boxes': tensor([[ 46.,  84., 146., 291.],
#                  [162.,  86., 239., 347.],
#                  [224.,  85., 299., 306.],
#                  [315.,  80., 390., 368.],
#                  [374.,  99., 470., 350.],
#                  [469.,  93., 558., 394.],
#                  [532.,  89., 604., 375.]])

        #print(f"Boxes bf conversion {idx}: {boxes}")

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = torch.tensor([idx])

        try:

            if num_objs == 0:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                area = torch.tensor([0])
            elif num_objs == 1:
                area = torch.tensor([(boxes[0][3] - boxes[0][1]) *
                                     (boxes[0][2] - boxes[0][0])])
            else:
                area = (boxes[:, 3] - boxes[:, 1]) * \
                    (boxes[:, 2] - boxes[:, 0])
        except Exception as e:
            print(e)
            print(f"Boxes failed  {idx}:{boxes}")
            import sys
            sys.stdout.flush()
            sys.exit(1)

        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        #print(f"target: {target}")

        optical_tensor = torchvision.transforms.ToTensor()(optical_img)
        optical_flow_tensor = torchvision.transforms.ToTensor()(optical_flow_img)
        merged_tensor = torch.cat((optical_tensor, optical_flow_tensor), 0)

        if self.transforms is not None:
            merged_tensor, target = self.transforms(merged_tensor, target)
            

# image: tensor([[[0.0627, 0.0314, 0.1098,  ..., 0.8157, 0.8314, 0.8471],
#         [0.0510, 0.0353, 0.0941,  ..., 0.7922, 0.7765, 0.8235],
#         [0.0627, 0.0627, 0.0980,  ..., 0.6353, 0.4510, 0.3608],
        #print(f"Merged {merged_tensor}")

        #txform = merged_tensor[:, None, None] / merged_tensor[:, None, None]
        ret = (merged_tensor, target)
        #self.cache[idx] = ret
        return ret

    def __len__(self):
        return len(self.imgs)


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        pretrained=False)

    # get number of input features for the classifier

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def get_fasterrcnn_model2(input_channels, num_classes):

    # fasterrcnn_mobilenet_v3_large_fpn

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
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=False,
        image_mean=[0.485, 0.456, 0.406, 0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225, 0.229, 0.224, 0.225],
        rpn_anchor_generator=anchor_generator)

    # get number of input features for the classifier

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.backbone.body.conv1 = torch.nn.Conv2d(input_channels, 64, kernel_size=7,
                                                stride=2, padding=3, bias=False)

    return model


def get_model(input_channels, classes):
    model = get_model_instance_segmentation(classes)
    model.backbone.body.conv1 = torch.nn.Conv2d(input_channels, 64, kernel_size=7,
                                                stride=2, padding=3, bias=False)
    return model


def get_transform(train):
    transforms = []
    #transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(
        description="PyTorch Detection Training", add_help=add_help)

    parser.add_argument("--epochs", default=26, type=int,
                        metavar="N", help="number of total epochs to run")
    parser.add_argument("--start_epoch", default=0,
                        type=int, help="start epoch")
    parser.add_argument("--resume", default="", type=str,
                        help="path of checkpoint")
    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true",
                        help="Use torch.cuda.amp for mixed precision training")

    return parser


def render_boxes(bboxes, image):
    pil_image = torchvision.transforms.ToPILImage()(image.squeeze_(0))
    draw = ImageDraw.Draw(pil_image)
    for bbox in bboxes:
        draw.rectangle(bbox, fill="red")
    return torchvision.transforms.ToTensor()(pil_image)


def write_to_tb(data_loader):
    writer = SummaryWriter()
    dataiter = iter(data_loader)
    images, labels = dataiter.next()
    opticals = []
    optical_flows = []
    for image, label in zip(images, labels):
        # print()
        bboxes = label['boxes'].numpy()
        optical = image[:3]
        #print(optical.shape)
        optical_flow = image[3:]
        #print(optical_flow.shape)
        annotated_optical = render_boxes(bboxes, optical)
        annotated_optical_flow = render_boxes(bboxes, optical_flow)
        #print(annotated_optical.shape)

        opticals.append(annotated_optical)
        optical_flows.append(annotated_optical_flow)
    # create grid of images
    optical_img_grid = torchvision.utils.make_grid(opticals)
    optical_flow_img_grid = torchvision.utils.make_grid(optical_flows)

    # show images
    # matplotlib_imshow(img_grid)

    # write to tensorboard
    writer.add_image('optical', optical_img_grid)
    writer.add_image('optical-flow', optical_flow_img_grid)


def main():
    args = get_args_parser().parse_args()

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations

    if False:
        dataset = PennFudanDataset(
            'PennFudanPed', get_transform(train=True))
        dataset_test = PennFudanDataset(
            'PennFudanPed', get_transform(train=False))

        # split the dataset in train and test set
        indices = torch.randperm(len(dataset)).tolist()
        dataset = torch.utils.data.Subset(dataset, indices[:-50])
        dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])
    else:
        dataset = PesmodOpticalFlowDataset(
            os.path.join('PESMOD', 'train'), get_transform(train=True))
        dataset_test = PesmodOpticalFlowDataset(
            os.path.join('PESMOD', 'test'), get_transform(train=False))
        indices = torch.randperm(len(dataset_test)).tolist()
        dataset_test = torch.utils.data.Subset(dataset_test, indices[:200])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=12, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    write_to_tb(data_loader)

    # get the model using our helper function
    #model = get_model_instance_segmentation(num_classes)
    model = get_fasterrcnn_model2(6, 2)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    scaler = None
    if args.amp:
        scaler = torch.cuda.amp.GradScaler()
        print("Using amp")

    output_dir = 'save'

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader,
                        device, epoch, 10, scaler)
        # update the learning rate
        lr_scheduler.step()

        if output_dir:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "args": {},
                "epoch": epoch,
            }
            if args.amp:
                checkpoint["scaler"] = scaler.state_dict()
            utils.save_on_master(checkpoint, os.path.join(
                output_dir, f"model_{epoch}.pth"))
            utils.save_on_master(checkpoint, os.path.join(
                output_dir, "checkpoint.pth"))

        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    print("That's it!")


if __name__ == "__main__":
    main()
