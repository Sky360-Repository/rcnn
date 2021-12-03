# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

from torchvision import transforms
from dense_optical_flow import DenseOpticalFlow
import json
from torchvision.models.detection.rpn import AnchorGenerator
import os
import random
import numpy as np
import torch
from PIL import Image, ImageDraw

import imgaug
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

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
        #print(f"obj_ids: {obj_ids}")
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]
        #print(f"masks: {masks}")

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
        #print(f"target: {target}")

        img = torchvision.transforms.ToTensor()(img)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class STFOpticalFlowDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.labels = {'motion': 1}

        # map of image filename to return item
        self.image_items_map = {}

        # Map of dataloaders assigned image number to image filenames
        self.image_files = {}

        # The current index of the self.image_files structure
        self.image_file_index = 0

        self.classes = {
            'unknown': 1
        }

        for key, value in self.classes.items():
            self.labels[value] = key

        # Map of imagename -> [{'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name}...]
        self.image_data = {}

        annotations_filename = os.path.join(self.root, 'annotations.json')
        with open(annotations_filename, 'r') as file:
            annotations_file = json.load(file)
        dof = None
        for frame in annotations_file['frames']:
            frame_number = frame['frame']

            boxes = np.zeros((0, 4))
            for ann_dict in frame['annotations']:
                bbox = ann_dict['bbox']

                ann_row = np.zeros((1, 4))
                ann_row[0, 0] = bbox[0]
                ann_row[0, 1] = bbox[1]
                ann_row[0, 2] = bbox[2]
                ann_row[0, 3] = bbox[3]

                boxes = np.append(
                    boxes, ann_row, axis=0)
            # transform from [x, y, w, h] to [x1, y1, x2, y2]
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
            area = (
                boxes[:, 3] - boxes[:, 1]) * \
                (boxes[:, 2] - boxes[:, 0])
            num_objs = len(boxes)
            target = {}
            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
            target["labels"] = torch.ones(
                (num_objs,), dtype=torch.int64)
            target["image_id"] = torch.tensor([frame_number])
            target["area"] = torch.tensor(area)
            target["iscrowd"] = torch.zeros((num_objs,), dtype=torch.int64)
            print(target)

            self._add(frame_number, target)

    def tracker_id_to_class_id(self, annotations_file, annotation):
        tracker_id = annotation['track_id']
        label = annotations_file['track_labels'][str(tracker_id)]
        return self.classes[label]

    def _add(self, frame_number, target):
        print(f"Adding: {frame_number}")
        if frame_number in self.image_items_map:
            raise Exception('dupe file')
        else:
            self.image_files[self.image_file_index] = {
                'frame_number': frame_number,
                'target': target
            }
            self.image_file_index += 1

    def __getitem__(self, idx):
        sample = self.image_files[idx]

        frame_number = sample['frame_number']

        image_dir = os.path.join(self.root, 'images')
        image_filename = os.path.join(
            image_dir, f"{frame_number:06}.original.jpg")
        img = self._load_image(image_filename)

        optical_flow_path = os.path.join(
            self.root, 'images', f"{frame_number:06}.optical_flow.jpg")

        optical_flow_img = self._load_image(optical_flow_path)

        target = sample['target']

        optical_tensor = torchvision.transforms.ToTensor()(img)
        optical_flow_tensor = torchvision.transforms.ToTensor()(optical_flow_img)
        merged_tensor = torch.cat((optical_tensor, optical_flow_tensor), 0)
        # print(merged_tensor)
        if self.transforms is not None:
            merged_tensor, target = self.transforms(merged_tensor, target)

        return merged_tensor, target

    def image_aspect_ratio(self, image_index):
        image_filename = self.image_files[image_index]
        print(f"image_aspect_ratio loading {image_filename}")
        image = self._load_image(image_filename)
        height, width, depth = image.shape
        print(f"image_aspect_ratio h,w,d: {height, width, depth}")
        return float(width) / float(height)

    def _load_image(self, image_filename):
        return Image.open(image_filename)

    def __len__(self):
        return len(self.image_files)

    def num_classes(self):
        return len(self.classes)


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
        self.filter_no_identifications()
        self.cache = {}
        self.stats = {}
        self.init_stats('optical')
        self.init_stats('optical_flow')

    def filter_no_identifications(self):
        print("Filtering")
        new_imgs = []
        new_of_imgs = []
        new_xmls = []
        for img, of_img, xml in zip(self.imgs, self.optical_flow_imgs, self.xmls):
            filename = os.path.join(self.root, "annotations", xml)
            tree = ET.parse(filename)
            tree_root = tree.getroot()
            count = 0
            for obj in tree_root.findall("object"):
                # bndbox = PesmodOpticalFlowDataset._get_and_check(
                #    obj, "bndbox", 1)
                count += 1
            # print(f"{xml}:{count}")
            if count > 0:
                print(f"Adding {img} with {count} boxes at {len(new_imgs)}")
                new_imgs.append(img)
                new_of_imgs.append(of_img)
                new_xmls.append(xml)
        self.imgs = new_imgs
        self.optical_flow_imgs = new_of_imgs
        self.xmls = new_xmls

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
            tree_root = tree.getroot()
            for member in tree_root.findall("object"):
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
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        optical_flow_path = os.path.join(
            self.root, "optical_flow", self.optical_flow_imgs[idx])

        # Images are H_W_C
        optical_img = Image.open(img_path)
        optical_flow_img = Image.open(optical_flow_path).convert('HSV')

        boxes = []
        xml_file = os.path.join(self.root, "annotations", self.xmls[idx])
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

        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = torch.tensor([idx])

        try:
            boxes_tensor = torch.tensor(boxes)
            if num_objs == 0:
                raise Exception(f"Found image with no objects: {idx}")
            elif num_objs == 1:
                area = torch.tensor([(boxes_tensor[0][3] - boxes_tensor[0][1]) *
                                     (boxes_tensor[0][2] - boxes_tensor[0][0])])
            else:
                area = (boxes_tensor[:, 3] - boxes_tensor[:, 1]) * \
                    (boxes_tensor[:, 2] - boxes_tensor[:, 0])
        except Exception as e:
            print(e)
            print(f"Boxes failed  {self.imgs[idx]}:{boxes}")
            import sys
            sys.stdout.flush()
            sys.exit(1)

        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        w, h = optical_img.size
        bbs = imgaug.BoundingBoxesOnImage.from_xyxy_array(
            boxes, (h, w, 3))

        imgs = [np.array(optical_img), np.array(optical_flow_img)]

        if self.transforms is not None:
            seq_det = self.transforms.to_deterministic()
            imgs = seq_det.augment_images(imgs)
            bbs = seq_det.augment_bounding_boxes(bbs)
            bbs = bbs.remove_out_of_image().clip_out_of_image()
            # print(
            #    f"merged after transform:[ {imgs[0].shape}, {imgs[1].shape} ]")
            # print(
            #    f"optical max:{imgs[0].max()},mean:{imgs[0].mean()}, OF Max:{imgs[1].max()},mean:{imgs[1].mean()}")

        target_boxes = imgaug.BoundingBoxesOnImage.to_xyxy_array(bbs)
        target['boxes'] = torch.as_tensor(target_boxes, dtype=torch.float32)

        # #Tensors are C_H_W and are scaled (div 255), but not normalized
        optical_tensor = torchvision.transforms.ToTensor()(imgs[0])
        optical_flow_tensor = torchvision.transforms.ToTensor()(imgs[1])
        # print(
        #    f"optical max:{optical_tensor.max()},mean:{optical_tensor.mean()}, OF Max:{optical_flow_tensor.max()},mean:{optical_flow_tensor.mean()}")

        self.update_stats('optical', optical_tensor)
        self.update_stats('optical_flow', optical_flow_tensor)
        # self.print_stats('optical')
        # self.print_stats('optical_flow')

        merged_tensor = torch.cat((optical_tensor, optical_flow_tensor), 0)
        #txform = merged_tensor[:, None, None] / merged_tensor[:, None, None]
        ret = (merged_tensor, target)
        #self.cache[idx] = ret
        return ret

    def init_stats(self, img_name):
        self.stats[img_name] = {
            'psum': torch.tensor([0.0, 0.0, 0.0]),
            'psum_sq': torch.tensor([0.0, 0.0, 0.0]),
            'pixel_count': 0
        }

    def update_stats(self, img_name, tensor):
        _, height, width = tensor.shape
        #print(f"update stats tensor size: {tensor.shape}")
        self.stats[img_name]['psum'] += tensor.sum(axis=[1, 2])
        self.stats[img_name]['psum_sq'] += (tensor**2).sum(axis=[1, 2])
        self.stats[img_name]['pixel_count'] += width*height

    def print_stats(self, img_name):
        count = self.stats[img_name]['pixel_count']
        total_mean = self.stats[img_name]['psum'] / count
        total_var = (self.stats[img_name]['psum_sq'] /
                     count) - (total_mean ** 2)
        total_std = torch.sqrt(total_var)
        print(
            f"{img_name} stats, mean:{total_mean}, std:{total_std}")

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
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.backbone.body.conv1 = torch.nn.Conv2d(input_channels, 64, kernel_size=7,
                                                stride=2, padding=3, bias=False)

    return model


def get_model(input_channels, classes):
    model = get_model_instance_segmentation(classes)
    model.backbone.body.conv1 = torch.nn.Conv2d(input_channels, 64, kernel_size=7,
                                                stride=2, padding=3, bias=False)
    return model


def get_long_transform():
    def sometimes(aug): return imgaug.augmenters.Sometimes(0.75, aug)
    return imgaug.augmenters.Sequential([
        sometimes(imgaug.augmenters.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            rotate=random.randint(0, 359)),
        ),
        imgaug.augmenters.SomeOf(
            (0, 5),
            [
                imgaug.augmenters.OneOf([
                    imgaug.augmenters.GaussianBlur((0, 3.0)),
                    imgaug.augmenters.AverageBlur(k=(2, 7)),
                    imgaug.augmenters.MedianBlur(k=(3, 11)),
                ]),
                imgaug.augmenters.OneOf([
                    imgaug.augmenters.Dropout(
                        (0.01, 0.1), per_channel=0.5),
                    imgaug.augmenters.CoarseDropout(
                        (0.03, 0.15), size_percent=(0.02, 0.05),
                        per_channel=0.2),
                ]
                )], random_order=True)
    ], random_order=True)


def get_transform(train):
    if train:
        return imgaug.augmenters.Sequential([
            imgaug.augmenters.GaussianBlur(sigma=(0, 3.0)),
            imgaug.augmenters.Affine(rotate=random.randint(0, 359))])
    else:
        return imgaug.augmenters.Sequential([])


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

    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    return parser


def render_boxes(bboxes, image):
    pil_image = torchvision.transforms.ToPILImage()(image.squeeze_(0))
    draw = ImageDraw.Draw(pil_image)
    for bbox in bboxes:
        draw.rectangle(bbox, fill="red")
    return torchvision.transforms.ToTensor()(pil_image)


def write_to_tb(data_loader, model):
    writer = SummaryWriter()
    dataiter = iter(data_loader)
    images, labels = dataiter.next()

    count = 0
    for image, target in zip(images, labels):

        optical = image[:3]
        optical_flow = image[3:]

        writer.add_image_with_boxes(
            f"optical-{count}", optical, target['boxes'], global_step=count)
        writer.add_image_with_boxes(
            f"optical-flow-{count}", optical_flow, target['boxes'], global_step=count)
        count += 1
    # Errors
    #writer.add_graph(model, input_to_model=images)


def main():
    args = get_args_parser().parse_args()

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations

    dataset_name = 'Combined'
    if dataset_name == 'stf':
        dataset_test = STFOpticalFlowDataset(
            '../simpletracker/output/2021_11_26-06_18_02_PM/video_000000/',
            # '../simpletracker/output/2021_11_26-07_18_24_PM/birds_and_plane_000000/',
            # '../simpletracker/output/2021_11_26-08_16_29_PM/Test_Trimmed_000000/',
            get_transform(train=False))
        indices = torch.randperm(len(dataset_test)).tolist()
        dataset_test = torch.utils.data.Subset(dataset_test, indices[:10])
        num_channels = 6
        batch_size = 6

    elif dataset_name == 'PennFudanDataset':
        dataset = PennFudanDataset(
            'PennFudanPed', get_transform(train=True))
        dataset_test = PennFudanDataset(
            'PennFudanPed', get_transform(train=False))

        # split the dataset in train and test set
        indices = torch.randperm(len(dataset)).tolist()
        dataset = torch.utils.data.Subset(dataset, indices[:-50])
        dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])
        num_channels = 3
        batch_size = 6

    elif dataset_name == 'Combined':
        dataset = PesmodOpticalFlowDataset(
            os.path.join('INPUT', 'train'), get_transform(train=True))
        dataset_test = PesmodOpticalFlowDataset(
            os.path.join('INPUT', 'train'), get_transform(train=False))
        indices = torch.randperm(len(dataset_test)).tolist()

        dataset_test = torch.utils.data.Subset(dataset_test, indices[:100])
        print(indices[:10])
        num_channels = 6
        batch_size = 6

    if not args.test_only:
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=8,
            collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=8,
        collate_fn=utils.collate_fn)

    model = get_fasterrcnn_model2(num_channels, 2)

    #write_to_tb(data_loader, model)

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

    if args.test_only:
        print("Testing only")
        evaluate(model, data_loader_test, device=device, epoch=0)
        return

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
        evaluate(model, data_loader_test, device=device, epoch=epoch)

    print("That's it!")


if __name__ == "__main__":
    main()
