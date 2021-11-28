import math
import sys
import time

import torch
import torchvision
import torchvision.models.detection.mask_rcnn
import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np

tb = SummaryWriter()


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(
        window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        tb.add_scalar("Loss", losses_reduced, epoch)
        for k, v in loss_dict_reduced.items():
            tb.add_scalar(k, v, epoch)

        for name, weight in model.named_parameters():
            if weight.grad is not None:
                tb.add_histogram(name, weight, epoch)
                tb.add_histogram(f'{name}.grad', weight.grad, epoch)

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


def _draw_single_box(image, xmin, ymin, xmax, ymax, color='black', thickness=2):
    from PIL import ImageDraw, ImageFont
    font = ImageFont.load_default()
    draw = ImageDraw.Draw(image)
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line([(left, top), (left, bottom), (right, bottom),
               (right, top), (left, top)], width=thickness, fill=color)
    return image


def draw_boxes(disp_image, boxes, color):
    # xyxy format
    num_boxes = boxes.shape[0]
    list_gt = range(num_boxes)
    for i in list_gt:
        disp_image = _draw_single_box(disp_image,
                                      boxes[i, 0],
                                      boxes[i, 1],
                                      boxes[i, 2],
                                      boxes[i, 3],
                                      color=color)
    return disp_image


def add_boxes_to_tensor_image(tensor, boxes, color):
    from PIL import Image
    height, width, channel = tensor.shape
    image = torchvision.transforms.ToPILImage()(tensor.squeeze_(0))
    image = draw_boxes(image, boxes, color)
    return image



def _convert_bbox(boxes):
    ret = boxes.clone()
    ret[:, 2] = boxes[:, 0] + boxes[:, 2]
    ret[:, 3] = boxes[:, 1] + boxes[:, 3]
    return ret


@torch.inference_mode()
def evaluate(model, data_loader, device, epoch):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    # print(iou_types)
    coco_evaluator = CocoEvaluator(coco, iou_types)
    #coco_eval.params.areaRng = [
    #    [0 ** 2, 1e5 ** 2], [0 ** 2, 20 ** 2], [20 ** 2, 30 ** 2], [30 ** 2, 1e5 ** 2]]

    # default: np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
    coco_evaluator.coco_eval['bbox'].params.iouThrs = np.linspace(
        .05,
        0.95, int(
            np.round((0.95 - .05) / 0.1)) + 1, endpoint=True)
#    coco_evaluator.recThrs = [0,.01,1]
    cnt = 0
    for images, targets in metric_logger.log_every(data_loader, 100, header):
        opticals = images
        print(images[0].shape)
        images = list(img.to(device) for img in images)
        #print(f"image shape:{images[0].shape}")
        # print(f"target:{targets[0]}")

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        print(f"outputs len:{len(outputs)}, images len:{len(opticals)}")

        for out, img, target in zip(outputs, opticals, targets):
            print(f"dealing with boxes:#{len(out['boxes'])}, img:{img.shape}")
            optical_image = img[:3]
            optical_flow_image = img[3:]
            boxes = out['boxes']
            image_tensor = torchvision.transforms.ToPILImage()(optical_image.squeeze_(0))
            of_image_tensor = torchvision.transforms.ToPILImage()(
                optical_flow_image.squeeze_(0))
            outputboxes = boxes
            targetboxes = _convert_bbox(target['boxes'])
            print(f"outputboxes:{boxes}")
            print(f"targetboxes:{target['boxes']}")
            image_tensor = draw_boxes(image_tensor, outputboxes, 'Red')
            image_tensor = draw_boxes(image_tensor, targetboxes, 'Green')
            of_image_tensor = draw_boxes(
                of_image_tensor, outputboxes, 'Red')
            of_image_tensor = draw_boxes(
                of_image_tensor, targetboxes, 'Green')
            optical_image = torchvision.transforms.ToTensor()(image_tensor)
            optical_flow_image = torchvision.transforms.ToTensor()(of_image_tensor)
            grid = torchvision.utils.make_grid(
                [optical_image, optical_flow_image])
            tb.add_image(f"output-{target['image_id'].numpy()[0]}",
                         grid, global_step=epoch)
            cnt += 1

        outputs = [{k: v.to(cpu_device) for k, v in t.items()}
                   for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target,
               output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time,
                             evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator
