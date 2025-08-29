#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#

# This file is originally from: [yolov7 repo](https://github.com/WongKinYiu/yolov7/blob/main/test.py)

import argparse
import json
import os
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import yaml
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import (
    coco80_to_coco91_class,
    check_dataset,
    check_file,
    check_img_size,
    box_iou,
    non_max_suppression,
    scale_coords,
    xyxy2xywh,
    xywh2xyxy,
    set_logging,
    increment_path,
    colorstr,
)
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized
import threading
import random
import time


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def run_weights_sharing_model(model, dataloader, augment, tid, opt):
    start_time = time.time()
    time_consume = 0
    timeBuff = []
    with torch.no_grad():
        for epoch in range(opt.inf_epoch_number):
            for batch_i, (img, _, _, shapes) in enumerate(dataloader):
                if epoch * len(dataloader) + batch_i >= opt.number_iter:
                    break
                img = img.float().contiguous(
                    memory_format=torch.channels_last
                )  # uint8 to fp12
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if opt.bf16:
                    img = img.to(torch.bfloat16)
                if opt.fp16:
                    img = img.to(torch.half)

                start_time = time.time()

                if opt.bf16:
                    with torch.autocast("cpu", dtype=torch.bfloat16):
                        out = model(img, augment=augment)[0]
                elif opt.fp16:
                    with torch.autocast("cpu", dtype=torch.half):
                        out = model(img, augment=augment)[0]
                else:
                    out = model(img, augment=augment)[0]
                end_time = time.time()

                out = non_max_suppression(
                    out.float(),
                    conf_thres=opt.conf_thres,
                    iou_thres=opt.iou_thres,
                    multi_label=True,
                )

                for si, pred in enumerate(out):
                    predn = pred.clone()
                    scale_coords(
                        img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1]
                    )  # native-space pred

                if epoch * len(dataloader) + batch_i >= opt.warmup_iterations:
                    time_consume += end_time - start_time
                    timeBuff.append(end_time - start_time)

        fps = (opt.number_iter - opt.warmup_iterations) * opt.batch_size / time_consume
        avg_time = time_consume * 1000 / (opt.number_iter - opt.warmup_iterations)
        timeBuff = np.asarray(timeBuff)
        p99 = np.percentile(timeBuff, 99)
        print("P99 Latency {:.2f} ms".format(p99 * 1000))
        print(
            "Instance num: %d Avg Time/Iteration: %f msec Throughput: %f fps"
            % (tid, avg_time, fps)
        )


def test(
    opt,
    data,
    weights=None,
    batch_size=32,
    imgsz=640,
    conf_thres=0.001,
    iou_thres=0.6,  # for NMS
    save_json=False,
    single_cls=False,
    augment=False,
    verbose=False,
    inductor=False,
    seed=None,
    int8=False,
    bf16=False,
    fp16=False,
    calibration=False,
    configure_dir="configure.json",
    performance=True,
    warmup_iterations=20,
    weight_sharing=False,
    number_instance=1,
    evaluate=True,
    model=None,
    dataloader=None,
    save_dir=Path(""),  # for saving images
    save_txt=False,  # for auto-labelling
    save_hybrid=False,  # for hybrid auto-labelling
    save_conf=False,  # save auto-label confidences
    plots=True,
    wandb_logger=None,
    compute_loss=None,
    half_precision=True,
    trace=False,
    is_coco=False,
    v5_metric=False,
):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        set_logging()
        device = select_device(opt.device, batch_size=batch_size)

        # Directories
        save_dir = Path(
            increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)
        )  # increment run
        (save_dir / "labels" if save_txt else save_dir).mkdir(
            parents=True, exist_ok=True
        )  # make dir

        # Load model
        weights = [os.path.join(opt.checkpoint_dir, w) for w in weights]
        if evaluate:
            with torch.no_grad():
                model = attempt_load(weights, map_location=device)  # load FP32 model
        else:
            model = attempt_load(weights, map_location=device)  # load FP32 model
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check img_size

    if weight_sharing:
        assert (
            performance and batch_size
        ), "please set performance and set batch_size to 1 if you want run weight sharing case for latency case"
    if calibration:
        assert int8, "please enable int8 path if you want to do int8 calibration path"

    # Half
    half = (
        device.type != "cpu" and half_precision
    )  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    if isinstance(data, str):
        is_coco = data.endswith("coco.yaml")
        with open(data) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
    val = data.get("val")
    data["val"] = [
        os.path.join(opt.dataset_dir, x)
        for x in (val if isinstance(val, list) else [val])
    ]
    check_dataset(data)  # check
    nc = 1 if single_cls else int(data["nc"])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Logging
    log_imgs = 0
    if wandb_logger and wandb_logger.wandb:
        log_imgs = min(wandb_logger.log_imgs, 100)
    # Dataloader

    if device.type != "cpu":
        model(
            torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters()))
        )  # run once
    task = (
        opt.task if opt.task in ("train", "val", "test") else "val"
    )  # path to train/val/test images

    # Set rect=False for performance test and inductor int8 as
    # 1) static shapes has better performance
    # 2) inductor int8 doesn't support complex dynamic shapes now
    # TODO set rect=True for accuracy test of inductor int8
    rect = False if (int8 or performance) else True
    dataloader = create_dataloader(
        data[task],
        imgsz,
        batch_size,
        gs,
        opt,
        pad=0.5,
        rect=rect,
        workers=0,
        prefix=colorstr(f"{task}: "),
    )[0]

    if opt.prepare_dataloader:
        print("Prepare for dataloader")
        return

    if v5_metric:
        print("Testing with YOLOv5 AP metric...")

    names = dict(
        enumerate(model.names if hasattr(model, "names") else model.module.names)
    )
    model.traced = False
    model = model.to(memory_format=torch.channels_last)

    if evaluate:
        print("using offical pytorch model to do inference\n")
        x = torch.rand(batch_size, 3, imgsz, imgsz).contiguous(
            memory_format=torch.channels_last
        )

        model.eval()
        from torch._inductor import config as inductor_config

        inductor_config.cpp_wrapper = True
        if not performance:
            from torch._dynamo import config as dynamo_config

            dynamo_config.use_recursive_dict_tags_for_guards = False

        if int8:
            from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
            import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
            from torch.ao.quantization.quantizer.x86_inductor_quantizer import (
                X86InductorQuantizer,
            )
            from torch.export import export_for_training

            use_dynamic_batch = (
                not performance and (len(dataloader.dataset) % batch_size) != 0
            )
            assert not use_dynamic_batch, (
                f"Doesn't support dynamic shapes for inductor int8 now. "
                f"Please make sure dataset length {len(dataloader.dataset)} % batch_size == 0."
            )

            print("[Info] Running torch.compile() INT8 quantization")
            with torch.no_grad():
                example_inputs = (x, augment)
                exported_model = export_for_training(
                    model,
                    example_inputs,
                    strict=True,
                ).module()
                quantizer = X86InductorQuantizer()
                quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())
                prepared_model = prepare_pt2e(exported_model, quantizer)

                if dataloader:
                    for i, (img, targets, paths, shapes) in enumerate(dataloader):
                        img = img.float().contiguous(
                            memory_format=torch.channels_last
                        )  # uint8 to fp32
                        img /= 255.0  # 0 - 255 to 0.0 - 1.0
                        prepared_model(img, augment)
                        if i == 4:
                            break
                else:
                    for _ in range(4):
                        x = torch.rand(batch_size, 3, imgsz, imgsz)
                        x = x.contiguous(memory_format=torch.channels_last)
                        prepared_model(x, augment)
                converted_model = convert_pt2e(prepared_model)
                torch.ao.quantization.move_exported_model_to_eval(converted_model)
                print("[Info] Running torch.compile() with default backend")
                model = torch.compile(converted_model, dynamic=rect)
        elif bf16:
            with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
                x = x.to(torch.bfloat16)
                print("[Info] Running torch.compile() BFloat16 with default backend")
                model = torch.compile(model, dynamic=rect)
        elif fp16:
            with torch.no_grad(), torch.autocast("cpu", dtype=torch.half):
                x = x.to(torch.half)
                print("[Info] Running torch.compile() Float16 with default backend")
                model = torch.compile(model, dynamic=rect)
        else:
            with torch.no_grad():
                print("[Info] Running torch.compile() Float32 with default backend")
                model = torch.compile(model, dynamic=rect)

        with torch.no_grad(), torch.autocast(
            "cpu", enabled=bf16 or fp16, dtype=torch.half if fp16 else torch.bfloat16
        ):
            _ = model(x, augment)[0]
            _ = model(x, augment)[0]

    if performance:
        opt.number_iter = opt.steps if opt.steps > 0 else 200

        if opt.warmup_iterations >= opt.number_iter:
            print(
                f"WARNING: warmup_iter {opt.warmup_iterations} is larger than number_iter {opt.number_iter}.\n"
                "Invalid setup. Please reset your warmup_iter and number_iter!"
            )
        opt.inf_epoch_number = (
            (opt.number_iter // len(dataloader) + 1)
            if (opt.number_iter > len(dataloader))
            else 1
        )
        print("***** Running Evaluation *****")
        print(f"Number_iter: {opt.number_iter}")
        print(f"Warmup_iter: {opt.warmup_iterations}")
        print(f"Evaluation Epoch: {opt.inf_epoch_number}")

        batch_time = AverageMeter("Time", ":6.3f")

        with torch.no_grad():
            if weight_sharing:
                threads = []
                for i in range(1, number_instance + 1):
                    thread = threading.Thread(
                        target=run_weights_sharing_model,
                        args=(model, dataloader, augment, i, opt),
                    )
                    threads.append(thread)
                    thread.start()
                for thread in threads:
                    thread.join()
                exit()
            else:
                for epoch in range(opt.inf_epoch_number):
                    for i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader)):
                        if epoch * len(dataloader) + i >= opt.number_iter:
                            break
                        img = img.float().contiguous(
                            memory_format=torch.channels_last
                        )  # uint8 to fp12
                        img /= 255.0  # 0 - 255 to 0.0 - 1.0
                        if bf16:
                            img = img.to(torch.bfloat16)
                        if fp16:
                            img = img.to(torch.half)
                        targets = targets.to(device)
                        nb, _, height, width = (
                            img.shape
                        )  # batch size, channels, height, width

                        if epoch * len(dataloader) + i >= warmup_iterations:
                            end = time.time()

                        if bf16:
                            with torch.autocast("cpu", dtype=torch.bfloat16):
                                out = model(img, augment=augment)[0]
                        elif fp16:
                            with torch.autocast("cpu", dtype=torch.half):
                                out = model(img, augment=augment)[0]
                        else:
                            out = model(img, augment=augment)[0]

                        if epoch * len(dataloader) + i >= warmup_iterations:
                            batch_time.update(time.time() - end)

                        targets[:, 2:] *= torch.Tensor(
                            [width, height, width, height]
                        ).to(
                            device
                        )  # to pixels
                        lb = (
                            [targets[targets[:, 0] == i, 1:] for i in range(nb)]
                            if save_hybrid
                            else []
                        )  # for autolabelling
                        out = non_max_suppression(
                            out.float(),
                            conf_thres=conf_thres,
                            iou_thres=iou_thres,
                            labels=lb,
                            multi_label=True,
                        )

                        for si, pred in enumerate(out):
                            # Rescale boxes from img_size
                            predn = pred.clone()
                            scale_coords(
                                img[si].shape[1:],
                                predn[:, :4],
                                shapes[si][0],
                                shapes[si][1],
                            )  # native-space pred

                latency = batch_time.avg / batch_size * 1000
                perf = batch_size / batch_time.avg

                # Print perf
                print("Inference latency %.3f ms" % latency)
                print("Throughput: {:.3f} fps".format(perf))
    else:
        seen = 0
        confusion_matrix = ConfusionMatrix(nc=nc)
        coco91class = coco80_to_coco91_class()
        s = ("%20s" + "%12s" * 6) % (
            "Class",
            "Images",
            "Labels",
            "P",
            "R",
            "mAP@.5",
            "mAP@.5:.95",
        )
        p, r, f1, mp, mr, map50, map, t0, t1 = (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        loss = torch.zeros(3, device=device)
        jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
        print("***** Running Evaluation *****")
        for batch_i, (img, targets, paths, shapes) in enumerate(
            tqdm(dataloader, desc=s)
        ):
            img = img.to(device, non_blocking=True)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img = img.contiguous(memory_format=torch.channels_last)
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if bf16:
                img = img.to(torch.bfloat16)
            if fp16:
                img = img.to(torch.half)
            targets = targets.to(device)
            nb, _, height, width = img.shape  # batch size, channels, height, width

            with torch.no_grad():
                # Run model
                t = time_synchronized()
                if bf16:
                    with torch.autocast("cpu", dtype=torch.bfloat16):
                        out, train_out = model(img, augment=augment)
                elif fp16:
                    with torch.autocast("cpu", dtype=torch.half):
                        out, train_out = model(img, augment=augment)
                else:
                    out, train_out = model(
                        img, augment=augment
                    )  # inference and training outputs
                t0 += time_synchronized() - t

                # Compute loss
                if compute_loss:
                    loss += compute_loss([x.float() for x in train_out], targets)[1][
                        :3
                    ]  # box, obj, cls

                # Run NMS
                targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(
                    device
                )  # to pixels
                lb = (
                    [targets[targets[:, 0] == i, 1:] for i in range(nb)]
                    if save_hybrid
                    else []
                )  # for autolabelling
                t = time_synchronized()
                out = non_max_suppression(
                    out.float(),
                    conf_thres=conf_thres,
                    iou_thres=iou_thres,
                    labels=lb,
                    multi_label=True,
                )
                t1 += time_synchronized() - t

            # Statistics per image
            for si, pred in enumerate(out):
                labels = targets[targets[:, 0] == si, 1:]
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []  # target class
                path = Path(paths[si])
                seen += 1

                if len(pred) == 0:
                    if nl:
                        stats.append(
                            (
                                torch.zeros(0, niou, dtype=torch.bool),
                                torch.Tensor(),
                                torch.Tensor(),
                                tcls,
                            )
                        )
                    continue

                # Predictions
                predn = pred.clone()
                scale_coords(
                    img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1]
                )  # native-space pred

                # Append to text file
                if save_txt:
                    gn = torch.tensor(shapes[si][0])[
                        [1, 0, 1, 0]
                    ]  # normalization gain whwh
                    for *xyxy, conf, cls in predn.tolist():
                        xywh = (
                            (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn)
                            .view(-1)
                            .tolist()
                        )  # normalized xywh
                        line = (
                            (cls, *xywh, conf) if save_conf else (cls, *xywh)
                        )  # label format
                        with open(save_dir / "labels" / (path.stem + ".txt"), "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                # W&B logging - Media Panel Plots
                if (
                    len(wandb_images) < log_imgs and wandb_logger.current_epoch > 0
                ):  # Check for test operation
                    if wandb_logger.current_epoch % wandb_logger.bbox_interval == 0:
                        box_data = [
                            {
                                "position": {
                                    "minX": xyxy[0],
                                    "minY": xyxy[1],
                                    "maxX": xyxy[2],
                                    "maxY": xyxy[3],
                                },
                                "class_id": int(cls),
                                "box_caption": "%s %.3f" % (names[cls], conf),
                                "scores": {"class_score": conf},
                                "domain": "pixel",
                            }
                            for *xyxy, conf, cls in pred.tolist()
                        ]
                        boxes = {
                            "predictions": {"box_data": box_data, "class_labels": names}
                        }  # inference-space
                        wandb_images.append(
                            wandb_logger.wandb.Image(
                                img[si], boxes=boxes, caption=path.name
                            )
                        )
                (
                    wandb_logger.log_training_progress(predn, path, names)
                    if wandb_logger and wandb_logger.wandb_run
                    else None
                )

                # Append to pycocotools JSON dictionary
                if save_json:
                    # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                    box = xyxy2xywh(predn[:, :4])  # xywh
                    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                    for p, b in zip(pred.tolist(), box.tolist()):
                        jdict.append(
                            {
                                "image_id": image_id,
                                "category_id": (
                                    coco91class[int(p[5])] if is_coco else int(p[5])
                                ),
                                "bbox": [round(x, 3) for x in b],
                                "score": round(p[4], 5),
                            }
                        )

                # Assign all predictions as incorrect
                correct = torch.zeros(
                    pred.shape[0], niou, dtype=torch.bool, device=device
                )
                if nl:
                    detected = []  # target indices
                    tcls_tensor = labels[:, 0]

                    # target boxes
                    tbox = xywh2xyxy(labels[:, 1:5])
                    scale_coords(
                        img[si].shape[1:], tbox, shapes[si][0], shapes[si][1]
                    )  # native-space labels
                    if plots:
                        confusion_matrix.process_batch(
                            predn, torch.cat((labels[:, 0:1], tbox), 1)
                        )

                    # Per target class
                    for cls in torch.unique(tcls_tensor):
                        ti = (
                            (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)
                        )  # prediction indices
                        pi = (
                            (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)
                        )  # target indices

                        # Search for detections
                        if pi.shape[0]:
                            # Prediction to target ious
                            ious, i = box_iou(predn[pi, :4], tbox[ti]).max(
                                1
                            )  # best ious, indices

                            # Append detections
                            detected_set = set()
                            for j in (ious > iouv[0]).nonzero(as_tuple=False):
                                d = ti[i[j]]  # detected target
                                if d.item() not in detected_set:
                                    detected_set.add(d.item())
                                    detected.append(d)
                                    correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                    if (
                                        len(detected) == nl
                                    ):  # all targets already located in image
                                        break

                # Append statistics (correct, conf, pcls, tcls)
                stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

            # Plot images
            if plots and batch_i < 3:
                f = save_dir / f"test_batch{batch_i}_labels.jpg"  # labels
                Thread(
                    target=plot_images,
                    args=(img, targets, paths, f, names),
                    daemon=True,
                ).start()
                f = save_dir / f"test_batch{batch_i}_pred.jpg"  # predictions
                Thread(
                    target=plot_images,
                    args=(img, output_to_target(out), paths, f, names),
                    daemon=True,
                ).start()

        # Compute statistics
        stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
        if len(stats) and stats[0].any():
            p, r, ap, f1, ap_class = ap_per_class(
                *stats, plot=plots, v5_metric=v5_metric, save_dir=save_dir, names=names
            )
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(
                stats[3].astype(np.int64), minlength=nc
            )  # number of targets per class
        else:
            nt = torch.zeros(1)

        # Print results
        pf = "%20s" + "%12i" * 2 + "%12.3g" * 4  # print format
        print(pf % ("all", seen, nt.sum(), mp, mr, map50, map))

        # Print results per class
        if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
            for i, c in enumerate(ap_class):
                print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

        # Print speeds
        t = tuple(x / seen * 1e3 for x in (t0, t1, t0 + t1)) + (
            imgsz,
            imgsz,
            batch_size,
        )  # tuple
        if not training:
            print(
                "Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g"
                % t
            )

        # Plots
        if plots:
            confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
            if wandb_logger and wandb_logger.wandb:
                val_batches = [
                    wandb_logger.wandb.Image(str(f), caption=f.name)
                    for f in sorted(save_dir.glob("test*.jpg"))
                ]
                wandb_logger.log({"Validation": val_batches})
        if wandb_images:
            wandb_logger.log({"Bounding Box Debugger/Images": wandb_images})

        # Save JSON
        if save_json and len(jdict):
            w = (
                Path(weights[0] if isinstance(weights, list) else weights).stem
                if weights is not None
                else ""
            )  # weights
            anno_json = os.path.join(
                opt.dataset_dir, "coco/annotations/instances_val2017.json"
            )  # annotations json
            pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
            print("\nEvaluating pycocotools mAP... saving %s..." % pred_json)
            with open(pred_json, "w") as f:
                json.dump(jdict, f)

            try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
                from pycocotools.coco import COCO
                from pycocotools.cocoeval import COCOeval

                anno = COCO(anno_json)  # init annotations api
                pred = anno.loadRes(pred_json)  # init predictions api
                eval = COCOeval(anno, pred, "bbox")
                if is_coco:
                    eval.params.imgIds = [
                        int(Path(x).stem) for x in dataloader.dataset.img_files
                    ]  # image IDs to evaluate
                eval.evaluate()
                eval.accumulate()
                eval.summarize()
                map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
            except Exception as e:
                print(f"pycocotools unable to run: {e}")

        # Return results
        model.float()  # for training
        if not training:
            s = (
                f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}"
                if save_txt
                else ""
            )
            print(f"Results saved to {save_dir}{s}")
        maps = np.zeros(nc) + map
        for i, c in enumerate(ap_class):
            maps[c] = ap[i]
        return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="test.py")
    parser.add_argument(
        "--weights", nargs="+", type=str, default="yolov7.pt", help="model.pt path(s)"
    )
    parser.add_argument("--data", type=str, default="", help="*.data path")
    parser.add_argument(
        "--batch-size", type=int, default=32, help="size of each image batch"
    )
    parser.add_argument(
        "--img-size", type=int, default=640, help="inference size (pixels)"
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.001, help="object confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.65, help="IOU threshold for NMS"
    )
    parser.add_argument(
        "--task", default="val", help="train, val, test, speed or study"
    )
    parser.add_argument(
        "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument(
        "--single-cls", action="store_true", help="treat as single-class dataset"
    )
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--verbose", action="store_true", help="report mAP by class")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument(
        "--save-hybrid",
        action="store_true",
        help="save label+prediction hybrid results to *.txt",
    )
    parser.add_argument(
        "--save-conf", action="store_true", help="save confidences in --save-txt labels"
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="save a cocoapi-compatible JSON results file",
    )
    parser.add_argument("--project", default="runs/test", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )
    parser.add_argument("--no-trace", action="store_true", help="don`t trace model")
    parser.add_argument(
        "--v5-metric",
        action="store_true",
        help="assume maximum recall as 1.0 in AP calculation",
    )
    parser.add_argument(
        "--inductor",
        action="store_true",
        default=False,
        help="use torch.compile() default inductor backend",
    )
    parser.add_argument(
        "--seed", default=None, type=int, help="seed for initializing inference. "
    )
    parser.add_argument(
        "--int8", action="store_true", default=False, help="enable int8 path"
    )
    parser.add_argument(
        "--bf16", action="store_true", default=False, help="enable bf16 path"
    )
    parser.add_argument(
        "--fp16", action="store_true", default=False, help="enable fp16 path"
    )
    parser.add_argument(
        "--calibration",
        action="store_true",
        default=False,
        help="doing calibration step for int8 path",
    )
    parser.add_argument(
        "--configure-dir",
        default="configure.json",
        type=str,
        metavar="PATH",
        help="path to int8 configures, default file name is configure.json",
    )
    parser.add_argument(
        "--performance", action="store_true", help="test the performance of inference"
    )
    parser.add_argument(
        "-w",
        "--warmup-iterations",
        default=20,
        type=int,
        metavar="N",
        help="number of warmup iterati ons to run",
    )
    parser.add_argument(
        "--weight-sharing",
        action="store_true",
        default=False,
        help="using weight_sharing to test the performance of inference",
    )
    parser.add_argument(
        "--number-instance",
        default=0,
        type=int,
        help="the instance numbers for test the performance of latcy, only works when enable weight-sharing",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    parser.add_argument("--steps", default=-1, type=int, help="steps for validation")
    parser.add_argument(
        "--dataset-dir",
        default="",
        type=str,
        help="directory which contains coco dataset",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="",
        type=str,
        help="directory which contains pretrained model",
    )
    parser.add_argument(
        "--calibration-steps", default=10, type=int, help="calibration iteration number"
    )
    parser.add_argument(
        "--drop-last",
        action="store_true",
        default=False,
        help="drop last iteration for val dataloader",
    )
    parser.add_argument(
        "--prepare-dataloader",
        action="store_true",
        default=False,
        help="prepare for val dataloader",
    )
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith("coco.yaml")
    opt.data = check_file(opt.data)  # check file
    print(opt)

    if opt.task in ("train", "val", "test"):  # run normally
        test(
            opt,
            opt.data,
            opt.weights,
            opt.batch_size,
            opt.img_size,
            opt.conf_thres,
            opt.iou_thres,
            opt.save_json,
            opt.single_cls,
            opt.augment,
            opt.verbose,
            opt.inductor,
            opt.seed,
            opt.int8,
            opt.bf16,
            opt.fp16,
            opt.calibration,
            opt.configure_dir,
            opt.performance,
            opt.warmup_iterations,
            opt.weight_sharing,
            opt.number_instance,
            opt.evaluate,
            save_txt=opt.save_txt | opt.save_hybrid,
            save_hybrid=opt.save_hybrid,
            save_conf=opt.save_conf,
            trace=not opt.no_trace,
            v5_metric=opt.v5_metric,
        )

    elif opt.task == "speed":  # speed benchmarks
        for w in opt.weights:
            test(
                opt.data,
                w,
                opt.batch_size,
                opt.img_size,
                0.25,
                0.45,
                save_json=False,
                plots=False,
                v5_metric=opt.v5_metric,
            )

    elif opt.task == "study":  # run over a range of settings and save/plot
        # python test.py --task study --data coco.yaml --iou 0.65 --weights yolov7.pt
        x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
        for w in opt.weights:
            f = f"study_{Path(opt.data).stem}_{Path(w).stem}.txt"  # filename to save to
            y = []  # y axis
            for i in x:  # img-size
                print(f"\nRunning {f} point {i}...")
                r, _, t = test(
                    opt.data,
                    w,
                    opt.batch_size,
                    i,
                    opt.conf_thres,
                    opt.iou_thres,
                    opt.save_json,
                    plots=False,
                    v5_metric=opt.v5_metric,
                )
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt="%10.4g")  # save
        os.system("zip -r study.zip study_*.txt")
        plot_study_txt(x=x)  # plot
