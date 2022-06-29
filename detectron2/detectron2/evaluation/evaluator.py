# Copyright (c) Facebook, Inc. and its affiliates.
import datetime
import logging
import time
from collections import OrderedDict, abc
from contextlib import ExitStack, contextmanager
from typing import List, Union
from cv2 import resize
from detectron2.structures import image_list
from matplotlib import patches
import torch
from torch import nn

from detectron2.utils.comm import get_world_size, is_main_process
from detectron2.utils.logger import log_every_n_seconds

#adda
import numpy as np
import cv2
from typing import Iterable
from torch import Tensor

class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        .. code-block:: python

            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...

        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


class DatasetEvaluators(DatasetEvaluator):
    """
    Wrapper class to combine multiple :class:`DatasetEvaluator` instances.

    This class dispatches every evaluation call to
    all of its :class:`DatasetEvaluator`.
    """

    def __init__(self, evaluators):
        """
        Args:
            evaluators (list): the evaluators to combine.
        """
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, inputs, outputs):
        for evaluator in self._evaluators:
            evaluator.process(inputs, outputs)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process() and result is not None:
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results


def inference_on_dataset(
    model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None,], sess=None, patching=False
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        for idx, inputs in enumerate(data_loader):
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            
            if sess is not None or patching == True:
                #Compare panoptic label list
                #print("ORIGINAL INPUT SIZE")
                #model.network_only = False
                #model.onnx = False
                #outputs = model(inputs)
                #Use modified inference
                
                print("RESIZING OUTPUTS")
                model.onnx = True
                #outputs = modified_inference(model, inputs, sess=sess, patching=patching)
                inputs = preprocess_input(inputs, model, resize=False)
                image = inputs[0]['image'].numpy().astype(np.float32)
                #expand dim.
                #image = np.expand_dims(image, axis=0)
                sem_seg_results, center_results, offset_results = sess.run(None, {'image': image} )
                
                #outputs of sess.run() are numpy arrays
                sem_seg_results = torch.from_numpy(sem_seg_results).to(model.device)
                center_results = torch.from_numpy(center_results).to(model.device)
                offset_results = torch.from_numpy(offset_results).to(model.device)
                #don't upsample in postprocess
                #inputs[0]['height'] //= 2
                #inputs[0]['width'] //= 2
                imagelist = get_imagelist(model, inputs)
                outputs = exec_postprocess(model, sem_seg_results,center_results,offset_results, inputs, imagelist)
                #resize outputs
                #outputs = resize_output(model, sem_seg_results, center_results, outputs, 1024, 2048)
            else:
                model.network_only=False
                print("IMAGE SIZE", inputs[0]['image'].size())
                inputs = resize_input(inputs)
                outputs = model(inputs)
                '''
                model.network_only=True
                sem_seg_results, center_results, offset_results = model(inputs)
                inputs[0]['height'] //= 2
                inputs[0]['width'] //= 2
                imagelist = get_imagelist(model, inputs)
                outputs = exec_postprocess(model, sem_seg_results,center_results,offset_results, inputs, imagelist)
                outputs = resize_output(model, sem_seg_results, center_results, outputs, 1024, 2048)
                '''
            '''
            #for short eval
            image = inputs[0]['image']
            print("IMG SIZE", image.size())
            image = (image-model.pixel_mean.cpu().detach()) / model.pixel_std.cpu().detach()
            image = image.numpy().astype(np.float32)
            sem_seg_results, center_results, offset_results = sess.run(None, {'image': image} )
            sem_seg_results = torch.from_numpy(sem_seg_results).to(model.device)
            center_results = torch.from_numpy(center_results).to(model.device)
            offset_results = torch.from_numpy(offset_results).to(model.device)
            imagelist = get_imagelist(model, inputs)
            outputs = exec_postprocess(model, sem_seg_results,center_results,offset_results, inputs, imagelist)
            '''
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()
            evaluator.process(inputs, outputs)
            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"ETA={eta}"
                    ),
                    n=5,
                )
            start_data_time = time.perf_counter()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)

def create_hist_dict(histogram):
    hist = {'semantic':{'res5':[],'res3':[],'res2':[]},'instance':{'res5':[],'res3':[],'res2':[]}}
    for i in range(12):
        if i<=5:
            hist['semantic']['res5'].append(histogram[i])
        elif i<=8:
            hist['semantic']['res3'].append(histogram[i])
        else:
            hist['semantic']['res2'].append(histogram[i])
    for i in range(12,24):
        if i-12<=5:
            hist['instance']['res5'].append(histogram[i])
        elif i-12<=8:
            hist['instance']['res3'].append(histogram[i])
        else:
            hist['instance']['res2'].append(histogram[i])
    return hist

def print_hist(histogram,tab=""):
    tab += "\t"
    if type(histogram) is dict:
        for key in histogram.keys():
            print("%s%s:"%(tab,key))
            print_hist(histogram[key], tab)
    else:
        for i, arr in enumerate(histogram):
            max_100,min_100,max_95,min_95 = get_min_max(arr)
            print("%s%dth max min max95 min95:%.2f,%.6f,%.2f,%.6f"%(tab,i,max_100,min_100,max_95,min_95))
            _,c,h,w=arr.shape
            if torch.is_tensor(arr):
                print("%ssparsity:%f"%(tab,1 - torch.count_nonzero(arr)/(c*h*w)))
            else:
                print("%ssparsity:%f"%(tab,1 - np.count_nonzero(arr)/(c*h*w)))

def get_min_max(x):
    #hist, bin= np.histogram(x)
    if torch.is_tensor(x):
        max_100 = torch.max(x)
        min_100 = torch.min(x)
        max_95 = torch.quantile(x,0.95)
        min_95 = torch.quantile(x,0.05)
    else:
        max_100 = x.max()
        min_100 = x.min()
        max_95 = np.percentile(x,95)
        min_95 = np.percentile(x,5)
    return max_100,min_100,max_95,min_95


def modified_inference(model, inputs, sess=None, patching=False):
    if patching:
        #model fed with each patch
        patches_outputs = []
        indices = [(0,0), (0,1), (1,0), (1,1)]
        image_origin = inputs[0]['image']
        for idx in indices:
            h = inputs[0]['height'] // 2 
            w = inputs[0]['width'] // 2
            inputs[0]['image'] = image_origin[:,idx[0]*h:(idx[0]+1)*h,idx[1]*w:(idx[1]+1)*w]
            #inference = inference_small_input(inputs, model, sess=sess)
            #sem_seg = inference[0]['sem_seg']
            #panoptic_seg = inference[0]['panoptic_seg']
            #instances = inference[0]['instances']
            #patches_outputs.append((sem_seg, panoptic_seg, instances))

            #Patching
            patches_outputs.append(inference_small_input(inputs, model, sess=sess))
        sem_seg_results, center_results, offset_results = patching_outputs(patches_outputs)
        imagelist = get_imagelist(model, inputs)
        outputs = exec_postprocess(model, sem_seg_results,center_results,offset_results, inputs, imagelist)
        #Patching Panoptic segmentation and Instances after postprocess
        #outputs = patching_outputs(patches_outputs)

    else:
        #Conver BRR to RGB for ONNX model only
        idx_tensor = torch.LongTensor([2,1,0])
        inputs[0]['image'] = inputs[0]['image'].index_select(0, idx_tensor)
        #resize
        inputs = resize_input(inputs)
        #sem_seg_results, center_results, offset_results = inference_small_input(inputs, model, sess=sess)
        outputs = inference_small_input(inputs, model, sess=sess)

    return outputs

def preprocess_input(inputs, model, resize=True):
    if resize:
        inputs = resize_input(inputs)
    inputs[0]['image'] = (inputs[0]['image'] - model.pixel_mean.cpu().detach()) / model.pixel_std.cpu().detach()
    return inputs

def resize_input(inputs):
    #resize input images into 512*1024
    image = inputs[0]['image']
    image = np.array(image).astype(dtype=np.float32)
    image = image.transpose(1,2,0)
    image = cv2.resize(image, dsize=(1024,512), interpolation=cv2.INTER_AREA)
    image = image.transpose(2,0,1)
    inputs[0]['image'] = torch.from_numpy(image)

    return inputs

def get_imagelist(model, inputs):
    size_divisibility = (
    model.size_divisibility
    if model.size_divisibility > 0
    else model.backbone.size_divisibility
    )
    from detectron2.structures import BitMasks, ImageList, Instances
    images = ImageList.from_tensors([inputs[0]['image']], size_divisibility)

    return images

def inference_small_input(inputs, model, sess=None):
    '''
    Return: sem_seg_results, center_results, offset_results  -> changed -> return outputs
    '''
    image = inputs[0]['image']
    image = np.array(image).astype(dtype=np.float32)
        
    #If using onnx model, run the inference session.
    if sess is not None:
        image = np.expand_dims(image, axis=0)
        sem_seg_results, center_results, offset_results = sess.run(None, {'image': image} )
        #outputs of sess.run() are numpy arrays
        sem_seg_results = torch.from_numpy(sem_seg_results).to(model.device)
        center_results = torch.from_numpy(center_results).to(model.device)
        offset_results = torch.from_numpy(offset_results).to(model.device)
    else:
        model.network_only = False
        #sem_seg_results, center_results, offset_results = model(inputs)
        return model(inputs)
    
    imagelist = get_imagelist(model, inputs)
    outputs = exec_postprocess(model, sem_seg_results, center_results, offset_results, inputs, imagelist)
    return outputs

def patching_outputs(patches_outputs):
    '''
    Inputs: list of sem_seg, center, offset results over patches
            each are dim [1, num targets, H, W] numpy arrays
    '''

    sem_seg_results, center_results, offset_results = list(zip(*patches_outputs))
    sem_output1 = torch.cat([sem_seg_results[0],sem_seg_results[1]], dim=-1)
    sem_output2 = torch.cat([sem_seg_results[2],sem_seg_results[3]], dim=-1)

    center_output1 = torch.cat([center_results[0],center_results[1]], dim=-1)
    center_output2 = torch.cat([center_results[2],center_results[3]], dim=-1)

    offset_output1 = torch.cat([offset_results[0],offset_results[1]], dim=-1)
    offset_output2 = torch.cat([offset_results[2],offset_results[3]], dim=-1)

    return (torch.cat([sem_output1, sem_output2], dim=-2),
            torch.cat([center_output1, center_output2], dim=-2), 
            torch.cat([offset_output1, offset_output2], dim=-2)
            )
    '''
    sem_seg, panoptic_seg, instances = list(zip(*patching_outputs))

    sem_seg_patched = patching_tensor(sem_seg)
    panoptic_seg_patched = patching_tensor(panoptic_seg)

    patched_results = []
    patched_results.append({'sem_seg':sem_seg_patched})
    patched_results[-1]['panoptic_seg'] = (panoptic_seg_patched, None)

    from detectron2.structures import BitMasks, ImageList, Instances
    image_size = instances.image_size
    instances_patched = Instances(image_size)
    pred_classes_patched = patching_tensor([inst.pred_classes for inst in instances])
    pred_masks_patched = patching_tensor([inst.pred_masks for inst in instances])

    return ()
    '''

def patching_tensor(tensors: Iterable[Tensor]):
    ret1 = torch.cat([tensors[0],tensors[1]], dim=-1)
    ret2 = torch.cat([tensors[2],tensors[3]], dim=-1)
    return torch.cat([ret1, ret2], dim=-2)



def exec_postprocess(model, sem_seg_results, center_results, offset_results, inputs, imagelist):

    outputs = model.post_process(sem_seg_results,
                                center_results,
                                offset_results, 
                                [{'height':inputs[0]['height'], 'width':inputs[0]['width']}],
                                imagelist
                                )
    return outputs

from detectron2.structures import BitMasks, ImageList, Instances
from torch.nn import functional as F

def resize_output(model, sem_seg_results, center_results, output, h_out, w_out):
    panoptic = output[0]['panoptic_seg'][0]
    panoptic_resized = nearest_resize(panoptic.unsqueeze(0).to(torch.float), panoptic.size(), h_out, w_out)[0].to(torch.long)
    
    r = nearest_resize(sem_seg_results, sem_seg_results.size()[-2:], h_out, w_out)
    semantic_prob = F.softmax(r, dim=0)
    c = bilinear_resize(center_results, center_results.size()[-2:], h_out, w_out)
    instance_resized = get_instance(model, panoptic_resized, h_out, w_out, semantic_prob, c)
    results = [{'sem_seg':r, 'panoptic_seg':(panoptic_resized, None)}]
    if instance_resized is not None:
        results[-1]["intances"] = instance_resized
    return results

def bilinear_resize(result, img_size, output_height, output_width):

    result = result[:, : img_size[0], : img_size[1]].expand(1, -1, -1, -1)
    
    result = F.interpolate(
        result, size=(output_height, output_width), mode="bilinear"
    )[0]

    return result

def nearest_resize(result, img_size, output_height, output_width):

    result = result[:, : img_size[0], : img_size[1]].expand(1, -1, -1, -1)
    
    result = F.interpolate(
        result, size=(output_height, output_width), mode="nearest-exact"
    )[0]

    return result

def get_instance(model, panoptic_image, height, width, semantic_prob, c):
    instances = []
    panoptic_image_cpu = panoptic_image.cpu().numpy()
    for panoptic_label in np.unique(panoptic_image_cpu):
        if panoptic_label == -1:
            continue
        pred_class = panoptic_label // model.meta.label_divisor
        isthing = pred_class in list(
            model.meta.thing_dataset_id_to_contiguous_id.values()
        )
        # Get instance segmentation results.
        if isthing:
            instance = Instances((height, width))
            # Evaluation code takes continuous id starting from 0
            instance.pred_classes = torch.tensor(
                [pred_class], device=panoptic_image.device
            )
            mask = panoptic_image == panoptic_label
            instance.pred_masks = mask.unsqueeze(0)
            # Average semantic probability
            sem_scores = semantic_prob[pred_class, ...]
            sem_scores = torch.mean(sem_scores[mask])
            # Center point probability
            mask_indices = torch.nonzero(mask).float()
            center_y, center_x = (
                torch.mean(mask_indices[:, 0]),
                torch.mean(mask_indices[:, 1]),
            )
            center_scores = c[0, int(center_y.item()), int(center_x.item())]
            # Confidence score is semantic prob * center prob.
            instance.scores = torch.tensor(
                [sem_scores * center_scores], device=panoptic_image.device
            )
            # Get bounding boxes
            instance.pred_boxes = BitMasks(instance.pred_masks).get_bounding_boxes()
            instances.append(instance)
    if len(instances) > 0:
        return Instances.cat(instances)