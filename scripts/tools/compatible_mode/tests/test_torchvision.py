import compatible_mode

import torch
from torchvision import models, transforms
from torchvision.models import get_model_builder, list_models


dev = "cuda"

# The following contains configuration parameters for all models which are used by
# the _test_*_model methods.
_model_params = {
    "inception_v3": {"input_shape": (1, 3, 299, 299), "init_weights": True},
    "retinanet_resnet50_fpn": {
        "num_classes": 20,
        "score_thresh": 0.01,
        "min_size": 224,
        "max_size": 224,
        "input_shape": (3, 224, 224),
        "real_image": True,
    },
    "retinanet_resnet50_fpn_v2": {
        "num_classes": 20,
        "score_thresh": 0.01,
        "min_size": 224,
        "max_size": 224,
        "input_shape": (3, 224, 224),
        "real_image": True,
    },
    "keypointrcnn_resnet50_fpn": {
        "num_classes": 2,
        "min_size": 224,
        "max_size": 224,
        "box_score_thresh": 0.17,
        "input_shape": (3, 224, 224),
        "real_image": True,
    },
    "fasterrcnn_resnet50_fpn": {
        "num_classes": 20,
        "min_size": 224,
        "max_size": 224,
        "input_shape": (3, 224, 224),
        "real_image": True,
    },
    "fasterrcnn_resnet50_fpn_v2": {
        "num_classes": 20,
        "min_size": 224,
        "max_size": 224,
        "input_shape": (3, 224, 224),
        "real_image": True,
    },
    "fcos_resnet50_fpn": {
        "num_classes": 2,
        "score_thresh": 0.05,
        "min_size": 224,
        "max_size": 224,
        "input_shape": (3, 224, 224),
        "real_image": True,
    },
    "maskrcnn_resnet50_fpn": {
        "num_classes": 10,
        "min_size": 224,
        "max_size": 224,
        "input_shape": (3, 224, 224),
        "real_image": True,
    },
    "maskrcnn_resnet50_fpn_v2": {
        "num_classes": 10,
        "min_size": 224,
        "max_size": 224,
        "input_shape": (3, 224, 224),
        "real_image": True,
    },
    "fasterrcnn_mobilenet_v3_large_fpn": {
        "box_score_thresh": 0.02076,
    },
    "fasterrcnn_mobilenet_v3_large_320_fpn": {
        "box_score_thresh": 0.02076,
        "rpn_pre_nms_top_n_test": 1000,
        "rpn_post_nms_top_n_test": 1000,
    },
    "vit_h_14": {
        "image_size": 56,
        "input_shape": (1, 3, 56, 56),
    },
    "mvit_v1_b": {
        "input_shape": (1, 3, 16, 224, 224),
    },
    "mvit_v2_s": {
        "input_shape": (1, 3, 16, 224, 224),
    },
    "s3d": {
        "input_shape": (1, 3, 16, 224, 224),
    },
    "googlenet": {"init_weights": True},
}

def _get_image(input_shape, real_image, device, dtype=None):
    """This routine loads a real or random image based on `real_image` argument.
    Currently, the real image is utilized for the following list of models:
    - `retinanet_resnet50_fpn`,
    - `retinanet_resnet50_fpn_v2`,
    - `keypointrcnn_resnet50_fpn`,
    - `fasterrcnn_resnet50_fpn`,
    - `fasterrcnn_resnet50_fpn_v2`,
    - `fcos_resnet50_fpn`,
    - `maskrcnn_resnet50_fpn`,
    - `maskrcnn_resnet50_fpn_v2`,
    in `test_classification_model` and `test_detection_model`.
    To do so, a keyword argument `real_image` was added to the abovelisted models in `_model_params`
    """
    if real_image:
        # TODO: Maybe unify file discovery logic with test_image.py
        GRACE_HOPPER = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "assets", "encode_jpeg", "grace_hopper_517x606.jpg"
        )

        img = Image.open(GRACE_HOPPER)

        original_width, original_height = img.size

        # make the image square
        img = img.crop((0, 0, original_width, original_width))
        img = img.resize(input_shape[1:3])

        convert_tensor = transforms.ToTensor()
        image = convert_tensor(img)
        assert tuple(image.size()) == input_shape
        return image.to(device=device, dtype=dtype)

    # RNG always on CPU, to ensure x in cuda tests is bitwise identical to x in cpu tests
    return torch.rand(input_shape).to(device=device, dtype=dtype)

def list_model_fns(module):
    return [get_model_builder(name) for name in list_models(module)]



def _test_classification_model_eval(model_fn):
    defaults = {
        "num_classes": 50,
        "input_shape": (1, 3, 224, 224),
    }
    model_name = model_fn.__name__
    kwargs = {**defaults, **_model_params.get(model_name, {})}
    num_classes = kwargs.get("num_classes")
    input_shape = kwargs.pop("input_shape")
    real_image = kwargs.pop("real_image", False)

    model = model_fn(**kwargs)
    model.eval().to(device=dev)
    x = _get_image(input_shape=input_shape, real_image=real_image, device=dev)
    out = model(x)

    print("Testing model:{} args: {}, result:{}".format(model_name, kwargs, out))

def test_classification_model_eval():
    for model in list_model_fns(models):
        try:
            _test_classification_model_eval(model)
        except Exception as e:
            print("model_name: {}, Exception :{}".format(model.__name__, e))



def _test_detection_model_eval(model_fn):
    model_name = model_fn.__name__
    input_shape = (3, 300, 300)
    model = model_fn()
    model.eval().to(device=dev)

    x = [torch.rand(input_shape, device=dev)]
    out = model(x)
    print("Testing detection model:{} result:{}".format(model_name, out))


def test_detection_model_eval():
    for model in list_model_fns(models.detection):
        try:
            _test_detection_model_eval(model)
        except Exception as e:
            print("model_name: {}, Exception :{}".format(model.__name__, e))


def _test_quantized_classification_model(model_fn):
    defaults = {
        "num_classes": 5,
        "input_shape": (1, 3, 224, 224),
        "quantize": True,
    }
    model_name = model_fn.__name__
    kwargs = {**defaults, **_model_params.get(model_name, {})}
    input_shape = kwargs.pop("input_shape")

    model = model_fn(**kwargs)
    model.eval()
    x = torch.rand(input_shape)
    out = model(x)
    print("Testing quantized_classification model:{} args: {}, result:{}".format(model_name, kwargs, out))


def test_quantized_classification_model():
    for model in list_model_fns(models.quantization):
        try:
            _test_quantized_classification_model(model)
        except Exception as e:
            print("model_name: {}, Exception :{}".format(model.__name__, e))
        
        


def _test_segmentation_model_eval(model_fn):
    defaults = {
        "num_classes": 10,
        "weights_backbone": None,
        "input_shape": (1, 3, 32, 32),
    }
    model_name = model_fn.__name__
    kwargs = {**defaults, **_model_params.get(model_name, {})}
    input_shape = kwargs.pop("input_shape")

    model = model_fn(**kwargs)
    model.eval().to(device=dev)

    x = torch.rand(input_shape).to(device=dev)

    out = model(x)
    print("Testing segmentation_model model:{} args: {}, result:{}".format(model_name, kwargs, out))


def test_segmentation_model_eval():
    for model in list_model_fns(models.segmentation):
        try:
            _test_segmentation_model_eval(model)
        except Exception as e:
            print("model_name: {}, Exception :{}".format(model.__name__, e))
        


def _test_video_model_eval(model_fn):
    defaults = {
        "input_shape": (1, 3, 4, 112, 112),
        "num_classes": 50,
    }
    model_name = model_fn.__name__

    kwargs = {**defaults, **_model_params.get(model_name, {})}
    num_classes = kwargs.get("num_classes")
    input_shape = kwargs.pop("input_shape")
    # test both basicblock and Bottleneck
    model = model_fn(**kwargs)
    model.eval().to(device=dev)
    # RNG always on CPU, to ensure x in cuda tests is bitwise identical to x in cpu tests
    x = torch.rand(input_shape).to(device=dev)
    out = model(x)
    print("Testing video model:{} args: {}, result:{}".format(model_name, kwargs, out))


def test_video_model_eval():
    for model in list_model_fns(models.video):
        try:
            _test_video_model_eval(model)
        except Exception as e:
            print("model_name: {}, Exception :{}".format(model.__name__, e))
        


def _test_raft_eval(model_fn):
    torch.manual_seed(0)
    model_name = model_fn.__name__
    # We need very small images, otherwise the pickle size would exceed the 50KB
    # As a result we need to override the correlation pyramid to not downsample
    # too much, otherwise we would get nan values (effective H and W would be
    # reduced to 1)
    corr_block = models.optical_flow.raft.CorrBlock(num_levels=2, radius=2)

    model = model_fn(corr_block=corr_block).eval().to(dev)

    bs = 1
    img1 = torch.rand(bs, 3, 80, 72, device=dev)
    img2 = torch.rand(bs, 3, 80, 72, device=dev)

    preds = model(img1, img2)
    flow_pred = preds[-1]
    print("Testing raft model:{} , result:{}".format(model_name, flow_pred))


def test_raft_eval():
    for model in list_model_fns(models.optical_flow):
        try:
            _test_raft_eval(model)
        except Exception as e:
            print("model_name: {}, Exception :{}".format(model.__name__, e))


if __name__ == "__main__":
    test_classification_model_eval()
    test_detection_model_eval()
    test_quantized_classification_model()
    test_segmentation_model_eval()
    test_video_model_eval()
    test_raft_eval()
    