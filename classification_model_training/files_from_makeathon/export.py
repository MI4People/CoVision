import argparse
import os
import torch

import onnx
from onnx_tf.backend import prepare

from efficientnet_pytorch import EfficientNet
from torch.autograd import Variable


def export_onnx(single_model_path: str, num_classes: int, outdir: str):
    model = EfficientNet.from_name(
        "efficientnet-b2", in_channels=3, num_classes=num_classes
    )
    model.load_state_dict(torch.load(single_model_path))
    model.set_swish(memory_efficient=False)

    dummy_input = Variable(torch.rand(1, 3, 224, 224))
    torch.onnx.export(model, dummy_input, os.path.join(outdir, "model_best.onnx"))


def export_pb(indir: str, outdir: str):
    onnx_model = onnx.load(os.path.join(indir, "model_best.onnx"))
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(os.path.join(outdir, "model_best.pb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=4, help="number of classes")
    parser.add_argument(
        "--single_model_path",
        default=None,
        type=str,
        help="Use only for single model prediction: path to the folder containing the model file: .bin",
    )
    parser.add_argument(
        "--outdir", type=str, help="outputdir where the files are stored"
    )
    opt = parser.parse_args()

    export_onnx(opt.single_model_path, opt.num_classes, opt.outdir)
    export_pb(opt.outdir, opt.outdir)
