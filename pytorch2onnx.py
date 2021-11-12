from model.mfad import FAD_HAM_Net
import torch
import onnxruntime
import argparse
import numpy as np
import os.path as osp
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Convert a PyTorch model to ONNX')
    parser.add_argument('--checkpoint', type=str, help='checkpoint file', required=True)
    parser.add_argument('--output', type=str, help='An output path for an ONNX model', required=True)
    parser.add_argument('--opset', type=int, default=11, help="ONNX opset")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    assert osp.exists(args.checkpoint), "File doesn't exist"
    # initialize model
    net = FAD_HAM_Net(pretrain=True, variant='resnet50')
    net.load_state_dict(torch.load(args.checkpoint, map_location=torch.device('cpu')))
    net.eval()
    # Create dummy values
    dummy_input = torch.rand((1, 3, 224, 224))
    # Convert a PyTorch model to ONNX
    torch.onnx.export(net, dummy_input, args.output, opset_version=args.opset, input_names=['input'],
                      output_names=['output'], dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                      keep_initializers_as_inputs=False)
    # Create dummy values to infer models
    dummy_input = torch.rand((2, 3, 224, 224))
    dummy_array = dummy_input.numpy()
    # Infer the PyTorch model
    out_pytorch = net(dummy_input)
    # Infer the ONNX model
    providers = ['CPUExecutionProvider']
    session = onnxruntime.InferenceSession(args.output, providers=providers)
    out_onnx = session.run(['output'], {'input': dummy_array})[0]
    # Validate
    try:
        np.testing.assert_array_almost_equal(out_pytorch.detach().numpy(), out_onnx)
        print("Model has validated and converted successfully")
    except AssertionError:
        os.remove(args.output)
        print("Model conversion failed (validation failed)")
