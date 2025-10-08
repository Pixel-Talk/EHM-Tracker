import os
import numpy as np
import onnxruntime as ort
class OnnxModel:
    def __init__(self, ckpt_fp:str, input_keys=[], output_keys=[], use_gpu=True) -> None:
        if use_gpu:
            providers = ['CUDAExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        self.ort_session = ort.InferenceSession(ckpt_fp, providers=providers)
        self.input_keys  = input_keys
        self.output_keys = output_keys

    def __call__(self, *args):
        input_kwargs = {k: v for k, v in zip(self.input_keys, args)}

        ret = self.ort_session.run(None, input_kwargs)
        return {k: v for k, v in zip(self.output_keys, ret)}
    
    def run(self, *args):
        return self(*args)
    