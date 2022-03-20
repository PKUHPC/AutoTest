import os
import time

import numpy as np
import tvm
from tvm import te, auto_scheduler, topi
from tvm.topi.nn.utils import get_pad_tuple


@auto_scheduler.register_workload
def conv_bwd_weight(N, CI, HI, WI, CO, HO, WO, KSIZE, stride, padding, dtype):
    strides = (stride, stride)
    shape_data = (N, CI, HI, WI)
    shape_weight = (CO, CI, KSIZE, KSIZE)
    shape_grad_output = (N, CO, HO, WO)
    # given tensor
    data = te.placeholder(shape_data, name="data", dtype=dtype)
    grad_output = te.placeholder(shape_grad_output, name="grad_output", dtype=dtype)
    # grad_weight
    dilation_h, dilation_w = (1, 1)
    batch, in_channel, in_h, in_w = shape_data
    out_channel, _, filter_h, filter_w = shape_weight
    grad_output_tmp = topi.tile(grad_output, [1, in_channel, 1, 1])
    grad_output_tmp = topi.reshape(grad_output_tmp, [batch * in_channel * out_channel, 1, HO, WO])
    data_tmp = topi.reshape(data, [1, in_channel * batch, HI, WI])
    grad_weight = topi.nn.group_conv2d_nchw(data_tmp, grad_output_tmp, stride=(dilation_h, dilation_w), 
        padding=padding, dilation=strides, groups=in_channel * batch, out_dtype=dtype)
    # infer shape of grad_weight
    _, _, grad_h, grad_w = shape_grad_output
    fpad_top, fpad_left, fpad_bottom, fpad_right = get_pad_tuple(padding, (filter_h, filter_w))
    padded_weight_grad_h = (in_h - (grad_h - 1) * strides[0] - 1 + fpad_top + fpad_bottom) // dilation_h + 1
    padded_weight_grad_w = (in_w - (grad_w - 1) * strides[1] - 1 + fpad_left + fpad_right) // dilation_w + 1
    grad_weight = topi.reshape(
        grad_weight, [batch, in_channel, out_channel, padded_weight_grad_h, padded_weight_grad_w])
    grad_weight = topi.sum(grad_weight, axis=0)
    grad_weight = topi.transpose(grad_weight, [1, 0, 2, 3])

    if padded_weight_grad_h > filter_h or padded_weight_grad_w > filter_w:
        grad_weight = topi.strided_slice(grad_weight, begin=[0, 0, 0, 0],
            end=[out_channel, in_channel, filter_h, filter_w])
        return [data, grad_output, grad_weight]

    return [data, grad_output, grad_weight]



def search_schedule(args: dict, func_name: str="", file_prefix: str="", num_search_trails: int=1000):
    time_begin = time.time()
    print(args)
    
    N = args["N"]
    CI = args["CI"]
    HI = args["HI"]
    WI = args["WI"]
    CO = args["CO"]
    HO = args["HO"]
    WO = args["WO"]
    ksize = args["ksize"]
    stride = args["stride"]
    padding = args["padding"]
    
    log_file = file_prefix + func_name + ".json"

    target = tvm.target.Target("cuda")
    task = tvm.auto_scheduler.create_task(conv_bwd_weight, (N, CI, HI, WI, CO, HO, WO, ksize, stride, padding, "float32"), target)

    ### search
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=num_search_trails,
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2,
    )
    sch, args = auto_scheduler.auto_schedule(task, tuning_options=tune_option)
    del measure_ctx

    ### load history
    # inp, res = auto_scheduler.load_best(log_file, task.workload_key)
    # sch, args = task.compute_dag.apply_steps_from_state(inp.state)

    # build func
    ctx = tvm.gpu()
    func = tvm.build(sch, args, target, name=func_name)
    # save result
    obj_fname = file_prefix + func_name + ".o"
    ptx_fname = file_prefix + func_name + ".ptx"
    func.save(obj_fname)
    func.imported_modules[0].save(ptx_fname)

    time_end = time.time()
    print("IterTime: ", (time_end - time_begin))

