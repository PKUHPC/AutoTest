import os
import time

import numpy as np
import tvm
from tvm import te, auto_scheduler, topi
from tvm.topi.nn.utils import get_pad_tuple


@auto_scheduler.register_workload
def conv_bwd_input(N, CI, HI, WI, CO, HO, WO, KSIZE, stride, padding, dtype):
    strides = (stride, stride)
    shape_weight = (CO, CI, KSIZE, KSIZE)
    shape_grad_output = (N, CO, HO, WO)
    # given tensor
    weight = te.placeholder(shape_weight, name="weight", dtype=dtype)
    grad_output = te.placeholder(shape_grad_output, name="grad_output", dtype=dtype)
    # grad_data
    out_h = (HO - 1) * strides[0] - 2 * padding + KSIZE
    out_w = (WO - 1) * strides[1] - 2 * padding + KSIZE
    output_padding = (HI - out_h, WI - out_w)
    grad_data = topi.nn.conv2d_transpose_nchw(grad_output, weight, strides, padding, dtype, output_padding)

    return [weight, grad_output, grad_data]



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
    task = tvm.auto_scheduler.create_task(conv_bwd_input, (N, CI, HI, WI, CO, HO, WO, ksize, stride, padding, "float32"), target)

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

