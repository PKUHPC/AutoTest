import os
import time

import numpy as np
import tvm
from tvm import te, auto_scheduler, topi

@auto_scheduler.register_workload
def sepconv(N, H, W, CO, CI, KSIZE, stride, dtype):
    groups = CI
    data = te.placeholder((N, CI, H, W), name="data", dtype=dtype)
    weight_depth = te.placeholder((CI, CI // groups, KSIZE, KSIZE), name="weight_depth", dtype=dtype)
    depth_out = topi.nn.group_conv2d_nchw(data, weight_depth, stride, padding=1, dilation=1, groups=groups, out_dtype="float32")
    weight_point = te.placeholder((CO, CI, 1, 1), name="weight_point", dtype=dtype)
    out = topi.nn.conv2d_nchw(depth_out, weight_point, stride=1, padding=0, dilation=1, out_dtype="float32")
    return [data, weight_depth, weight_point, depth_out, out]


def search_schedule(args: dict, func_name: str="", file_prefix: str="", num_search_trails: int=1000):
    time_begin = time.time()
    print(args)
    
    N = args["N"]
    CI = args["CI"]
    H = args["H"]
    W = args["W"]
    CO = args["CO"]
    ksize = args["ksize"]
    stride = args["stride"]
    padding = args["padding"]
    
    log_file = file_prefix + func_name + ".json"

    target = tvm.target.Target("cuda")
    task = tvm.auto_scheduler.create_task(sepconv, (N, H, W, CO, CI, ksize, stride, "float32"), target)

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
