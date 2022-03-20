import os
import time

import numpy as np
import tvm
from tvm import te, auto_scheduler, topi


@auto_scheduler.register_workload
def pool_avg_fwd(N, C, H, W, KSIZE, stride, padding, dtype):
    data = te.placeholder((N, C, H, W), name="data", dtype=dtype)
    out = topi.nn.pool(data, [KSIZE, KSIZE], [stride, stride], [padding, padding, padding, padding], 
            "avg", ceil_mode=False, layout="NCHW", count_include_pad=True)
    return [data, out]

def search_schedule(args: dict, func_name: str="", file_prefix: str="", num_search_trails: int=1000):
    time_begin = time.time()
    print(args)
    
    N = args["N"]
    C = args["C"]
    H = args["H"]
    W = args["W"]
    ksize = args["ksize"]
    stride = args["stride"]
    padding = args["padding"]

    log_file = file_prefix + func_name + ".json"

    target = tvm.target.Target("cuda")
    task = tvm.auto_scheduler.create_task(pool_avg_fwd, (N, C, H, W, ksize, stride, padding, "float32"), target)

    ### search
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=num_search_trails,
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2,
    )
    sch, args = auto_scheduler.auto_schedule(task, tuning_options=tune_option)
    print("deleting measure_ctx...")
    del measure_ctx

    ### load history
    # inp, res = auto_scheduler.load_best(log_file, task.workload_key)
    # sch, args = task.compute_dag.apply_steps_from_state(inp.state)

    # build func
    ctx = tvm.gpu()
    print("building func")
    func = tvm.build(sch, args, target, name=func_name)
    print("func built")
    # save result
    obj_fname = file_prefix + func_name + ".o"
    ptx_fname = file_prefix + func_name + ".ptx"
    print("saving obj_fname")
    func.save(obj_fname)
    print("saving ptx_fname")
    func.imported_modules[0].save(ptx_fname)

    time_end = time.time()
    print("IterTime: ", (time_end - time_begin))