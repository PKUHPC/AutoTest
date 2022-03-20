import os
import time

import numpy as np
import tvm
from tvm import te, auto_scheduler, topi
from tvm.topi.nn.utils import get_pad_tuple


@auto_scheduler.register_workload
def batch_norm_bwd(N, C, H, W, dtype="float32"):
  dshape = (N, C, H, W)
  oshape = (C, )
  bshape = (1, C, 1, 1)
  sshape = (1, )
  data = te.placeholder(dshape, name="data", dtype=dtype)
  scale = te.placeholder(oshape, name="scale", dtype=dtype)
  saved_mean = te.placeholder(oshape, name="saved_mean", dtype=dtype)
  saved_var = te.placeholder(oshape, name="saved_var", dtype=dtype)
  eps = te.placeholder(sshape, name="eps", dtype=dtype)
  grad_output = te.placeholder(dshape, name="data", dtype=dtype)

  axis = (0, 2, 3)
  num_ele = dshape[0] * dshape[2] * dshape[3]
  frac_num_ele = 1.0 / num_ele
  # compute mean
  mean_sum = topi.sum(data, axis, True)
  mean = topi.multiply(mean_sum, frac_num_ele)
  var_sub = topi.subtract(data, mean)
  var_mul = topi.multiply(var_sub, var_sub)
  var_sum = topi.sum(var_mul, axis, True)
  var = topi.multiply(var_sum, frac_num_ele)
  # compute output_norm
  output_add = topi.add(var, eps)
  output_sqrt = topi.sqrt(output_add)
  output_sub = topi.subtract(data, mean)
  output_norm = topi.divide(output_sub, output_sqrt)
  # compute grad_input
  dx_hat = topi.multiply(grad_output, topi.reshape(scale, bshape))
  grad_input_sum1 = topi.sum(dx_hat * output_norm, axis, True)
  grad_input_sum2 = topi.sum(dx_hat, axis, True)
  grad_input_left = topi.divide(frac_num_ele , output_sqrt)
  grad_input_right1 = topi.subtract(topi.multiply(dx_hat, num_ele), grad_input_sum2)
  grad_input_right2 = topi.multiply(output_norm, grad_input_sum1)
  grad_input = topi.multiply(grad_input_left, topi.subtract(grad_input_right1, grad_input_right2))
  # compute grad_scale and grad_bias
  grad_scale = topi.sum(grad_output * output_norm, axis)
  grad_bias = topi.sum(grad_output, axis)
  
  return [data, scale, saved_mean, saved_var, eps, grad_output, grad_input, grad_scale, grad_bias]


def search_schedule(args: dict, func_name: str="", file_prefix: str="", num_search_trails: int=1000):
    time_begin = time.time()
    print(args)

    N = args["N"]
    C = args["C"]
    H = args["H"]
    W = args["W"]
    
    log_file = file_prefix + func_name + ".json"

    target = tvm.target.Target("cuda")
    task = tvm.auto_scheduler.create_task(batch_norm_bwd, (N, C, H, W, "float32"), target)

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

