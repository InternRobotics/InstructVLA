import os
import sys
parent_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.getcwd())

import numpy as np
import tensorflow as tf

from simpler_env.evaluation.argparse import get_args
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator
# from simpler_env.policies.octo.octo_server_model import OctoServerInference
# from simpler_env.policies.rt1.rt1_model import RT1Inference

# try:
#     from simpler_env.policies.octo.octo_model import OctoInference
# except ImportError as e:
#     print("Octo is not correctly imported.")
#     print(e)


if __name__ == "__main__":
    args = get_args()

    os.environ["DISPLAY"] = ""
    # prevent a single jax process from taking up all the GPU memory
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 0:
        # prevent a single tf process from taking up all the GPU memory
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=args.tf_memory_limit)],
        )

    # policy model creation; update this if you are using a new policy model

    if 'state' not in args.ckpt_path: #"meta" in args.ckpt_path:
        from simpler_env.policies.sim_instructvla import MetaInstructVLAInference
        assert args.ckpt_path is not None
        model = MetaInstructVLAInference(
            saved_model_path=args.ckpt_path,
            policy_setup=args.policy_setup,
            action_scale=args.action_scale,
        )
    elif "state" in args.ckpt_path:
        from simpler_env.policies.sim_instructvla import MetaStateInstructVLAInference
        model = MetaStateInstructVLAInference(
            saved_model_path=args.ckpt_path,
            policy_setup=args.policy_setup,
            action_scale=args.action_scale,
        )
    else:
        raise NotImplementedError()
    args.logging_dir = os.path.join(os.path.dirname(os.path.dirname(args.ckpt_path)), "results"+"_"+os.path.splitext(os.path.basename(args.ckpt_path))[0])
    # run real-to-sim evaluation
    success_arr = maniskill2_evaluator(model, args)
    print(args)
    print(" " * 10, "Average success", np.mean(success_arr))
