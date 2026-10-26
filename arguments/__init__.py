#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""  # Path to the source data set
        self._target_path = ""  # Path to the target data set for pose and expression transfer
        self._model_path = ""  # Path to the folder to save trained models
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.bind_to_mesh = False
        self.adaptive_mesh = False
        self.adaptive_split_mode = "1to3"
        self.adaptive_max_level = 4
        self.debug_vis = False
        self.debug_vis_interval = 1000
        self.disable_flame_static_offset = False
        self.not_finetune_flame_params = False
        self.select_camera_id = -1
        self.lod_include_descendants = True
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        # 3D Gaussians
        self.iterations = 800_000  # default total training iterations
        self.position_lr_init = 0.005  # (scaled up according to mean triangle scale)  #0.00016 (original)
        self.position_lr_final = 0.00005  # (scaled up according to mean triangle scale) # 0.0000016 (original)
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 600_000  # 30_000 (original)
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.017  # (scaled up according to mean triangle scale)  # 0.005 (original)
        self.rotation_lr = 0.001
        self.densification_interval = 500  # 100 (original)
        self.opacity_reset_interval = 60_000 # 3000 (original)
        self.densify_from_iter = 20_000
        self.densify_until_iter = 600_000
        # Base gradient threshold for densification with progressive decay until warmup iters
        self.densify_grad_threshold = 0.0002
        self.densify_grad_threshold_base = 0.0002
        self.densify_grad_threshold_decay = 0.9
        # Decay window [warmup_iters, stop_iters], step in step_iters
        self.densify_grad_threshold_warmup_iters = 300_000
        self.densify_grad_threshold_stop_iters = 500_000
        self.densify_grad_threshold_step_iters = 5_000
        self.densify_min_opacity = 0.005  # min opacity threshold used during densify_and_prune
        
        # GaussianAvatars
        self.flame_expr_lr = 1e-3
        self.flame_trans_lr = 1e-6
        self.flame_pose_lr = 1e-5
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_mse = 0.0
        self.lambda_xyz = 5e-3
        self.threshold_xyz = 1.
        self.metric_xyz = False
        self.lambda_scale = 0.2
        self.threshold_scale = 0.6
        self.metric_scale = False
        self.lambda_dynamic_offset = 0.
        self.lambda_laplacian = 0.
        self.lambda_dynamic_offset_std = 0  #1.

        # Progressive adaptive max-level schedule
        # Start cap for MAX_LEVEL, add `progressive_adaptive_max_level_step` every
        # `progressive_adaptive_max_level_period` iterations, capped by model's
        # configured `adaptive_max_level`.
        self.progressive_adaptive_max_level_base = 0
        self.progressive_adaptive_max_level_step = 1
        self.progressive_adaptive_max_level_period = 50_000

        # Visualization interval (save render every N iterations; 0 to disable)
        self.visualize_interval = 5000

        # Extra periodic checkpoint interval (0 to disable)
        self.extra_checkpoint_interval = 1000

        # Render visualization snapshots
        # Save current levels' renders every N iterations (0 to disable)
        self.render_visualize_interval = 5000
        # How many highest levels to save each time
        # <= 0 means save all available levels
        self.render_visualize_top_k = 0
        # Geometric loss weighting across levels: finest=1.0, then * decay per level
        self.level_loss_decay = 0.5
        self.density_score = "signed"
        self.child_initialization = "original"
        self.lambda_perceptual = 0.0
        self.density_budget = 130144
        self.density_budget_start = 50000
        self.density_budget_end = 598000
        self.supervision_mode = "multi"
        self.supervision_ramp_start = 200000
        self.supervision_ramp_end = 250000

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
