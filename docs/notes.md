```bash
#[ aurora_frameworks-2025.2.0](󱙝 nanochat-aurora_frameworks-2025.2.0)
#[/t/d/f/p/s/nanochat][🌱 saforem2/ezpz][!?] [📦 v0.1.0][󰔛  13s]
#[11/25/25 @ 13:22:33][x1921c0s1b0n0]
; MODEL_TAG=d10 mpiexec -np 16 -n 8 --hostfile $PBS_NODEFILE python3 -m scripts.base_eval
2025-11-25 13:22:58,902 - nanochat.common - INFO - Autodetected device type: xpu
2025-11-25 13:22:58,903 - nanochat.common - INFO - Autodetected device type: xpu
2025-11-25 13:22:58,904 - nanochat.common - INFO - Autodetected device type: xpu
2025-11-25 13:22:58,905 - nanochat.common - INFO - Autodetected device type: xpu
2025-11-25 13:22:58,913 - nanochat.common - INFO - Autodetected device type: xpu
2025-11-25 13:22:58,914 - nanochat.common - INFO - Autodetected device type: xpu
2025-11-25 13:22:58,920 - nanochat.common - INFO - Autodetected device type: xpu
2025-11-25 13:22:58,926 - nanochat.common - INFO - Autodetected device type: xpu
2025-11-25 13:23:01,271 - numexpr.utils - INFO - Note: detected 208 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
2025-11-25 13:23:01,271 - numexpr.utils - INFO - Note: NumExpr detected 208 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
2025-11-25 13:23:01,271 - numexpr.utils - INFO - NumExpr defaulting to 16 threads.
2025-11-25 13:23:01,275 - numexpr.utils - INFO - Note: detected 208 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
2025-11-25 13:23:01,276 - numexpr.utils - INFO - Note: NumExpr detected 208 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
2025-11-25 13:23:01,276 - numexpr.utils - INFO - NumExpr defaulting to 16 threads.
2025-11-25 13:23:01,288 - numexpr.utils - INFO - Note: detected 208 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
2025-11-25 13:23:01,288 - numexpr.utils - INFO - Note: NumExpr detected 208 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
2025-11-25 13:23:01,288 - numexpr.utils - INFO - NumExpr defaulting to 16 threads.
2025-11-25 13:23:01,302 - numexpr.utils - INFO - Note: detected 208 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
2025-11-25 13:23:01,302 - numexpr.utils - INFO - Note: NumExpr detected 208 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
2025-11-25 13:23:01,302 - numexpr.utils - INFO - NumExpr defaulting to 16 threads.
2025-11-25 13:23:01,323 - numexpr.utils - INFO - Note: detected 208 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
2025-11-25 13:23:01,323 - numexpr.utils - INFO - Note: NumExpr detected 208 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
2025-11-25 13:23:01,323 - numexpr.utils - INFO - NumExpr defaulting to 16 threads.
2025-11-25 13:23:01,331 - numexpr.utils - INFO - Note: detected 208 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
2025-11-25 13:23:01,331 - numexpr.utils - INFO - Note: NumExpr detected 208 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
2025-11-25 13:23:01,331 - numexpr.utils - INFO - NumExpr defaulting to 16 threads.
2025-11-25 13:23:01,333 - numexpr.utils - INFO - Note: detected 208 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
2025-11-25 13:23:01,334 - numexpr.utils - INFO - Note: NumExpr detected 208 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
2025-11-25 13:23:01,334 - numexpr.utils - INFO - NumExpr defaulting to 16 threads.
2025-11-25 13:23:01,344 - numexpr.utils - INFO - Note: detected 208 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
2025-11-25 13:23:01,344 - numexpr.utils - INFO - Note: NumExpr detected 208 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
2025-11-25 13:23:01,344 - numexpr.utils - INFO - NumExpr defaulting to 16 threads.
[2025-11-25 13:23:05,340927][I][ezpz/dist:1278:setup_torch_distributed] Using fw='ddp' with torch_{device,backend}= {xpu, xccl}
[2025-11-25 13:23:05,343664][I][ezpz/dist:1137:setup_torch_DDP] Caught MASTER_PORT=35423 from environment!
[2025-11-25 13:23:05,344357][I][ezpz/dist:1153:setup_torch_DDP] Using torch.distributed.init_process_group with
- master_addr='x1921c0s1b0n0'
- master_port='35423'
- world_size=8
- rank=0
- local_rank=0
- timeout=datetime.timedelta(seconds=3600)
- backend='xccl'
[2025-11-25 13:23:05,345786][I][ezpz/dist:840:init_process_group] Calling torch.distributed.init_process_group_with: rank=0 world_size=8 backend=xccl
[2025-11-25 13:23:05,882066][I][ezpz/pbs:179:get_pbs_launch_cmd] ✅ Using [24/24] GPUs [2 hosts] x [12 GPU/host]
[2025-11-25 13:23:05,883211][I][ezpz/dist:521:print_dist_setup] [device='xpu'][rank=0/7][local_rank=0/11][node=0/0]
[2025-11-25 13:23:05,883802][W][ezpz/dist:527:print_dist_setup] Using [8 / 24] available "xpu" devices !!
[2025-11-25 13:23:05,884285][C][ezpz/dist:531:print_dist_setup] num_nodes_from_hostfile = [num_nodes_from_hostfile=2]vs.[wsa=8 // gpus_per_node=12] = 1¯\_(ツ)_/¯ ??
[2025-11-25 13:23:05,885002][I][ezpz/dist:1506:setup_torch] Using device='xpu' with backend='xccl' + 'xccl' for distributed training.
[2025-11-25 13:23:05,885486][I][ezpz/dist:1553:setup_torch] ['x1921c0s1b0n0'][0/7]
[2025-11-25 13:23:05,885010][I][ezpz/dist:1553:setup_torch] ['x1921c0s1b0n0'][4/7]
[2025-11-25 13:23:05,885011][I][ezpz/dist:1553:setup_torch] ['x1921c0s1b0n0'][2/7]
[2025-11-25 13:23:05,885010][I][ezpz/dist:1553:setup_torch] ['x1921c0s1b0n0'][6/7]
[2025-11-25 13:23:05,887887][W][ezpz/dist:707:get_torch_device_type] device_type: xpu passed to ezpz.dist.get_torch_device_type
[2025-11-25 13:23:05,887898][W][ezpz/dist:707:get_torch_device_type] device_type: xpu passed to ezpz.dist.get_torch_device_type
[2025-11-25 13:23:05,887897][W][ezpz/dist:707:get_torch_device_type] device_type: xpu passed to ezpz.dist.get_torch_device_type
[2025-11-25 13:23:05,887899][W][ezpz/dist:707:get_torch_device_type] device_type: xpu passed to ezpz.dist.get_torch_device_type
[2025-11-25 13:23:05,885007][I][ezpz/dist:1553:setup_torch] ['x1921c0s2b0n0'][1/7]
[2025-11-25 13:23:05,888342][I][nanochat/common:216:compute_init] Distributed world size: 8
[2025-11-25 13:23:05,885018][I][ezpz/dist:1553:setup_torch] ['x1921c0s2b0n0'][3/7]
[2025-11-25 13:23:05,885003][I][ezpz/dist:1553:setup_torch] ['x1921c0s2b0n0'][5/7]
[2025-11-25 13:23:05,887893][W][ezpz/dist:707:get_torch_device_type] device_type: xpu passed to ezpz.dist.get_torch_device_type
[2025-11-25 13:23:05,885007][I][ezpz/dist:1553:setup_torch] ['x1921c0s2b0n0'][7/7]
[2025-11-25 13:23:05,887893][W][ezpz/dist:707:get_torch_device_type] device_type: xpu passed to ezpz.dist.get_torch_device_type
[2025-11-25 13:23:05,887892][W][ezpz/dist:707:get_torch_device_type] device_type: xpu passed to ezpz.dist.get_torch_device_type
[2025-11-25 13:23:05,887890][W][ezpz/dist:707:get_torch_device_type] device_type: xpu passed to ezpz.dist.get_torch_device_type
[2025-11-25 13:23:05,889147][I][nanochat/checkpoint_manager:23:log0] Loading model from /home/foremans/.cache/nanochat/base_checkpoints/d10 with step 5075
[2025-11-25 13:23:06,170489][I][nanochat/checkpoint_manager:23:log0] Building model with config: {'sequence_len': 2048, 'vocab_size': 65536, 'n_layer': 10, 'n_head': 5, 'n_kv_head': 5, 'n_embd': 640}
/opt/aurora/25.190.0/frameworks/aurora_frameworks-2025.2.0/lib/python3.10/site-packages/torch/distributed/distributed_c10d.py:4807: UserWarning: No device idis provided via `init_process_group` or `barrier `. Using the current device set by the user.
  warnings.warn(  # warn only once
/opt/aurora/25.190.0/frameworks/aurora_frameworks-2025.2.0/lib/python3.10/site-packages/torch/distributed/distributed_c10d.py:4807: UserWarning: No device idis provided via `init_process_group` or `barrier `. Using the current device set by the user.
  warnings.warn(  # warn only once
/opt/aurora/25.190.0/frameworks/aurora_frameworks-2025.2.0/lib/python3.10/site-packages/torch/distributed/distributed_c10d.py:4807: UserWarning: No device idis provided via `init_process_group` or `barrier `. Using the current device set by the user.
  warnings.warn(  # warn only once
/opt/aurora/25.190.0/frameworks/aurora_frameworks-2025.2.0/lib/python3.10/site-packages/torch/distributed/distributed_c10d.py:4807: UserWarning: No device idis provided via `init_process_group` or `barrier `. Using the current device set by the user.
  warnings.warn(  # warn only once
/opt/aurora/25.190.0/frameworks/aurora_frameworks-2025.2.0/lib/python3.10/site-packages/torch/distributed/distributed_c10d.py:4807: UserWarning: No device idis provided via `init_process_group` or `barrier `. Using the current device set by the user.
  warnings.warn(  # warn only once
2025:11:25-13:24:16:(185590) |CCL_WARN| value of CCL_OP_SYNC changed to be 1 (default:0)
2025:11:25-13:24:16:(185590) |CCL_WARN| value of CCL_PROCESS_LAUNCHER changed to be pmix (default:hydra)
/opt/aurora/25.190.0/frameworks/aurora_frameworks-2025.2.0/lib/python3.10/site-packages/torch/distributed/distributed_c10d.py:4807: UserWarning: No device idis provided via `init_process_group` or `barrier `. Using the current device set by the user.
  warnings.warn(  # warn only once
/opt/aurora/25.190.0/frameworks/aurora_frameworks-2025.2.0/lib/python3.10/site-packages/torch/distributed/distributed_c10d.py:4807: UserWarning: No device idis provided via `init_process_group` or `barrier `. Using the current device set by the user.
  warnings.warn(  # warn only once
/opt/aurora/25.190.0/frameworks/aurora_frameworks-2025.2.0/lib/python3.10/site-packages/torch/distributed/distributed_c10d.py:4807: UserWarning: No device idis provided via `init_process_group` or `barrier `. Using the current device set by the user.
  warnings.warn(  # warn only once
Evaluating: hellaswag_zeroshot (0-shot, type: multiple_choice)... accuracy: 0.3040 | centered: 0.0720 | time: 71.64s
Evaluating: jeopardy (10-shot, type: language_modeling)... accuracy: 0.0099 | centered: 0.0099 | time: 39.59s
Evaluating: bigbench_qa_wikidata (10-shot, type: language_modeling)... accuracy: 0.3204 | centered: 0.3204 | time: 60.92s
Evaluating: arc_easy (10-shot, type: multiple_choice)... accuracy: 0.5105 | centered: 0.3474 | time: 85.57s
Evaluating: arc_challenge (10-shot, type: multiple_choice)... accuracy: 0.2585 | centered: 0.0114 | time: 27.28s
Evaluating: copa (0-shot, type: multiple_choice)... accuracy: 0.6000 | centered: 0.2000 | time: 4.55s
Evaluating: commonsense_qa (10-shot, type: multiple_choice)... accuracy: 0.2850 | centered: 0.1063 | time: 28.08s
Evaluating: piqa (10-shot, type: multiple_choice)... accuracy: 0.6202 | centered: 0.2405 | time: 89.82s
Evaluating: openbook_qa (0-shot, type: multiple_choice)... accuracy: 0.2960 | centered: 0.0613 | time: 9.29s
Evaluating: lambada_openai (0-shot, type: language_modeling)... accuracy: 0.2521 | centered: 0.2521 | time: 31.40s
Evaluating: hellaswag (10-shot, type: multiple_choice)... accuracy: 0.3046 | centered: 0.0728 | time: 241.04s
Evaluating: winograd (0-shot, type: schema)... accuracy: 0.5531 | centered: 0.1062 | time: 7.15s
Evaluating: winogrande (0-shot, type: schema)... accuracy: 0.5162 | centered: 0.0324 | time: 5.49s
Evaluating: bigbench_dyck_languages (10-shot, type: language_modeling)... accuracy: 0.1430 | centered: 0.1430 | time: 55.70s
Evaluating: agi_eval_lsat_ar (3-shot, type: multiple_choice)... accuracy: 0.2696 | centered: 0.0870 | time: 2.37s
Evaluating: bigbench_cs_algorithms (10-shot, type: language_modeling)... accuracy: 0.3689 | centered: 0.3689 | time: 39.94s
Evaluating: bigbench_operators (10-shot, type: language_modeling)... accuracy: 0.1095 | centered: 0.1095 | time: 11.98s
Evaluating: bigbench_repeat_copy_logic (10-shot, type: language_modeling)... accuracy: 0.0000 | centered: 0.0000 | time: 2.06s
Evaluating: squad (10-shot, type: language_modeling)... accuracy: 0.0355 | centered: 0.0355 | time: 387.37s
Evaluating: coqa (0-shot, type: language_modeling)... accuracy: 0.0785 | centered: 0.0785 | time: 181.78s
Evaluating: boolq (10-shot, type: multiple_choice)... accuracy: 0.5428 | centered: -0.2031 | time: 180.06s
Evaluating: bigbench_language_identification (10-shot, type: multiple_choice)... accuracy: 0.2523 | centered: 0.1774 | time: 440.23s
================================================================================
Model: base_model (step 5075)
================================================================================
Task                               , Accuracy  , Centered
hellaswag_zeroshot                 , 0.304023  , 0.072031
jeopardy                           , 0.009920  , 0.009920
bigbench_qa_wikidata               , 0.320407  , 0.320407
arc_easy                           , 0.510522  , 0.347363
arc_challenge                      , 0.258532  , 0.011377
copa                               , 0.600000  , 0.200000
commonsense_qa                     , 0.285012  , 0.106265
piqa                               , 0.620239  , 0.240479
openbook_qa                        , 0.296000  , 0.061333
lambada_openai                     , 0.252086  , 0.252086
hellaswag                          , 0.304621  , 0.072827
winograd                           , 0.553114  , 0.106227
winogrande                         , 0.516180  , 0.032360
bigbench_dyck_languages            , 0.143000  , 0.143000
agi_eval_lsat_ar                   , 0.269565  , 0.086956
bigbench_cs_algorithms             , 0.368939  , 0.368939
bigbench_operators                 , 0.109524  , 0.109524
bigbench_repeat_copy_logic         , 0.000000  , 0.000000
squad                              , 0.035478  , 0.035478
coqa                               , 0.078542  , 0.078542
boolq                              , 0.542813  , -0.203123
bigbench_language_identification   , 0.252300  , 0.177448
CORE                               ,           , 0.119520

took: 33m 36s
```
