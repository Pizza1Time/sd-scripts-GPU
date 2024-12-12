# training with captions

import argparse
import math
import os
from multiprocessing import Value
from typing import List
import toml

from tqdm import tqdm

import torch
from library.device_utils import init_ipex, clean_memory_on_device

# init_ipex() # Do not need in torch-xla

# from accelerate.utils import set_seed # Not available in torch-xla
from diffusers import DDPMScheduler
from library import sdxl_model_util

import library.train_util as train_util

from library.utils import setup_logging, add_logging_arguments

from torch.utils.data.distributed import DistributedSampler

setup_logging()
import logging

logger = logging.getLogger(__name__)

import library.config_util as config_util
import library.sdxl_train_util as sdxl_train_util
from library.config_util import (
    ConfigSanitizer,
    BlueprintGenerator,
)
import library.custom_train_functions as custom_train_functions
from library.custom_train_functions import (
    apply_snr_weight,
    prepare_scheduler_for_custom_training,
    scale_v_prediction_loss_like_noise_prediction,
    add_v_prediction_like_loss,
    apply_debiased_estimation,
    apply_masked_loss,
)
from library.sdxl_original_unet import SdxlUNet2DConditionModel

# from accelerate.utils import DummyOptim, DummyScheduler  # Import Dummy classes # Not available in torch-xla

# Add necessary imports for torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.utils as xu
import numpy as np

UNET_NUM_BLOCKS_FOR_BLOCK_LR = 23

def get_block_params_to_optimize(unet: SdxlUNet2DConditionModel, block_lrs: List[float]) -> List[dict]:
    block_params = [[] for _ in range(len(block_lrs))]

    for i, (name, param) in enumerate(unet.named_parameters()):
        if name.startswith("time_embed.") or name.startswith("label_emb."):
            block_index = 0  # 0
        elif name.startswith("input_blocks."):  # 1-9
            block_index = 1 + int(name.split(".")[1])
        elif name.startswith("middle_block."):  # 10-12
            block_index = 10 + int(name.split(".")[1])
        elif name.startswith("output_blocks."):  # 13-21
            block_index = 13 + int(name.split(".")[1])
        elif name.startswith("out."):  # 22
            block_index = 22
        else:
            raise ValueError(f"unexpected parameter name: {name}")

        block_params[block_index].append(param)

    params_to_optimize = []
    for i, params in enumerate(block_params):
        if block_lrs[i] == 0:  # 0のときは学習しない do not optimize when lr is 0
            continue
        params_to_optimize.append({"params": params, "lr": block_lrs[i]})

    return params_to_optimize

def append_block_lr_to_logs(block_lrs, logs, lr_scheduler, optimizer_type):
    names = []
    block_index = 0
    while block_index < UNET_NUM_BLOCKS_FOR_BLOCK_LR + 2:
        if block_index < UNET_NUM_BLOCKS_FOR_BLOCK_LR:
            if block_lrs[block_index] == 0:
                block_index += 1
                continue
            names.append(f"block{block_index}")
        elif block_index == UNET_NUM_BLOCKS_FOR_BLOCK_LR:
            names.append("text_encoder1")
        elif block_index == UNET_NUM_BLOCKS_FOR_BLOCK_LR + 1:
            names.append("text_encoder2")

        block_index += 1

    train_util.append_lr_to_logs_with_names(logs, lr_scheduler, optimizer_type, names)

# def train(args):
def train(index, args):
    # train_util.verify_training_args(args) # Do not need to verify here
    train_util.prepare_dataset_args(args, True)
    sdxl_train_util.verify_sdxl_training_args(args)
    # deepspeed_utils.prepare_deepspeed_args(args) # Remove deepspeed part
    setup_logging(args, reset=True)

    # These lines are no longer necessary when using torch_xla
    # assert (
    #     not args.weighted_captions
    # ), "weighted_captions is not supported currently / weighted_captionsは現在サポートされていません"
    # assert (
    #     not args.train_text_encoder or not args.cache_text_encoder_outputs
    # ), "cache_text_encoder_outputs is not supported when training text encoder / text encoderを学習するときはcache_text_encoder_outputsはサポートされていません"

    if args.block_lr:
        block_lrs = [float(lr) for lr in args.block_lr.split(",")]
        assert (
            len(block_lrs) == UNET_NUM_BLOCKS_FOR_BLOCK_LR
        ), f"block_lr must have {UNET_NUM_BLOCKS_FOR_BLOCK_LR} values / block_lrは{UNET_NUM_BLOCKS_FOR_BLOCK_LR}個の値を指定してください"
    else:
        block_lrs = None

    cache_latents = args.cache_latents
    use_dreambooth_method = args.in_json is None

    if args.seed is not None:
        # set_seed(args.seed)  # 乱数系列を初期化する # Replace with torch_xla seed
        xm.set_rng_state(args.seed)

    tokenizer1, tokenizer2 = sdxl_train_util.load_tokenizers(args)

    # データセットを準備する
    if args.dataset_class is None:
        blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, args.masked_loss, True))
        if args.dataset_config is not None:
            logger.info(f"Load dataset config from {args.dataset_config}")
            user_config = config_util.load_user_config(args.dataset_config)
            ignored = ["train_data_dir", "in_json"]
            if any(getattr(args, attr) is not None for attr in ignored):
                logger.warning(
                    "ignore following options because config file is found: {0} / 設定ファイルが利用されるため以下のオプションは無視されます: {0}".format(
                        ", ".join(ignored)
                    )
                )
        else:
            if use_dreambooth_method:
                logger.info("Using DreamBooth method.")
                user_config = {
                    "datasets": [
                        {
                            "subsets": config_util.generate_dreambooth_subsets_config_by_subdirs(
                                args.train_data_dir, args.reg_data_dir
                            )
                        }
                    ]
                }
            else:
                logger.info("Training with captions.")
                user_config = {
                    "datasets": [
                        {
                            "subsets": [
                                {
                                    "image_dir": args.train_data_dir,
                                    "metadata_file": args.in_json,
                                }
                            ]
                        }
                    ]
                }

        blueprint = blueprint_generator.generate(user_config, args, tokenizer=[tokenizer1, tokenizer2])
        train_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    else:
        train_dataset_group = train_util.load_arbitrary_dataset(args, [tokenizer1, tokenizer2])

    current_epoch = Value("i", 0)
    current_step = Value("i", 0)
    ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
    collator = train_util.collator_class(current_epoch, current_step, ds_for_collator)

    train_dataset_group.verify_bucket_reso_steps(32)

    if args.debug_dataset:
        train_util.debug_dataset(train_dataset_group, True)
        return
    if len(train_dataset_group) == 0:
        logger.error(
            "No data found. Please verify the metadata file and train_data_dir option. / 画像がありません。メタデータおよびtrain_data_dirオプションを確認してください。"
        )
        return

    if cache_latents:
        assert (
            train_dataset_group.is_latent_cacheable()
        ), "when caching latents, either color_aug or random_crop cannot be used / latentをキャッシュするときはcolor_augとrandom_cropは使えません"

    if args.cache_text_encoder_outputs:
        assert (
            train_dataset_group.is_text_encoder_output_cacheable()
        ), "when caching text encoder output, either caption_dropout_rate, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used / text encoderの出力をキャッシュするときはcaption_dropout_rate, shuffle_caption, token_warmup_step, caption_tag_dropout_rateは使えません"

    # acceleratorを準備する
    # logger.info("prepare accelerator")
    # accelerator = train_util.prepare_accelerator(args)
    
    # Log device information
    device = xm.xla_device()
    logger.info(f"Main device: {device}, device ordinal: {xm.get_ordinal()}")

    # Get local device ordinal and all devices.
    local_ordinal = xm.get_local_ordinal()
    all_devices = xm.get_xla_supported_devices()

    # Print all available devices for this process.
    logger.info(f"All available devices for this process: {all_devices}")

    # mixed precisionに対応した型を用意しておき適宜castする
    # weight_dtype, save_dtype = train_util.prepare_dtype(args) # Replace with torch_xla dtype
    # vae_dtype = torch.float32 if args.no_half_vae else weight_dtype
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    vae_dtype = torch.float32 if args.no_half_vae else weight_dtype

    if args.mixed_precision == "no":
        save_dtype = torch.float32
    else:
        save_dtype = weight_dtype

    # モデルを読み込む
    (
        load_stable_diffusion_format,
        text_encoder1,
        text_encoder2,
        vae,
        unet,
        logit_scale,
        ckpt_info,
    ) = sdxl_train_util.load_target_model(args, None , "sdxl", weight_dtype) # Remove accelerator
    # logit_scale = logit_scale.to(accelerator.device, dtype=weight_dtype)
    logit_scale = logit_scale.to(device, dtype=weight_dtype)

    # verify load/save model formats
    if load_stable_diffusion_format:
        src_stable_diffusion_ckpt = args.pretrained_model_name_or_path
        src_diffusers_model_path = None
    else:
        src_stable_diffusion_ckpt = None
        src_diffusers_model_path = args.pretrained_model_name_or_path

    if args.save_model_as is None:
        save_stable_diffusion_format = load_stable_diffusion_format
        use_safetensors = args.use_safetensors
    else:
        save_stable_diffusion_format = args.save_model_as.lower() == "ckpt" or args.save_model_as.lower() == "safetensors"
        use_safetensors = args.use_safetensors or ("safetensors" in args.save_model_as.lower())
        # assert save_stable_diffusion_format, "save_model_as must be ckpt or safetensors / save_model_asはckptかsafetensorsである必要があります"

    # Diffusers版のxformers使用フラグを設定する関数
    def set_diffusers_xformers_flag(model, valid):
        def fn_recursive_set_mem_eff(module: torch.nn.Module):
            if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                module.set_use_memory_efficient_attention_xformers(valid)

            for child in module.children():
                fn_recursive_set_mem_eff(child)

        fn_recursive_set_mem_eff(model)

    # モデルに xformers とか memory efficient attention を組み込む
    if args.diffusers_xformers:
        # もうU-Netを独自にしたので動かないけどVAEのxformersは動くはず
        # accelerator.print("Use xformers by Diffusers")
        logger.info("Use xformers by Diffusers")
        # set_diffusers_xformers_flag(unet, True)
        set_diffusers_xformers_flag(vae, True)
    else:
        # Windows版のxformersはfloatで学習できなかったりするのでxformersを使わない設定も可能にしておく必要がある
        # accelerator.print("Disable Diffusers' xformers")
        logger.info("Disable Diffusers' xformers")
        train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers, args.sdpa)
        #if torch.__version__ >= "2.0.0":  # PyTorch 2.0.0 以上対応のxformersなら以下が使える
        vae.set_use_memory_efficient_attention_xformers(args.xformers) # This version should be compatible with torch-xla

    # 学習を準備する
    if cache_latents:
        # vae.to(accelerator.device, dtype=vae_dtype)
        vae.to(device, dtype=vae_dtype)
        vae.requires_grad_(False)
        vae.eval()
        with torch.no_grad():
            train_dataset_group.cache_latents(vae, args.vae_batch_size, args.cache_latents_to_disk, True) # Replace accelerator.is_main_process
        vae.to("cpu")
        # clean_memory_on_device(accelerator.device)
        #xm.clear_cache()

        # accelerator.wait_for_everyone()
        xm.rendezvous("wait_after_cache_latents") # wait for all processes

    # 学習を準備する：モデルを適切な状態にする
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    train_unet = args.learning_rate > 0
    train_text_encoder1 = False
    train_text_encoder2 = False

    if args.train_text_encoder:
        # TODO each option for two text encoders?
        # accelerator.print("enable text encoder training")
        logger.info("enable text encoder training")
        if args.gradient_checkpointing:
            text_encoder1.gradient_checkpointing_enable()
            text_encoder2.gradient_checkpointing_enable()
        lr_te1 = args.learning_rate_te1 if args.learning_rate_te1 is not None else args.learning_rate  # 0 means not train
        lr_te2 = args.learning_rate_te2 if args.learning_rate_te2 is not None else args.learning_rate  # 0 means not train
        train_text_encoder1 = lr_te1 > 0
        train_text_encoder2 = lr_te2 > 0

        # caching one text encoder output is not supported
        if not train_text_encoder1:
            text_encoder1.to(weight_dtype)
        if not train_text_encoder2:
            text_encoder2.to(weight_dtype)
        text_encoder1.requires_grad_(train_text_encoder1)
        text_encoder2.requires_grad_(train_text_encoder2)
        text_encoder1.train(train_text_encoder1)
        text_encoder2.train(train_text_encoder2)
    else:
        text_encoder1.to(weight_dtype)
        text_encoder2.to(weight_dtype)
        text_encoder1.requires_grad_(False)
        text_encoder2.requires_grad_(False)
        text_encoder1.eval()
        text_encoder2.eval()

        # TextEncoderの出力をキャッシュする
        if args.cache_text_encoder_outputs:
            # Text Encodes are eval and no grad
            with torch.no_grad():
                train_dataset_group.cache_text_encoder_outputs(
                    (tokenizer1, tokenizer2),
                    (text_encoder1, text_encoder2),
                    # accelerator.device,
                    device,
                    None,
                    args.cache_text_encoder_outputs_to_disk,
                    # accelerator.is_main_process,
                    True,
                )
            # accelerator.wait_for_everyone()
            xm.rendezvous("wait_after_cache_text_encoder_outputs")

    if not cache_latents:
        vae.requires_grad_(False)
        vae.eval()
        # vae.to(accelerator.device, dtype=vae_dtype)
        vae.to(device, dtype=vae_dtype)

    unet.requires_grad_(train_unet)
    if not train_unet:
        # unet.to(accelerator.device, dtype=weight_dtype)  # because of unet is not prepared
        unet.to(device, dtype=weight_dtype)

    training_models = []
    params_to_optimize = []
    if train_unet:
        training_models.append(unet)
        if block_lrs is None:
            params_to_optimize.append({"params": list(unet.parameters()), "lr": args.learning_rate})
        else:
            params_to_optimize.extend(get_block_params_to_optimize(unet, block_lrs))

    if train_text_encoder1:
        training_models.append(text_encoder1)
        params_to_optimize.append({"params": list(text_encoder1.parameters()), "lr": args.learning_rate_te1 or args.learning_rate})
    if train_text_encoder2:
        training_models.append(text_encoder2)
        params_to_optimize.append({"params": list(text_encoder2.parameters()), "lr": args.learning_rate_te2 or args.learning_rate})

    # calculate number of trainable parameters
    n_params = 0
    for params in params_to_optimize:
        for p in params["params"]:
            n_params += p.numel()

    # accelerator.print(f"train unet: {train_unet}, text_encoder1: {train_text_encoder1}, text_encoder2: {train_text_encoder2}")
    logger.info(f"train unet: {train_unet}, text_encoder1: {train_text_encoder1}, text_encoder2: {train_text_encoder2}")
    # accelerator.print(f"number of models: {len(training_models)}")
    logger.info(f"number of models: {len(training_models)}")
    # accelerator.print(f"number of trainable parameters: {n_params}")
    logger.info(f"number of trainable parameters: {n_params}")

    # 学習に必要なクラスを準備する
    # accelerator.print("prepare optimizer, data loader etc.")
    logger.info("prepare optimizer, data loader etc.")
    #_, _, optimizer = train_util.get_optimizer(args, trainable_params=params_to_optimize)
    optimizer_type, optimizer_kwargs, optimizer = train_util.get_optimizer(args, trainable_params=params_to_optimize)

    # dataloaderを準備する
    # DataLoaderのプロセス数：0 は persistent_workers が使えないので注意
    n_workers = min(args.max_data_loader_n_workers, os.cpu_count())  # cpu_count or max_data_loader_n_workers
    # train_dataloader = torch.utils.data.DataLoader(
    #     train_dataset_group,
    #     batch_size=1,
    #     shuffle=True,
    #     collate_fn=collator,
    #     num_workers=n_workers,
    #     persistent_workers=args.persistent_data_loader_workers,
    # )
    # Convert to XLA DataLoader
    if args.max_train_epochs is not None:
        num_samples = len(train_dataset_group)
        num_update_steps_per_epoch = math.ceil(
            num_samples / args.gradient_accumulation_steps
        )
        args.max_train_steps = args.max_train_epochs * num_update_steps_per_epoch
        logger.info(
    f"override steps. steps for {args.max_train_epochs} epochs is: {args.max_train_steps}"
        )

    train_sampler = DistributedSampler(
        train_dataset_group,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_group,
        batch_size=args.train_batch_size,
        sampler=train_sampler,
        collate_fn=collator,
        num_workers=args.max_data_loader_n_workers,
        drop_last=True,
        pin_memory=True
    )

    # データセット側にも学習ステップを送信
    train_dataset_group.set_max_train_steps(args.max_train_steps)

    # lr schedulerを用意する
    #lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)
    # if accelerator.state.deepspeed_plugin is not None and "optimizer" in accelerator.state.deepspeed_plugin.deepspeed_config:
    #     # Must use a Dummy Optimizer when config has specified an optimizer. 
    #     optimizer = DummyOptim(optimizer.param_groups) # 
    # if accelerator.state.deepspeed_plugin is not None and "scheduler" in accelerator.state.deepspeed_plugin.deepspeed_config:
    #     # Must use a Dummy Scheduler when config has specified a scheduler.
    #     # Pass num_training_steps to the DummyScheduler so it can calculate lr correctly.
    #     lr_scheduler = DummyScheduler(
    #         optimizer, 
    #         total_num_steps=args.max_train_steps, 
    #         warmup_num_steps=args.lr_warmup_steps if "num_warmup_steps" not in accelerator.state.deepspeed_plugin.deepspeed_config["scheduler"]["params"] else accelerator.state.deepspeed_plugin.deepspeed_config["scheduler"]["params"]["num_warmup_steps"]
    #     )
    # else:
    #     lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)
    lr_scheduler = train_util.get_scheduler_fix(args, optimizer, xm.xrt_world_size())

    # 実験的機能：勾配も含めたfp16/bf16学習を行う　モデル全体をfp16/bf16にする
    if args.full_fp16:
        # assert (
        #     args.mixed_precision == "fp16"
        # ), "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
        # accelerator.print("enable full fp16 training.")
        logger.info("enable full fp16 training.")
        unet.to(weight_dtype)
        text_encoder1.to(weight_dtype)
        text_encoder2.to(weight_dtype)
    elif args.full_bf16:
        # assert (
        #     args.mixed_precision == "bf16"
        # ), "full_bf16 requires mixed precision='bf16' / full_bf16を使う場合はmixed_precision='bf16'を指定してください。"
        # accelerator.print("enable full bf16 training.")
        logger.info("enable full bf16 training.")
        unet.to(weight_dtype)
        text_encoder1.to(weight_dtype)
        text_encoder2.to(weight_dtype)

    # freeze last layer and final_layer_norm in te1 since we use the output of the penultimate layer
    if train_text_encoder1:
        text_encoder1.text_model.encoder.layers[-1].requires_grad_(False)
        text_encoder1.text_model.final_layer_norm.requires_grad_(False)

    if train_unet:
        unet = unet.to(device)
    if train_text_encoder1:
        text_encoder1 = text_encoder1.to(device)
    if train_text_encoder2:
        text_encoder2 = text_encoder2.to(device)

    # optimizer = optimizer.to(device) # Handled when creating the optimizer
    # lr_scheduler = lr_scheduler.to(device) # Handled inside the lr_scheduler

    # Move models to XLA device
    for m in training_models:
        m.to(device)

    # Use Torch-XLA's Parallel Loader
    para_loader = pl.ParallelLoader(train_dataloader, [device])
    train_dataloader = para_loader.per_device_loader(device)

    # TextEncoderの出力をキャッシュするときにはCPUへ移動する
    if args.cache_text_encoder_outputs:
        # move Text Encoders for sampling images. Text Encoder doesn't work on CPU with fp16
        text_encoder1.to("cpu", dtype=torch.float32)
        text_encoder2.to("cpu", dtype=torch.float32)
        # clean_memory_on_device(accelerator.device)
        #xm.clear_cache()
    else:
        # make sure Text Encoders are on GPU
        # text_encoder1.to(accelerator.device)
        # text_encoder2.to(accelerator.device)
        text_encoder1.to(device)
        text_encoder2.to(device)

    # 実験的機能：勾配も含めたfp16学習を行う　PyTorchにパッチを当ててfp16でのgrad scaleを有効にする
    # if args.full_fp16:
        # During deepseed training, accelerate not handles fp16/bf16|mixed precision directly via scaler. Let deepspeed engine do.
        # -> But we think it's ok to patch accelerator even if deepspeed is enabled.
    #     train_util.patch_accelerator_for_fp16_training(accelerator) # Do not need this in torch-xla

    # resumeする
    # train_util.resume_from_local_or_hf_if_specified(accelerator, args)
    # train_util.resume_from_local_or_hf_if_specified(None, args) # Remove accelerator # TODO need to verify how to resume in torch-xla
    
    #global_step = 0 
    
    if args.resume is not None:
        logger.info(f"resume training from {args.resume}")
        resume_path = args.resume
        if train_util.is_safetensors(args.resume):
            resume_extension = ".safetensors"
            state_dict = train_util.load_file_from_safetensors(args.resume)
        else:
            #global_step = 0
            resume_extension = ".ckpt"
            state_dict = torch.load(args.resume)

        # load state_dict
        if train_unet and "unet" in state_dict:
            logger.info(f"loading U-Net from {args.resume}")
            unet.load_state_dict(state_dict["unet"])
        if train_text_encoder1 and "text_encoder1" in state_dict:
            logger.info(f"loading Text Encoder 1 from {args.resume}")
            text_encoder1.load_state_dict(state_dict["text_encoder1"])
        if train_text_encoder2 and "text_encoder2" in state_dict:
            logger.info(f"loading Text Encoder 2 from {args.resume}")
            text_encoder2.load_state_dict(state_dict["text_encoder2"])

        global_step = 0
        if "global_step" in state_dict:
            global_step = state_dict["global_step"]
            logger.info(f"resumed from step {global_step}")
        
        if "epoch" in state_dict and args.resume_start_epoch is None:
            args.resume_start_epoch = state_dict["epoch"] + 1
            logger.info(f"resumed from epoch {args.resume_start_epoch - 1}")

        if args.resume_start_epoch is not None:
            # if resuming from a later epoch, we need to fast forward the dataloader
            for _ in range(args.resume_start_epoch * len(train_dataloader)):
                next(train_dataloader)
            logger.info(f"fast forward to epoch {args.resume_start_epoch}")

        if "optimizer" in state_dict and not args.resume_no_sync_optimizer_state:
            logger.info(f"loading optimizer state from {args.resume}") # This line was incomplete         
                
        if "lr_scheduler" in state_dict:
            lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
            logger.info(f"loaded scheduler state from {args.resume}")
            
            try:
                optimizer.load_state_dict(state_dict["optimizer"])
                logger.info(f"loaded optimizer state from {args.resume}")
            except Exception as e:
                logger.warning(
                    f"Could not load optimizer state from {args.resume} due to {e}.\n"
                    "Please make sure that you are using the same optimizer and arguments as before.\n"
                    "Skipping optimizer state loading.\n"
                )
        else:
            if "optimizer" in state_dict:
                logger.warning(
                    f"State from {args.resume} contains optimizer state, but --resume_no_sync_optimizer_state is specified.\n"
                    "Skipping optimizer state loading.\n"
                )
            else:
                logger.info(f"State from {args.resume} does not contain optimizer state. Skipping optimizer state loading.\n")

    # epoch数を計算する
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0):
        args.save_every_n_epochs = math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1

    # 学習する
    # total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    # accelerator.print("running training / 学習開始")
    logger.info("running training / 学習開始")
    # accelerator.print(f"  num examples / サンプル数: {train_dataset_group.num_train_images}")
    logger.info(f"  num examples / サンプル数: {train_dataset_group.num_train_images}")
    # accelerator.print(f"  num batches per epoch / 1epochのバッチ数: {len(train_dataloader)}")
    logger.info(f"  num batches per epoch / 1epochのバッチ数: {len(train_dataloader)}")
    # accelerator.print(f"  num epochs / epoch数: {num_train_epochs}")
    logger.info(f"  num epochs / epoch数: {num_train_epochs}")
    # accelerator.print(
    logger.info(
        f"  batch size per device / バッチサイズ: {', '.join([str(d.batch_size) for d in train_dataset_group.datasets])}"
    )
    # accelerator.print(
    #     f"  total train batch size (with parallel & distributed & accumulation) / 総バッチサイズ（並列学習、勾配合計含む）: {total_batch_size}"
    # )
    # accelerator.print(f"  gradient accumulation steps / 勾配を合計するステップ数 = {args.gradient_accumulation_steps}")
    logger.info(f"  gradient accumulation steps / 勾配を合計するステップ数 = {args.gradient_accumulation_steps}")
    # accelerator.print(f"  total optimization steps / 学習ステップ数: {args.max_train_steps}")
    logger.info(f"  total optimization steps / 学習ステップ数: {args.max_train_steps}")

    global_step = 0
    if args.resume is not None:
        if "global_step" in state_dict:
            global_step = state_dict["global_step"]
            logger.info(f"resumed from step {global_step}")

    # progress_bar = tqdm(range(args.max_train_steps), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps")
    # progress_bar = tqdm(range(args.max_train_steps), smoothing=0, disable=not xm.is_master_ordinal(), desc="steps")
    if xm.is_master_ordinal():
        progress_bar = tqdm(total=args.max_train_steps, smoothing=0, desc="steps", position=0)
        progress_bar.update(global_step)

    noise_scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False
    )
    prepare_scheduler_for_custom_training(noise_scheduler, device)
    if args.zero_terminal_snr:
        custom_train_functions.fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler)

    # if accelerator.is_main_process:
    if xm.is_master_ordinal():
        init_kwargs = {}
        if args.wandb_run_name:
            init_kwargs["wandb"] = {"name": args.wandb_run_name}
        if args.log_tracker_config is not None:
            init_kwargs = toml.load(args.log_tracker_config)
        # accelerator.init_trackers("finetuning" if args.log_tracker_name is None else args.log_tracker_name, init_kwargs=init_kwargs) # Replace with tensorboard
        if args.logging_dir is not None:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = os.path.join(args.logging_dir, "tensorboard", str(xm.get_ordinal()))
            writer = SummaryWriter(log_dir)

    # For --sample_at_first
    sdxl_train_util.sample_images(
        None, # Removed accelerator
        args,
        0,
        global_step,
        device, # Removed accelerator.device
        vae,
        [tokenizer1, tokenizer2],
        [text_encoder1, text_encoder2],
        unet,
        #Removed prompt_replacement=None
    )

    loss_recorder = train_util.LossRecorder()
    
    if args.resume_start_epoch is None:
        args.resume_start_epoch = 0

    for epoch in range(args.resume_start_epoch, num_train_epochs):
        # accelerator.print(f"\nepoch {epoch+1}/{num_train_epochs}")
        logger.info(f"\nepoch {epoch+1}/{num_train_epochs}")
        current_epoch.value = epoch + 1

        for m in training_models:
            m.train()

        for step, batch in enumerate(train_dataloader):
            current_step.value = global_step
            # with accelerator.accumulate(*training_models):
            with torch.autograd.set_detect_anomaly(args.detect_anomaly) if args.detect_anomaly else xu.no_context():
                for param in training_models[0].parameters():
                    if param.grad is not None:
                        param.grad.detach_()
                        param.grad.zero_()

                if "latents" in batch and batch["latents"] is not None:
                    # latents = batch["latents"].to(accelerator.device).to(dtype=weight_dtype)
                    latents = batch["latents"].to(device, dtype=weight_dtype)
                else:
                    with torch.no_grad():
                        # latentに変換
                        latents = vae.encode(batch["images"].to(device, dtype=vae_dtype)).latent_dist.sample().to(weight_dtype)

                        # NaNが含まれていれば警告を表示し0に置き換える
                        if torch.any(torch.isnan(latents)):
                            # accelerator.print("NaN found in latents, replacing with zeros")
                            logger.warning("NaN found in latents, replacing with zeros")
                            latents = torch.nan_to_num(latents, 0, out=latents)
                latents = latents * sdxl_model_util.VAE_SCALE_FACTOR

                if "text_encoder_outputs1_list" not in batch or batch["text_encoder_outputs1_list"] is None:
                    input_ids1 = batch["input_ids"]
                    input_ids2 = batch["input_ids2"]
                    with torch.set_grad_enabled(args.train_text_encoder):
                        # Get the text embedding for conditioning
                        # TODO support weighted captions
                        # if args.weighted_captions:
                        #     encoder_hidden_states = get_weighted_text_embeddings(
                        #         tokenizer,
                        #         text_encoder,
                        #         batch["captions"],
                        #         accelerator.device,
                        #         args.max_token_length // 75 if args.max_token_length else 1,
                        #         clip_skip=args.clip_skip,
                        #     )
                        # else:
                        # input_ids1 = input_ids1.to(accelerator.device)
                        # input_ids2 = input_ids2.to(accelerator.device)
                        input_ids1 = input_ids1.to(device)
                        input_ids2 = input_ids2.to(device)
                        # unwrap_model is fine for models not wrapped by accelerator
                        encoder_hidden_states1, encoder_hidden_states2, pool2 = train_util.get_hidden_states_sdxl(
                            args.max_token_length,
                            input_ids1,
                            input_ids2,
                            tokenizer1,
                            tokenizer2,
                            text_encoder1,
                            text_encoder2,
                            # None if not args.full_fp16 else weight_dtype,
                            weight_dtype,
                            # accelerator=accelerator,
                            accelerator=None,
                        )
                else:
                    # encoder_hidden_states1 = batch["text_encoder_outputs1_list"].to(accelerator.device).to(weight_dtype)
                    # encoder_hidden_states2 = batch["text_encoder_outputs2_list"].to(accelerator.device).to(weight_dtype)
                    # pool2 = batch["text_encoder_pool2_list"].to(accelerator.device).to(weight_dtype)
                    encoder_hidden_states1 = batch["text_encoder_outputs1_list"].to(device, dtype=weight_dtype)
                    encoder_hidden_states2 = batch["text_encoder_outputs2_list"].to(device, dtype=weight_dtype)
                    pool2 = batch["text_encoder_pool2_list"].to(device, dtype=weight_dtype)

                    # # verify that the text encoder outputs are correct
                    # ehs1, ehs2, p2 = train_util.get_hidden_states_sdxl(
                    #     args.max_token_length,
                    #     batch["input_ids"].to(text_encoder1.device),
                    #     batch["input_ids2"].to(text_encoder1.device),
                    #     tokenizer1,
                    #     tokenizer2,
                    #     text_encoder1,
                    #     text_encoder2,
                    #     None if not args.full_fp16 else weight_dtype,
                    # )
                    # b_size = encoder_hidden_states1.shape[0]
                    # assert ((encoder_hidden_states1.to("cpu") - ehs1.to(dtype=weight_dtype)).abs().max() > 1e-2).sum() <= b_size * 2
                    # assert ((encoder_hidden_states2.to("cpu") - ehs2.to(dtype=weight_dtype)).abs().max() > 1e-2).sum() <= b_size * 2
                    # assert ((pool2.to("cpu") - p2.to(dtype=weight_dtype)).abs().max() > 1e-2).sum() <= b_size * 2
                    # logger.info("text encoder outputs verified")

                # get size embeddings
                orig_size = batch["original_sizes_hw"]
                crop_size = batch["crop_top_lefts"]
                target_size = batch["target_sizes_hw"]
                # embs = sdxl_train_util.get_size_embeddings(orig_size, crop_size, target_size, accelerator.device).to(weight_dtype)
                embs = sdxl_train_util.get_size_embeddings(orig_size, crop_size, target_size, device).to(weight_dtype)

                # concat embeddings
                vector_embedding = torch.cat([pool2, embs], dim=1).to(weight_dtype)
                text_embedding = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=2).to(weight_dtype)

                # Sample noise, sample a random timestep for each image, and add noise to the latents,
                # with noise offset and/or multires noise if specified
                noise, noisy_latents, timesteps, huber_c = train_util.get_noise_noisy_latents_and_timesteps(args, noise_scheduler, latents)

                noisy_latents = noisy_latents.to(weight_dtype)  # TODO check why noisy_latents is not weight_dtype

                # Predict the noise residual
                # with accelerator.autocast():
                with torch.cuda.amp.autocast(enabled=args.mixed_precision == "fp16" or args.mixed_precision == "bf16", dtype=weight_dtype):
                    noise_pred = unet(noisy_latents, timesteps, text_embedding, vector_embedding)

                target = noise

                if (
                    args.min_snr_gamma
                    or args.scale_v_pred_loss_like_noise_pred
                    or args.v_pred_like_loss
                    or args.debiased_estimation_loss
                    or args.masked_loss
                ):
                    # do not mean over batch dimension for snr weight or scale v-pred loss
                    loss = train_util.conditional_loss(noise_pred.float(), target.float(), reduction="none", loss_type=args.loss_type, huber_c=huber_c)
                    if args.masked_loss:
                        loss = apply_masked_loss(loss, batch)
                    loss = loss.mean([1, 2, 3])

                    if args.min_snr_gamma:
                        loss = apply_snr_weight(loss, timesteps, noise_scheduler, args.min_snr_gamma)
                    if args.scale_v_pred_loss_like_noise_pred:
                        loss = scale_v_prediction_loss_like_noise_prediction(loss, timesteps, noise_scheduler)
                    if args.v_pred_like_loss:
                        loss = add_v_prediction_like_loss(loss, timesteps, noise_scheduler, args.v_pred_like_loss)
                    if args.debiased_estimation_loss:
                        loss = apply_debiased_estimation(loss, timesteps, noise_scheduler)

                    loss = loss.mean()  # mean over batch dimension
                else:
                    loss = train_util.conditional_loss(noise_pred.float(), target.float(), reduction="mean", loss_type=args.loss_type, huber_c=huber_c)

                # accelerator.backward(loss)
                loss.backward()
                # if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                if args.max_grad_norm != 0.0:
                    params_to_clip = []
                    for m in training_models:
                        params_to_clip.extend(m.parameters())
                    # accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    xm.optimizer_step(optimizer, barrier=True) # barrier is True for gradient clipping
                    torch.nn.utils.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                else:
                    xm.optimizer_step(optimizer, barrier=True) # barrier is True to sync gradients

                # optimizer.step()
                lr_scheduler.step()
                # optimizer.zero_grad(set_to_none=True)
                # for param in training_models[0].parameters():
                #     param.grad = None

            # Checks if the accelerator has performed an optimization step behind the scenes
            # if accelerator.sync_gradients:
            if xm.is_master_ordinal():
                progress_bar.update(1)
            global_step += 1

            # Within the training loop in sdxl_train.py, where sample_images is called
            sdxl_train_util.sample_images(
                None, # Removed accelerator
                args,
                None, #Removed epoch
                global_step,
                device, # Removed accelerator.device
                vae,
                [tokenizer1, tokenizer2],
                [text_encoder1, text_encoder2],
                unet,
                #Removed prompt_replacement=None
            )

            # 指定ステップごとにモデルを保存
            if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
                # accelerator.wait_for_everyone()
                xm.rendezvous("save_model_stepwise")
                # if accelerator.is_main_process:
                if xm.is_master_ordinal():
                    src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path
                    sdxl_train_util.save_sd_model_on_epoch_end_or_stepwise(
                        args,
                        False,
                        # accelerator,
                        None,
                        src_path,
                        save_stable_diffusion_format,
                        use_safetensors,
                        save_dtype,
                        epoch,
                        num_train_epochs,
                        global_step,
                        # accelerator.unwrap_model(text_encoder1),
                        # accelerator.unwrap_model(text_encoder2),
                        # accelerator.unwrap_model(unet),
                        text_encoder1,
                        text_encoder2,
                        unet,
                        vae,
                        logit_scale,
                        ckpt_info,
                        force_sync_upload=True if args.huggingface_repo_id is not None else False,
                    )

            current_loss = loss.detach().item()
            if args.logging_dir is not None:
                logs = {"loss": current_loss}
                if block_lrs is None:
                    train_util.append_lr_to_logs(logs, lr_scheduler, args.optimizer_type, including_unet=train_unet)
                else:
                    append_block_lr_to_logs(block_lrs, logs, lr_scheduler, args.optimizer_type)  # U-Net is included in block_lrs

                # accelerator.log(logs, step=global_step) # Replace with tensorboard
                if xm.is_master_ordinal():
                    for k, v in logs.items():
                        writer.add_scalar(k, v, global_step)

            loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
            avr_loss: float = loss_recorder.moving_average
            logs = {"avr_loss": avr_loss}  # , "lr": lr_scheduler.get_last_lr()[0]}
            if xm.is_master_ordinal():
                progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        # end of epoch
        if args.logging_dir is not None:
            logs = {"loss/epoch": loss_recorder.moving_average}
            # accelerator.log(logs, step=epoch + 1)
            if xm.is_master_ordinal():
                for k, v in logs.items():
                    writer.add_scalar(k, v, epoch + 1)

        # accelerator.wait_for_everyone()
        xm.rendezvous("save_model_epochwise")

        if args.save_every_n_epochs is not None:
            # if accelerator.is_main_process:
            if xm.is_master_ordinal():
                src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path
                sdxl_train_util.save_sd_model_on_epoch_end_or_stepwise(
                    args,
                    True,
                    # accelerator,
                    None,
                    src_path,
                    save_stable_diffusion_format,
                    use_safetensors,
                    save_dtype,
                    epoch,
                    num_train_epochs,
                    global_step,
                    # accelerator.unwrap_model(text_encoder1),
                    # accelerator.unwrap_model(text_encoder2),
                    # accelerator.unwrap_model(unet),
                    text_encoder1,
                    text_encoder2,
                    unet,
                    vae,
                    logit_scale,
                    ckpt_info,
                    force_sync_upload=True if args.huggingface_repo_id is not None else False,
                )

        sdxl_train_util.sample_images(
            # accelerator,
            None,
            args,
            epoch + 1,
            global_step,
            # accelerator.device,
            device,
            vae,
            [tokenizer1, tokenizer2],
            [text_encoder1, text_encoder2],
            unet,
            prompt_replacement=None
        )

    # is_main_process = accelerator.is_main_process
    is_main_process = xm.is_master_ordinal()
    # if is_main_process:
    # unet = accelerator.unwrap_model(unet)
    # text_encoder1 = accelerator.unwrap_model(text_encoder1)
    # text_encoder2 = accelerator.unwrap_model(text_encoder2)

    # accelerator.end_training()
    # No equivalent in torch-xla, manage training loop manually
    if is_main_process:
        if args.logging_dir is not None:
            writer.close()

    if args.save_state or args.save_state_on_train_end:        
        # train_util.save_state_on_train_end(args, accelerator) # Replace with torch-xla saving
        if xm.is_master_ordinal():
            logger.info("saving state.")
            save_model_folder = os.path.join(args.output_dir, "state")
            os.makedirs(save_model_folder, exist_ok=True)
            if train_unet:
                torch.save(unet.state_dict(), os.path.join(save_model_folder, "unet.pt"))
            if train_text_encoder1:
                torch.save(text_encoder1.state_dict(), os.path.join(save_model_folder, "text_encoder1.pt"))
            if train_text_encoder2:
                torch.save(text_encoder2.state_dict(), os.path.join(save_model_folder, "text_encoder2.pt"))
            torch.save(optimizer.state_dict(), os.path.join(save_model_folder, "optimizer.pt"))
            torch.save(lr_scheduler.state_dict(), os.path.join(save_model_folder, "lr_scheduler.pt"))
            torch.save({"global_step": global_step}, os.path.join(save_model_folder, "global_step.pt"))
            torch.save({"epoch": epoch + 1}, os.path.join(save_model_folder, "epoch.pt"))
        xm.rendezvous("saving_train_state")

    # del accelerator  # この後メモリを使うのでこれは消す

    if is_main_process:
        src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path
        sdxl_train_util.save_sd_model_on_train_end(
            args,
            src_path,
            save_stable_diffusion_format,
            use_safetensors,
            save_dtype,
            epoch,
            global_step,
            text_encoder1,
            text_encoder2,
            unet,
            vae,
            logit_scale,
            ckpt_info,
        )
        logger.info("model saved.")

def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    add_logging_arguments(parser)
    train_util.add_sd_models_arguments(parser)
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, False)
    train_util.add_masked_loss_arguments(parser)
    # deepspeed_utils.add_deepspeed_arguments(parser) # Remove deepspeed
    train_util.add_sd_saving_arguments(parser)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    custom_train_functions.add_custom_train_arguments(parser)
    sdxl_train_util.add_sdxl_training_arguments(parser)

    parser.add_argument(
        "--learning_rate_te1",
        type=float,
        default=None,
        help="learning rate for text encoder 1 (ViT-L) / text encoder 1 (ViT-L)の学習率",
    )
    parser.add_argument(
        "--learning_rate_te2",
        type=float,
        default=None,
        help="learning rate for text encoder 2 (BiG-G) / text encoder 2 (BiG-G)の学習率",
    )

    parser.add_argument(
        "--diffusers_xformers", action="store_true", help="use xformers by diffusers / Diffusersでxformersを使用する"
    )
    parser.add_argument("--train_text_encoder", action="store_true", help="train text encoder / text encoderも学習する")
    parser.add_argument(
        "--no_half_vae",
        action="store_true",
        help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precisionでも fp16/bf16 VAEを使わずfloat VAEを使う",
    )
    parser.add_argument(
        "--block_lr",
        type=str,
        default=None,
        help=f"learning rates for each block of U-Net, comma-separated, {UNET_NUM_BLOCKS_FOR_BLOCK_LR} values / "
        + f"U-Netの各ブロックの学習率、カンマ区切り、{UNET_NUM_BLOCKS_FOR_BLOCK_LR}個の値",
    )
    parser.add_argument(
        "--detect_anomaly",
        action="store_true",
        help="Enable anomaly detection for the autograd engine / autogradの異常検知を有効にする",
    )
    parser.add_argument("--resume_start_epoch", type=int, default=None, help="Start training from this epoch")

    return parser

if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    # train(args)
    # Start training processes with xmp.spawn
    xmp.spawn(train, args=(args,), nprocs=args.num_processes, start_method="fork")