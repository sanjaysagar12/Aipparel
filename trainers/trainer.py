# Training loop func
from torch.utils.data import DataLoader
from typing import Literal, Optional, Dict, Any
import logging
log = logging.getLogger(__name__)
import torch
import wandb as wb
from PIL import Image
from dataclasses import dataclass, field
import os
import deepspeed
import numpy as np
import transformers
from functools import partial
from omegaconf import DictConfig
# My modules

from data.data_wrappers.data_wrapper import DataWrapper
from data.garment_tokenizers.special_tokens import DecodeErrorTypes
from models.aipparel_model import AIpparelForCausalLM
from trainers.utils import dict_to_cuda, AverageMeter, ProgressMeter, master_log, dict_to_cpu, dict_to_dtype
from .convert_zero_to_torch import get_fp32_state_dict_from_zero_checkpoint
import torch.distributed as dist
import time
import json

@dataclass
class WandbConfig:
    wandb_dir: Optional[str] = None
    wandb_cache_dir: Optional[str] = None

@dataclass
class ExperimentConfig:
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    run_id: Optional[str] = None
    local_dir: Optional[str] = None
    wandb_info: WandbConfig = field(default_factory=WandbConfig)

@dataclass
class TrainerConfig():
    _target_: str
    experiment_cfg: ExperimentConfig
    data_wrapper: DataWrapper 
    ddp_rank: int 
    ddp_local_rank: int
    ddp_world_size: int
    precision: str 
    lr: float
    beta1: float 
    beta2: float 
    grad_accumulation_steps: int 
    batch_size: int
    epoches: int
    steps_per_epoch: int
    save_freq: int
    
    

class Trainer():
    def __init__(
        self,
        experiment_cfg: ExperimentConfig,
        data_wrapper: DataWrapper, 
        ddp_rank: int, 
        ddp_local_rank: int,
        ddp_world_size: int,
        precision: str, 
        lr: float,
        beta1: float, 
        beta2: float, 
        grad_accumulation_steps: int, 
        batch_size: int,
        num_steps: int,
        save_freq: int,
        output_dir: str
        ):
        self.experiment_cfg = experiment_cfg
        self.datawrapper = data_wrapper
        self.lr = lr
        self.ddp_rank = ddp_rank
        self.ddp_local_rank = ddp_local_rank
        self.ddp_world_size = ddp_world_size
        self.device = f"cuda:{ddp_local_rank}"
        self.device_type = "cuda" if self.device.startswith("cuda") else "cpu"
        self.master_process = (self.ddp_rank == 0)
        self.save_freq = save_freq
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.grad_accumulation_steps = grad_accumulation_steps
        self.precision=precision
        self.output_dir = output_dir
        self.cast_dtype = torch.half if self.precision == "fp16" else (torch.bfloat16 if self.precision == "bf16" else torch.float32)
        self.optimizer_config = {
            "type": "AdamW",
            "params": {
                "lr": lr,
                "weight_decay": 0.0,
                "betas": (beta1, beta2)
            }
        }
        self.scheduler_config = {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": self.num_steps,
                "warmup_min_lr": 0,
                "warmup_max_lr": lr,
                "warmup_num_steps": 100,
                "warmup_type": "linear"
            }
        }
        self.start_step = 0
        

    def training_setup(
        self, 
        in_config: DictConfig,
        model: AIpparelForCausalLM, 
        tokenizer: transformers.PreTrainedTokenizer, 
        conv_type: str,
        from_start: bool = False,
        resume: Optional[str]=None
        ):
        self.tokenizer = tokenizer
        ds_config = {
            "train_micro_batch_size_per_gpu": self.batch_size,
            "gradient_accumulation_steps": self.grad_accumulation_steps,
            "fp16": {
                "enabled": self.precision == "fp16",
            },
            "bf16": {
                "enabled": self.precision == "bf16",
            },
            "gradient_clipping": 1.0,
            "zero_optimization": {
                "stage": 2,
                "contiguous_gradients": True,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 5e8,
                "allgather_bucket_size": 5e8,
            },
        }
        ds_config["optimizer"] = self.optimizer_config
        ds_config["scheduler"] = self.scheduler_config
        
        self.model_engine, self.optimizer, self.train_loader, self.scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            training_data=self.datawrapper.training,
            collate_fn=partial(
                self.datawrapper.collate_fn,
                tokenizer=tokenizer,
                conv_type=conv_type,
                use_mm_start_end=model.config.mm_use_im_start_end,
                local_rank=self.ddp_local_rank,
            ),
            config=ds_config,
        )
        if self.datawrapper.validation is not None:
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                self.datawrapper.validation, shuffle=False, drop_last=False
            )
            self.val_loader = DataLoader(
                self.datawrapper.validation,
                batch_size=1,
                shuffle=False,
                num_workers=12,
                pin_memory=False,
                sampler=val_sampler,
                collate_fn=partial(
                    self.datawrapper.collate_fn,
                    tokenizer=tokenizer,
                    conv_type=conv_type,
                    use_mm_start_end=model.config.mm_use_im_start_end,
                    local_rank=self.ddp_local_rank,
                ),
            )
        else:
            self.val_loader = None
            
        self._start_experiment(in_config, resume, from_start)
        master_log(self.ddp_local_rank, log, 'NN training Using device: {}'.format(self.device))
        
        self.log_dir = self.output_dir
    
    def eval_setup(
        self, 
        in_config: DictConfig,
        model: AIpparelForCausalLM, 
        tokenizer: transformers.PreTrainedTokenizer, 
        conv_type: str,
        resume: Optional[str]=None
        ):
        
        ds_config = {
            "train_micro_batch_size_per_gpu": self.batch_size,
            "gradient_accumulation_steps": 1,
            "fp16": {
                "enabled": self.precision == "fp16",
            },
            "bf16": {
                "enabled": self.precision == "bf16",
            },
            "gradient_clipping": 1.0,
            "zero_optimization": {
                "stage": 2,
                "contiguous_gradients": True,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 5e8,
                "allgather_bucket_size": 5e8,
            },
        }
        ds_config["optimizer"] = self.optimizer_config
        self.model_engine, _, _, _ = deepspeed.initialize(
            model=model,
            config=ds_config,
        )
        
        assert self.datawrapper.validation is not None
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            self.datawrapper.validation, shuffle=False, drop_last=False
        )
        self.val_loader = DataLoader(
            self.datawrapper.validation,
            batch_size=1,
            shuffle=False,
            num_workers=12,
            pin_memory=False,
            sampler=val_sampler,
            collate_fn=partial(
                self.datawrapper.collate_fn,
                tokenizer=tokenizer,
                conv_type=conv_type,
                use_mm_start_end=model.config.mm_use_im_start_end,
                local_rank=self.ddp_local_rank,
            ),
        )
        train_val_sampler = torch.utils.data.distributed.DistributedSampler(
            self.datawrapper.training, shuffle=False, drop_last=False
        )
        self.train_val_loader = DataLoader(
            self.datawrapper.training,
            batch_size=1,
            shuffle=False,
            num_workers=12,
            pin_memory=False,
            sampler=train_val_sampler,
            collate_fn=partial(
                self.datawrapper.collate_fn,
                tokenizer=tokenizer,
                conv_type=conv_type,
                use_mm_start_end=model.config.mm_use_im_start_end,
                local_rank=self.ddp_local_rank,
            ),
        )
        self.tokenizer=tokenizer
        self.log_dir = self.output_dir
        self._start_experiment(in_config, resume, False)
        if self.master_process:
            wb.watch(self.model_engine, log='all')
        
    
    def generation_step(self, step: int, subset: Literal["validation", "train"]="validation"):
        if subset == "validation":
            loader = self.val_loader
            gen_len = len(self.val_loader)
        elif subset == "train":
            loader = self.train_val_loader
            gen_len = min(len(self.train_val_loader), 200)
        else:
            raise ValueError(f"Subset {subset} not supported")
        
        eval_sample_time = AverageMeter("Time", ":6.3f")
        self.model_engine.eval()
        if self.val_loader is None:
            log.info("No validation data provided, skipping validation")
            return None, None
        all_patterns = []
        all_gt_patterns = []
        all_questions = []
        all_image_paths = []
        all_output_texts = []
        all_errored_texts = []
        all_sample_types = []
        save_path = f'{self.output_dir}/step_{step}/{subset}'
        os.makedirs(save_path, exist_ok=True)
        dist.barrier()
        for i, input_dict in enumerate(loader):
            if i == gen_len:
                break
            torch.cuda.empty_cache()
            input_dict = dict_to_cuda(input_dict)
            input_dict = dict_to_dtype(input_dict,
                self.cast_dtype,
                [
                    "images_clip",
                    "param_targets",
                    "param_target_endpoints",
                    "param_target_transformations",
                    "questions_pattern_endpoints",
                    "questions_pattern_transformations"
                    
                ]
            )

            end = time.time()
            output_dict = self.model_engine.module.evaluate(
                input_dict["images_clip"],
                input_dict["question_ids"],
                input_dict["question_attention_masks"],
                endpoints=input_dict["questions_pattern_endpoints"],
                endpoints_mask=input_dict["questions_pattern_endpoints_mask"],
                transformations=input_dict["questions_pattern_transformations"],
                transformations_mask=input_dict["questions_pattern_transformations_mask"],
                max_new_tokens=2100
            )
            output_dict = dict_to_cpu(output_dict)
            output_dict = dict_to_dtype(output_dict, torch.float32)
            output_dict["input_mask"] = torch.arange(output_dict["output_ids"].shape[1]).reshape(1, -1) >= input_dict["question_ids"].shape[1]
            output_text, patterns, error_type = self.datawrapper.decode(output_dict, self.tokenizer)
            try:
                data_name = input_dict["gt_patterns"][0][-1].name
                os.makedirs(os.path.join(save_path, data_name), exist_ok=True)
                patterns.serialize(os.path.join(save_path, data_name), spec_only=False, with_3d=False, with_text=False, view_ids=False, to_subfolder=False, tag=f'_pred')
                for gt_pattern in input_dict["gt_patterns"][0]:
                    gt_pattern.serialize(os.path.join(save_path, data_name), spec_only=False, with_3d=False, with_text=False, view_ids=False, to_subfolder=False, tag=f'_gt')
                f = open(os.path.join(save_path, data_name, "output.txt"), "w")
                question = input_dict["questions_list"][0]
                f.write(f"Question: {question}\n")
                f.write(f"Output Text: {output_text}\n")
                f.close()
                if os.path.isfile(input_dict["image_paths"][0]):
                    cond_img = Image.open(input_dict["image_paths"][0])
                    cond_img.save(os.path.join(save_path, data_name, 'input.png'))
            except Exception as e:
                log.error(e)
                pass
            if error_type != DecodeErrorTypes.NO_ERROR:
                all_errored_texts.append((error_type, output_text))
            eval_sample_time.update(time.time() - end)
            end = time.time()
            all_gt_patterns.append(input_dict["gt_patterns"][0])
            all_sample_types.append(input_dict["sample_type"][0].cpu().numpy())  
            all_patterns.append(patterns)
            all_questions.append(input_dict["questions_list"][0])
            all_image_paths.append(input_dict["image_paths"][0])
            all_output_texts.append(output_text)
            log.info("Rank: {}, Progress: [{}/{}]".format(self.ddp_rank, i, gen_len))
            
        
        # save all the errored texts as single txt file
        with open(os.path.join(save_path, f"errored_texts{self.ddp_local_rank}.txt"), "w") as f:
            for text in all_errored_texts:
                f.write("Error Type: " + str(text[0]) + "\n")
                f.write("Output Text: " + str(text[1]) + "\n")
        log.info(f"Saved {len(all_errored_texts)} errored texts to {save_path}")
            
        (total_num_panel_correct, 
         num_edge_accs, 
         num_edge_correct_accs, 
         vertex_L2s, 
         transl_l2s, 
         rots_l2s, 
         stitch_accs, 
         _) = self.datawrapper.evaluate_patterns(all_patterns, [p[-1] for p in all_gt_patterns])
        
        modewise_panel_accs = {}
        total_panel_accs = total_num_panel_correct.mean()
        modewise_edge_accs = {}
        total_edge_acc = num_edge_accs[num_edge_accs != -1].sum() \
            / max((num_edge_accs != -1).sum(), 1)
        modewise_edge_correct_accs = {}
        total_edge_correct_acc = num_edge_correct_accs[num_edge_correct_accs != -1].sum() \
            / max((num_edge_correct_accs != -1).sum(), 1)
        modewise_vertex_L2s = {}
        total_vertex_L2 = vertex_L2s[vertex_L2s != -1].sum() \
            / max((vertex_L2s != -1).sum(), 1)
        modewise_transl_l2s = {}
        total_transl_l2 = transl_l2s[transl_l2s != -1].sum() \
            / max((transl_l2s != -1).sum(), 1)
        modewise_rots_l2s = {}
        total_rots_l2 = rots_l2s[rots_l2s != -1].sum() \
            / max((rots_l2s != -1).sum(), 1)
        modewise_stitch_accs = {}
        total_stitch_acc = stitch_accs[stitch_accs != -1].sum() \
            / max((stitch_accs != -1).sum(), 1)
        
        all_sample_types = np.array(all_sample_types)
        for i, mode in enumerate(self.datawrapper.get_mode_names()):
            mode_mask = all_sample_types == i
            mode_mask = torch.from_numpy(mode_mask).to(num_edge_accs).bool()
            if not mode_mask.any():
                continue
            modewise_panel_accs[mode] = total_num_panel_correct[mode_mask].mean()
            modewise_edge_accs[mode] = num_edge_accs[torch.logical_and(mode_mask, num_edge_accs != -1)].sum() \
                / max((num_edge_accs[mode_mask] != -1).sum(), 1)
            modewise_edge_correct_accs[mode] = num_edge_correct_accs[torch.logical_and(mode_mask, num_edge_correct_accs != -1)].sum() \
                / max((num_edge_correct_accs[mode_mask] != -1).sum(), 1)
            modewise_vertex_L2s[mode] = vertex_L2s[torch.logical_and(mode_mask, vertex_L2s != -1)].sum() \
                / max((vertex_L2s[mode_mask] != -1).sum(), 1)
            modewise_transl_l2s[mode] = transl_l2s[torch.logical_and(mode_mask, transl_l2s != -1)].sum() \
                / max((transl_l2s[mode_mask] != -1).sum(), 1)
            modewise_rots_l2s[mode] = rots_l2s[torch.logical_and(mode_mask, rots_l2s != -1)].sum() \
                / max((rots_l2s[mode_mask] != -1).sum(), 1)
            modewise_stitch_accs[mode] = stitch_accs[torch.logical_and(mode_mask, stitch_accs != -1)].sum() \
                / max((stitch_accs[mode_mask] != -1).sum(), 1)

        eval_sample_time.all_reduce()
        for mode in modewise_panel_accs.keys():
            dist.all_reduce(modewise_panel_accs[mode], dist.ReduceOp.AVG)
            dist.all_reduce(modewise_edge_accs[mode], dist.ReduceOp.AVG)
            dist.all_reduce(modewise_edge_correct_accs[mode], dist.ReduceOp.AVG)
            dist.all_reduce(modewise_vertex_L2s[mode], dist.ReduceOp.AVG)
            dist.all_reduce(modewise_transl_l2s[mode], dist.ReduceOp.AVG)
            dist.all_reduce(modewise_rots_l2s[mode], dist.ReduceOp.AVG)
            dist.all_reduce(modewise_stitch_accs[mode], dist.ReduceOp.AVG)
        dist.all_reduce(total_panel_accs, dist.ReduceOp.AVG)
        dist.all_reduce(total_edge_acc, dist.ReduceOp.AVG)
        dist.all_reduce(total_edge_correct_acc, dist.ReduceOp.AVG)
        dist.all_reduce(total_vertex_L2, dist.ReduceOp.AVG)
        dist.all_reduce(total_transl_l2, dist.ReduceOp.AVG)
        dist.all_reduce(total_rots_l2, dist.ReduceOp.AVG)
        if self.master_process:
            log.info(
                "{}: Step: [{}]\t"
                "Sample: {:.3f} ({:.3f})\t"
                "Num Panel Accuracy: {:.8f}\t"
                "Num Edge Accuracy: {:.8f}\t"
                "Num Correct Edge Accuracy: {:.8f}\t"
                "Vertex L2: {:.8f}\t"
                "translation L2: {:.8f}\t"
                "rotation L2: {:.8f}\t"
                "stitch acc: {:.8f}\t".format(
                    subset,
                    step,
                    eval_sample_time.val,
                    eval_sample_time.avg,
                    total_panel_accs.cpu().item(),
                    total_edge_acc.cpu().item(),
                    total_edge_correct_acc.cpu().item(),
                    total_vertex_L2.cpu().item(),
                    total_transl_l2.cpu().item(),
                    total_rots_l2.cpu().item(),
                    total_stitch_acc.cpu().item(),
                )
            )
            log_dict = {
                f"val/{subset}_num_panel_accuracy": total_panel_accs.cpu().item(),
                f"val/{subset}_num_edge_accuracy": total_edge_acc.cpu().item(),
                f"val/{subset}_num_correct_edge_accuracy": total_edge_correct_acc.cpu().item(),
                f"val/{subset}_vertex_L2": total_vertex_L2.cpu().item(),
                f"val/{subset}_translation_L2": total_transl_l2.cpu().item(),
                f"val/{subset}_rotation_L2": total_rots_l2.cpu().item(),
                f"val/{subset}_stitch_acc": total_stitch_acc.cpu().item(),
                f"val/{subset}_sample_time": eval_sample_time.avg,
            }
            for mode in modewise_panel_accs.keys():
                log_dict.update({
                    f"val/{subset}_{mode}_num_panel_accuracy": modewise_panel_accs[mode].cpu().item(),
                    f"val/{subset}_{mode}_num_edge_accuracy": modewise_edge_accs[mode].cpu().item(),
                    f"val/{subset}_{mode}_num_correct_edge_accuracy": modewise_edge_correct_accs[mode].cpu().item(),
                    f"val/{subset}_{mode}_vertex_L2": modewise_vertex_L2s[mode].cpu().item(),
                    f"val/{subset}_{mode}_translation_L2": modewise_transl_l2s[mode].cpu().item(),
                    f"val/{subset}_{mode}_rotation_L2": modewise_rots_l2s[mode].cpu().item(),
                    f"val/{subset}_{mode}_stitch_acc": modewise_stitch_accs[mode].cpu().item(),
                })
            wb.log(log_dict, step=step)
            
    @torch.no_grad()   
    def eval_step(self, step: int):
        # validation loss
        batch_time_meter = AverageMeter("Time", ":6.3f")
        data_time_meter = AverageMeter("Data", ":6.3f")
        total_loss_meter = AverageMeter("Total Loss", ":.4f")
        ce_loss_meter = AverageMeter("CE Loss", ":.4f")
        edge_loss_meter = AverageMeter("Edge Loss", ":.4f")
        
        mode_names = self.datawrapper.get_mode_names()
        
        modewise_total_loss_meters = [AverageMeter(f"{mode} Total Loss", ":.4f") for mode in mode_names]
        modewise_ce_loss_meters = [AverageMeter(f"{mode} CE Loss", ":.4f") for mode in mode_names]
        modewise_edge_loss_meters = [AverageMeter(f"{mode} Edge Loss", ":.4f") for mode in mode_names]
        edge_type_loss_meters = {
            self.datawrapper.panel_edge_type_indices.get_index_token(
                ind
                ).value: AverageMeter(
                    f"{self.datawrapper.panel_edge_type_indices.get_index_token(ind).value} Loss", 
                    ":.4f"
                ) 
            for ind in self.datawrapper.panel_edge_type_indices.get_all_indices()
        }
        all_meters: list[AverageMeter] = (
            [
                batch_time_meter, 
                data_time_meter, 
                total_loss_meter, 
                ce_loss_meter,
                edge_loss_meter
            ] 
            + modewise_total_loss_meters 
            + modewise_ce_loss_meters 
            + modewise_edge_loss_meters 
            + [meter for meter in edge_type_loss_meters.values()]
        )
        progress = ProgressMeter(
            log,
            self.ddp_rank,
            all_meters[:-len(edge_type_loss_meters)],
            prefix="Step:[{}]",
        )
        self.model_engine.eval()
        if self.val_loader is None:
            log.info("No validation data provided, skipping validation")
            return None, None
        
        end = time.time()
        for kk, input_dict in enumerate(self.val_loader):
            torch.cuda.empty_cache()
            input_dict = dict_to_cuda(input_dict)
            input_dict = dict_to_dtype(input_dict,
                self.cast_dtype,
                [
                    "images_clip",
                    "param_targets",
                    "param_target_endpoints",
                    "param_target_transformations",
                ])
            data_time_meter.update(time.time() - end)
            end = time.time()
            
            output_dict = self.model_engine(**input_dict)
            loss = output_dict["total_loss"]
            ce_loss = output_dict["ce_loss"]
            edge_loss = output_dict.get("edge_loss", 0)
            edge_loss_meter.update(edge_loss.mean(), self.batch_size)
            for k, meter in edge_type_loss_meters.items():
                if f"{k}_loss" in output_dict:
                    meter.update(output_dict[f"{k}_loss"], self.batch_size)
            for i, _ in enumerate(mode_names):
                mask = input_dict["sample_type"] == i
                if not mask.any():
                    continue
                mode_batch_size = mask.sum()
                modewise_total_loss_meters[i].update(loss[mask].mean(), mode_batch_size)
                modewise_ce_loss_meters[i].update(ce_loss[mask].mean(), mode_batch_size)
                modewise_edge_loss_meters[i].update(edge_loss[mask].mean(), mode_batch_size)
                    
            total_loss_meter.update(loss.mean(), self.batch_size)
            ce_loss_meter.update(ce_loss.mean(), self.batch_size)
            
            batch_time_meter.update(time.time() - end)
            end = time.time()
            log.info("Rank: {}, Progress: [{}/{}]".format(self.ddp_rank, kk, len(self.val_loader)))
        
        for meter in all_meters:
            meter.all_reduce()
        
        if self.master_process:
            progress.display(step + 1)
            log_dict = {f"val/{meter.name}": meter.avg for meter in all_meters}
            wb.log(log_dict)


    def _start_experiment(self, in_config: DictConfig, resume: Optional[str]=None, from_start: bool=False):
        # resume deepspeed checkpoint
        os.environ['WANDB_DIR'] = self.experiment_cfg.wandb_info.wandb_dir
        os.environ['WANDB_CACHE_DIR'] = self.experiment_cfg.wandb_info.wandb_cache_dir
        if self.master_process:
            wb.init(
                name=self.experiment_cfg.run_name, project=self.experiment_cfg.project_name, config=in_config, 
                resume='allow', id=self.experiment_cfg.run_id,    # Resuming functionality
                dir=self.experiment_cfg.local_dir,
                anonymous='allow')

        if resume:
            if os.path.isdir(resume):
                log.info("Reconstructing torch checkpoints from {}".format(resume))
                latest_path = os.path.join(resume, 'latest')
                if os.path.isfile(latest_path):
                    with open(latest_path, 'r') as fd:
                        tag = fd.read().strip()
                else:
                    raise ValueError(f"Unable to find 'latest' file at {latest_path}")

                state_dict = get_fp32_state_dict_from_zero_checkpoint(resume, tag, exclude_frozen_parameters=True)
                if not from_start:
                    self.start_step = int(tag.replace("global_step", ""))
            elif os.path.isfile(resume) and resume.endswith('.pth'):
                log.info("Loading torch checkpoint from {}".format(resume))
                state_dict = torch.load(resume, map_location='cpu')
            else:
                raise ValueError("Unknown checkpoint format: {}".format(resume))
            self.model_engine.module.load_state_dict(state_dict, strict=False)
            
            master_log(
                self.ddp_rank, 
                log,
                "resume training from {}, start from epoch {}".format(
                    resume, self.start_step
                )
            )
        if self.master_process:
            wb.watch(self.model_engine, log='all')
    def fit(self):
        """Fit provided model to reviosly configured dataset"""
        
        if not self.datawrapper:
            raise RuntimeError('{}::Error::fit before dataset was provided. run use_dataset() first'.format(self.__class__.__name__))
        
        self._fit_loop()
        master_log(self.ddp_local_rank, log, "Finished training")


    def _fit_loop(self):
        """Fit loop with the setup already performed. Assumes wandb experiment was initialized"""
        batch_time_meter = AverageMeter("Time", ":6.3f")
        data_time_meter = AverageMeter("Data", ":6.3f")
        total_loss_meter = AverageMeter("Total Loss", ":.4f")
        ce_loss_meter = AverageMeter("CE Loss", ":.4f")
        edge_loss_meter = AverageMeter("Edge Loss", ":.4f")
        
        mode_names = self.datawrapper.get_mode_names()
        
        modewise_total_loss_meters = [AverageMeter(f"{mode} Total Loss", ":.4f") for mode in mode_names]
        modewise_ce_loss_meters = [AverageMeter(f"{mode} CE Loss", ":.4f") for mode in mode_names]
        modewise_edge_loss_meters = [AverageMeter(f"{mode} Edge Loss", ":.4f") for mode in mode_names]
        edge_type_loss_meters = {
            self.datawrapper.panel_edge_type_indices.get_index_token(
                ind
                ).value: AverageMeter(
                    f"{self.datawrapper.panel_edge_type_indices.get_index_token(ind).value} Loss", 
                    ":.4f"
                ) 
            for ind in self.datawrapper.panel_edge_type_indices.get_all_indices()
        }
        all_meters: list[AverageMeter] = (
            [
                batch_time_meter, 
                data_time_meter, 
                total_loss_meter, 
                ce_loss_meter,
                edge_loss_meter
            ] 
            + modewise_total_loss_meters 
            + modewise_ce_loss_meters 
            + modewise_edge_loss_meters 
            + [meter for meter in edge_type_loss_meters.values()]
        )
        
        progress = ProgressMeter(
            log, 
            self.ddp_local_rank, 
            all_meters,
            prefix="Step: [{}]",
        )
        
        train_loader_iter = iter(self.train_loader)
        self.model_engine.train()
        master_log(self.ddp_local_rank, log, "Start training")
        for step in range(self.num_steps):
            # training step
            for i in range(self.grad_accumulation_steps):
                start_time = time.time()
                try:
                    input_dict = next(train_loader_iter)
                except:
                    train_loader_iter = iter(self.train_loader)
                    input_dict = next(train_loader_iter)
                
                data_time_meter.update(time.time() - start_time)
                input_dict = dict_to_cuda(input_dict)
                input_dict = dict_to_dtype(
                input_dict,
                self.cast_dtype,
                [
                    "images_clip",
                    "param_targets",
                    "param_target_endpoints",
                    "param_target_transformations",
                ],
            )

                output_dict = self.model_engine(**input_dict)
                loss = output_dict["total_loss"]
                self.model_engine.backward(loss.mean())
                self.model_engine.step()

                # logging stuff
                batch_time_meter.update(time.time() - start_time)
                ce_loss = output_dict["ce_loss"]
                total_loss_meter.update(loss.mean().item(), self.batch_size)
                ce_loss_meter.update(ce_loss.mean().item(), self.batch_size)
                edge_loss = output_dict.get("edge_loss", 0)
                edge_loss_meter.update(edge_loss.mean().item(), self.batch_size)
                for k, meter in edge_type_loss_meters.items():
                    if f"{k}_loss" in output_dict:
                        meter.update(output_dict[f"{k}_loss"].item(), self.batch_size)
                for i in range(len(mode_names)):
                    mask = input_dict["sample_type"] == i
                    if not mask.any():
                        continue
                    mode_batch_size = mask.sum()
                    modewise_total_loss_meters[i].update(loss[mask].mean(), mode_batch_size)
                    modewise_ce_loss_meters[i].update(ce_loss[mask].mean(), mode_batch_size)
                    modewise_edge_loss_meters[i].update(edge_loss[mask].mean(), mode_batch_size)
                    
            if self.ddp_world_size > 1:
                for meter in all_meters:
                    meter.all_reduce()
                if self.master_process:
                    progress.display(step + 1)
                    log_dict = {f"train/{meter.name}": meter.avg for meter in all_meters}
                    wb.log(log_dict, step=step)

                for meter in all_meters:
                    meter.reset()

            if step != 0:
                curr_lr = self.scheduler.get_last_lr()
                if self.master_process:
                    wb.log({"train/lr": curr_lr[0]}, step)
                
            is_last_step = step == self.num_steps - 1
            if (step % self.save_freq == 0 ) or is_last_step:
                self.model_engine.save_checkpoint(os.path.join(self.log_dir, f"ckpt_{step}"))
    
    def inferance_setup(
        self,
        in_config: DictConfig,
        model: AIpparelForCausalLM,
        tokenizer: transformers.PreTrainedTokenizer,
        conv_type: str,
        resume: Optional[str] = None,
        input_dict: Dict[Any, Any] = None,
    ):
        ds_config = {
            "train_micro_batch_size_per_gpu": self.batch_size,
            "gradient_accumulation_steps": 1,
            "fp16": {
                "enabled": self.precision == "fp16",
            },
            "bf16": {
                "enabled": self.precision == "bf16",
            },
            "gradient_clipping": 1.0,
            "zero_optimization": {
                "stage": 2,
                "contiguous_gradients": True,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 5e8,
                "allgather_bucket_size": 5e8,
            },
        }
        ds_config["optimizer"] = self.optimizer_config
        
        
        self.model_engine, _, _, _ = deepspeed.initialize(
            model=model,
            config=ds_config,
        )

        self.tokenizer = tokenizer
        self.log_dir = self.output_dir
        self._start_experiment(in_config, resume, False)
        
        # ADDED: Explicit checkpoint loading
        if resume and os.path.exists(resume):
            log.info(f"Loading checkpoint from {resume}")
            try:
                state_dict = torch.load(resume, map_location='cpu')
                self.model_engine.module.load_state_dict(state_dict, strict=False)
                log.info("Checkpoint loaded successfully")
            except Exception as e:
                log.error(f"Failed to load checkpoint: {e}")
        
        self.model_engine.eval()
        
        output_dict = self.model_engine.module.evaluate(
            input_dict["images_clip"],
            input_dict["question_ids"],
            input_dict["question_attention_masks"],
            endpoints=input_dict["questions_pattern_endpoints"],
            endpoints_mask=input_dict["questions_pattern_endpoints_mask"],
            transformations=input_dict["questions_pattern_transformations"],
            transformations_mask=input_dict["questions_pattern_transformations_mask"],
            max_new_tokens=2100
        )

        output_dict = dict_to_cpu(output_dict)
        output_dict = dict_to_dtype(output_dict, torch.float32)
        output_dict["input_mask"] = torch.arange(output_dict["output_ids"].shape[1]).reshape(1, -1) >= input_dict["question_ids"].shape[1]
            
        output_text, patterns, error_type = self.datawrapper.decode(output_dict, self.tokenizer)
        print("Output dictionary keys:", list(output_dict.keys()))
        print("Output text:", output_text)
        print("Patterns:", patterns)
        print("Error type:", error_type)
        print("Patterns spec", patterns.spec)
        save_path = f'{self.output_dir}/text_inference_output'
        os.makedirs(save_path, exist_ok=True)
        try:
            with open(os.path.join(save_path, "output.txt"), "w") as f:
                f.write(f"Output Text: {output_text}\n")
                f.write(f"Error Type: {error_type}\n")
            with open(os.path.join(save_path, "patterns.json"), "w") as f:
                json.dump(patterns.spec, f, indent=4)
        except Exception as e:
            log.error(f"Error saving output: {e}")
