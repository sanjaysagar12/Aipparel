import os
from dataclasses import dataclass, field
import logging 
log = logging.getLogger(__name__)
import hydra
from typing import Literal
import torch
import transformers
from typing import Optional
from omegaconf import OmegaConf
from trainers.utils import dict_to_cuda, dict_to_dtype, dict_to_cpu, AverageMeter
import deepspeed
from models.aipparel_model import AIpparelForCausalLM, AIpparelConfig
from models.llava import conversation as conversation_lib
from data.data_wrappers.data_wrapper import DataWrapper, DataWrapperConfig
from trainers.trainer import Trainer, TrainerConfig, ExperimentConfig
from data.datasets.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN)

import wandb as wb

from data.datasets.utils import (SHORT_QUESTION_LIST, 
                                 ANSWER_LIST, 
                                 DEFAULT_PLACEHOLDER_TOKEN, 
                                 DESCRIPTIVE_TEXT_SHORT_QUESTION_LIST, 
                                 SPECULATIVE_TEXT_SHORT_QUESTION_LIST, 
                                 SHORT_QUESTION_WITH_TEXT_LIST,
                                 EDITING_QUESTION_LIST,
                                 SampleToTensor
                                 )
import random

from models.llava.mm_utils import tokenizer_image_and_pattern_token

# FastAPI imports
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from typing import Dict, Any
from fastapi.middleware.cors import CORSMiddleware
# Global variables to store initialized components
app = FastAPI(title="AIpparel Inference API", version="1.0.0")
# Add CORS middleware to accept all requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all headers
)
inference_components = None

@dataclass
class MainConfig:
    version: str
    model_max_length: int
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    data_wrapper: DataWrapperConfig = field(default_factory=DataWrapperConfig)
    model: AIpparelConfig = field(default_factory=AIpparelConfig)
    precision: Literal["bf16", "fp16"] = "bf16"
    evaluate: bool = False
    conv_type: Literal["default", "v0", "v1", "vicuna_v1", "llama_2", "plain", "v0_plain", "llava_v0", "v0_mmtag", "llava_v1", "v1_mmtag", "llava_llama_2", "mpt"] = "llava_v1"
    pre_trained: Optional[str] = None
    from_start: bool = False

# Pydantic models for API
class InferenceRequest(BaseModel):
    user_input: str

class InferenceResponse(BaseModel):
    status: str
    patterns: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# FastAPI endpoints
@app.get("/")
async def root():
    return {"message": "AIpparel Inference API is running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "components_loaded": inference_components is not None
    }

@app.post("/inference", response_model=InferenceResponse)
async def run_inference(request: InferenceRequest):
    """Run inference on user input"""
    global inference_components
    
    if inference_components is None:
        return InferenceResponse(
            status="error",
            error="Model components not initialized. Please initialize the model first."
        )
    
    try:
        tokenizer, trainer, cfg, ddp_local_rank, torch_dtype = inference_components
        
        # Create input dictionary for inference
        input_dict = create_input_dict(request.user_input, tokenizer, cfg, ddp_local_rank, torch_dtype)
        
        # Run inference
        patterns = trainer.inference_setup(input_dict=input_dict)

        return InferenceResponse(
            status="success",
            patterns=patterns
        )
    
    except Exception as e:
        log.error(f"Error during inference: {str(e)}")
        return InferenceResponse(
            status="error",
            error=f"Inference failed: {str(e)}"
        )

def create_input_dict(user_input: str, tokenizer, cfg, ddp_local_rank, torch_dtype):
    """Create input dictionary for inference"""
    descriptive_text = user_input
    descriptive_text_question_list = DESCRIPTIVE_TEXT_SHORT_QUESTION_LIST
    answer_list = ANSWER_LIST
    
    # questions and answers
    questions = []
    answers = []
    print(f"[<>]GCDMM: descriptive_text_question_list: {descriptive_text_question_list}")

    question_template = random.choice(descriptive_text_question_list).format(sent=descriptive_text)
    print(f"[<>]GCDMM: question_template: {question_template}")
    questions.append(question_template)
    answer_template = random.choice(answer_list).format(pattern=DEFAULT_PLACEHOLDER_TOKEN)
    answers.append(answer_template)

    # Set conversation template
    conversation_lib.default_conversation = conversation_lib.conv_templates[cfg.conv_type]

    conversations = []
    conv = conversation_lib.default_conversation.copy()
    conv.messages = []
    conv.append_message(conv.roles[0], questions[0])
    conv.append_message(conv.roles[1], answers[0])
    conversations.append(conv.get_prompt())
    print(f"[<>]GCDMM: conversations: {conversations}")

    input_ids = tokenizer_image_and_pattern_token(
        prompt=conversations[0], 
        tokenizer=tokenizer, 
        pattern_ids=[[]], 
        pattern_place_holder_token=DEFAULT_PLACEHOLDER_TOKEN, 
        return_tensors="pt"
    )
    input_ids_list = [input_ids]

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id
    )

    attention_masks = input_ids.ne(tokenizer.pad_token_id)
    
    # Create input_dict with correct device and dtype
    input_dict = dict()
    input_dict["question_ids"] = input_ids.to(ddp_local_rank)
    input_dict["question_attention_masks"] = attention_masks.to(ddp_local_rank)
    input_dict["images_clip"] = torch.zeros((1, 3, 224, 224), dtype=torch_dtype, device=ddp_local_rank)
    input_dict["questions_pattern_endpoints"] = torch.zeros((1, 0, 2), dtype=torch_dtype, device=ddp_local_rank)
    input_dict["questions_pattern_endpoints_mask"] = torch.zeros((1, 0), dtype=torch.bool, device=ddp_local_rank)
    input_dict["questions_pattern_transformations"] = torch.zeros((1, 0, 7), dtype=torch_dtype, device=ddp_local_rank)
    input_dict["questions_pattern_transformations_mask"] = torch.zeros((1, 0), dtype=torch.bool, device=ddp_local_rank)
    
    return input_dict

def setup_model_and_trainer(cfg: MainConfig):
    """Setup model, tokenizer and trainer"""
    log.info(f"Working directory : {os.getcwd()}")
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    log.info(f"Output directory : {output_dir}")
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    master_process = (ddp_rank == 0)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        cfg.version,
        cache_dir=None,
        model_max_length=cfg.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    tokenizer.pad_token = tokenizer.unk_token
    data_wrapper: DataWrapper = hydra.utils.instantiate(
        cfg.data_wrapper,
        output_dir=output_dir
    )
    all_new_tokens = data_wrapper.get_all_token_names()

    num_added_tokens = tokenizer.add_tokens(all_new_tokens)

    if master_process:
        log.info(f"Added {num_added_tokens} tokens to the tokenizer.")
    token_name2_idx_dict = {}
    for token in all_new_tokens:
        token_idx = tokenizer(token, add_special_tokens=False).input_ids[0]
        token_name2_idx_dict[token] = token_idx
    
    if master_process:
        log.info(f"Token name to index dictionary: {token_name2_idx_dict}")
    data_wrapper.set_token_indices(token_name2_idx_dict)
    
    if cfg.model.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    torch_dtype = torch.float32
    if cfg.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif cfg.precision == "fp16":
        torch_dtype = torch.half

    model = AIpparelForCausalLM.from_pretrained(
        cfg.version, 
        torch_dtype=torch_dtype, 
        low_cpu_mem_usage=True, 
        **cfg.model, 
        vision_tower=data_wrapper.dataset.vision_tower, 
        panel_edge_indices=data_wrapper.panel_edge_type_indices, 
        gt_stats=data_wrapper.gt_stats
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=ddp_local_rank)
    
    # FIX 2: Set mm_projector dtype to match
    model.get_model().mm_projector.to(dtype=torch_dtype, device=ddp_local_rank)

    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    model.resize_token_embeddings(len(tokenizer))
    print("LEN: ", len(tokenizer), "tokens in tokenizer")
   
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        experiment_cfg=cfg.experiment,
        data_wrapper=data_wrapper,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
        ddp_local_rank=ddp_local_rank,
        precision=cfg.precision,
        output_dir=output_dir,
    )
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    
    # FIX 4: Pass checkpoint path for loading
    trainer.inference_eval_setup(
        model=model,
        in_config=config_dict,
        tokenizer=tokenizer,
        conv_type=cfg.conv_type,
        resume=cfg.pre_trained,  # FIXED: Use checkpoint path
        
    )
    
    return tokenizer, trainer, cfg, ddp_local_rank, torch_dtype

@hydra.main(version_base=None, config_path='./configs', config_name='config')
def main(cfg: MainConfig):
    global inference_components
    
    # Get DDP rank info
    ddp_rank = int(os.environ.get('RANK', 0))
    master_process = (ddp_rank == 0)
    
    # Setup model and trainer
    tokenizer, trainer, cfg, ddp_local_rank, torch_dtype = setup_model_and_trainer(cfg)
    
    # Store components globally for API access
    inference_components = (tokenizer, trainer, cfg, ddp_local_rank, torch_dtype)
    
    # Only start FastAPI server on the master process (rank 0)
    if master_process:
        log.info("Starting FastAPI server on port 8000 (master process)")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        log.info(f"Worker process (rank {ddp_rank}) initialized, waiting...")
        # Keep the worker process alive
        import time
        while True:
            time.sleep(60)  # Sleep to keep the process alive

if __name__ == "__main__":
    main()
