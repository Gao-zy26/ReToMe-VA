import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from attentionControl import AttentionControlEdit_uc
import diff_latent_attack_vid
from diffusers.utils.import_utils import is_xformers_available
from packaging import version
import numpy as np
import os
import random
import sys
import argparse
from gluoncv.torch.model_zoo import get_model
from gluoncv.torch.engine.config import get_cfg_defaults
from config import CONFIG_PATH
from utils import view_images, unorm
from datasets.kinetics import get_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained_diffusion_path',
                    default="stabilityai/stable-diffusion-2-base",
                    type=str,
                    help='Change the path to `stabilityai/stable-diffusion-2-base` if want to use the pretrained model')

parser.add_argument('--diffusion_steps', default=20, type=int, help='Total DDIM sampling steps')
parser.add_argument('--start_step', default=15, type=int, help='Which DDIM step to start the attack')
parser.add_argument('--iterations', default=4, type=int, help='Start iterations of optimizing the adv_image')
parser.add_argument('--res', default=224, type=int, help='Input image resized resolution')
parser.add_argument('--attack_loss_weight', default=10, type=int, help='attack loss weight factor')
parser.add_argument('--self_attn_loss_weight', default=100, type=int, help='self attention loss weight factor')
parser.add_argument('--model_t', type=str, default='slow_50_16', help='model type')
parser.add_argument("--input_path", type=str, help="input path")
parser.add_argument("--input_csv", type=str, help="input sample csv file")
parser.add_argument("--local_rank", default=0, type=int, help='node rank for distributed training')
parser.add_argument("--test_dir", type=str, help="Output adv frames path, test_dir")
parser.add_argument("--enable_xformers_memory_efficient_attention", default=True, help="Whether or not to use xformers.")
parser.add_argument("--retome", action='store_true', help="Whether or not to use retome.")
parser.add_argument("--merge_ratio", default=0.5, type=float, help='merge_ratio')

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(42)

def attack_single_vid(frames, label, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    adv_images, success, _  = diff_latent_attack_vid.attack_per_step(ldm_stable, frames, label, args.controller, model_t, model_name=args.model_t, 
                                        save_path=save_dir, args=args)
    idxs = [i for i in range(args.noFrames)]
    for i in idxs: # TODO: save 的接口？
        view_images(adv_images[i:i+1] * 255, show=False, save_path=os.path.join(save_dir, '%06d.png'%i))
    return success


if __name__ == "__main__":
    args = parser.parse_args()
    assert args.res % 32 == 0 and args.res >= 96, "Please ensure the input resolution be a multiple of 32 and also >= 96."
    device = torch.device('cuda', args.local_rank)
    args.device = device
    torch.backends.cudnn.benchmark = True

    diffusion_steps = args.diffusion_steps 
    start_step = args.start_step  
    iterations = args.iterations  
    res = args.res  

    ldm_stable = StableDiffusionPipeline.from_pretrained(
        args.pretrained_diffusion_path,
        use_safetensors=True,
    ).to(device)
    ldm_stable.scheduler = DDIMScheduler.from_config(ldm_stable.scheduler.config)
    ldm_stable.enable_vae_slicing()

    if args.retome:
        print(f'merge_ratio: {args.merge_ratio}')

    if args.enable_xformers_memory_efficient_attention:
        print('use xformers!')
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                print("xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details.")
            ldm_stable.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    cfg = get_cfg_defaults()
    cfg.merge_from_file(CONFIG_PATH[args.model_t])
    args.noFrames = cfg.CONFIG.DATA.CLIP_LEN
    print(f'use model {args.model_t}, number of frames {args.noFrames}')
    cfg.CONFIG.MODEL.PRETRAINED = True

    model_t = get_model(cfg).eval().to(device) 
    model_t.to(device)
    model_t.requires_grad_(False)
    data_dir = args.input_path

    self_replace_steps = 1.
    args.controller = AttentionControlEdit_uc(diffusion_steps, self_replace_steps, args.res, args.noFrames)

    save_dir = args.test_dir

    '''data'''
    cfg.CONFIG.DATA.VAL_ANNO_PATH = args.input_csv
    cfg.CONFIG.DATA.VAL_DATA_PATH = args.input_path
    cfg.CONFIG.VAL.BATCH_SIZE = 1
    dataset_loader = get_dataset(cfg)
    total = 0
    for step, data in enumerate(dataset_loader):
        vid = data[0][0]
        frames = unorm(vid.permute(1,0,2,3))
        label = data[1].item()
        success = attack_single_vid(frames, label, os.path.join(save_dir, f'{step}-{str(label)}'))
        total+=success
        print('success:', total)
    

    
