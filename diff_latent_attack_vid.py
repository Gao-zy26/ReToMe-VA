import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import os
from tqdm import tqdm
from torch import optim
from utils import register_attention_control
from attentionControl import EmptyControl
import retome

def encoder(image, model, res=224):
    if image.shape[2] != res:
        image = nnf.interpolate(image, res)
    generator = torch.Generator().manual_seed(8888)
    norm_image = 2.0 * image - 1.0
    gpu_generator = torch.Generator(device=model.device)
    gpu_generator.manual_seed(generator.initial_seed())
    return 0.18215 * model.vae.encode(norm_image).latent_dist.sample(generator=gpu_generator)

def init_uc(model, batch_size):
    max_length = 77
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    return uncond_embeddings

@torch.no_grad()
def ddim_reverse_sample(images, model, num_inference_steps: int = 20, res=224):
    batch_size = images.shape[0]
    images = images.to(model.device)
    uc = init_uc(model, batch_size).to(model.device)
    model.scheduler.set_timesteps(num_inference_steps)

    latents = encoder(images, model, res).to(model.device, model.dtype)
    timesteps = model.scheduler.timesteps.flip(0) 

    all_latents = [latents]

    for t in tqdm(timesteps[:-1], desc="DDIM_inverse", disable=True):
        noise_pred = model.unet(latents, t, encoder_hidden_states=uc)["sample"]

        next_timestep = t + model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
        alpha_bar_next = model.scheduler.alphas_cumprod[next_timestep] \
            if next_timestep <= model.scheduler.config.num_train_timesteps else torch.tensor(0.0)

        reverse_x0 = (1 / torch.sqrt(model.scheduler.alphas_cumprod[t]) * (
                latents - noise_pred * torch.sqrt(1 - model.scheduler.alphas_cumprod[t])))

        latents = reverse_x0 * torch.sqrt(alpha_bar_next) + torch.sqrt(1 - alpha_bar_next) * noise_pred

        all_latents.append(latents.cpu())

    return latents, all_latents

def diffusion_step2img(model, latents, context, t):
    B,C,H,W = latents.shape
    batch_size = B // 2
    latents_input = latents.to(model.device)
    noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
    latents = model.scheduler.step(noise_pred, t, latents)
    out_image = model.vae.decode(1 / 0.18215 * latents["pred_original_sample"][batch_size:])['sample']
    out_image = (out_image / 2 + 0.5).clamp(0, 1)
    
    return out_image, latents["prev_sample"][batch_size:]

def latent2image(vae, latents, return_type='np'):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    if return_type == 'np':
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).astype(np.uint8)
    return image

def make_vid(test_image):
    """frames to vid and norm"""
    test_image = test_image.permute(0, 2, 3, 1)
    mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=test_image.dtype, device=test_image.device)
    std = torch.as_tensor([0.229, 0.224, 0.225], dtype=test_image.dtype, device=test_image.device)
    test_image = test_image[:, :, :].sub(mean).div(std)
    test_image = test_image.permute(0, 3, 1, 2)
    vid = test_image.permute(1,0,2,3).unsqueeze(0)
    return vid


@torch.no_grad()
def attack_per_step(
        model,
        images, 
        label,
        controller,
        classifier, 
        model_name="slow_50",
        save_path="",
        verbose=True,
        args=None
):
    num_inference_steps = args.diffusion_steps
    start_step = args.start_step
    iterations = args.iterations
    res = args.res
    device = model.device
    label = torch.Tensor([label]).long().to(device)

    model.vae.requires_grad_(False)
    model.text_encoder.requires_grad_(False)
    model.unet.requires_grad_(False)
    classifier.requires_grad_(False)

    height = width = res
    test_image = images

    if args.retome:
        merge_ratio = args.merge_ratio
        merge_to = 1

    vid = make_vid(test_image.clone()).to(device)
    pred = classifier(vid)
    print('before attack, pred vs label:', torch.argmax(pred, 1).detach().item(), label.item())
    del vid, pred
    
    success = False
    if args.retome:
        retome.apply_patch(model, align_batch=True, merge_ratio=merge_ratio, batch_size=1, merge_to=merge_to)

    with torch.no_grad():
        latents, inversion_latents = ddim_reverse_sample(images, model, num_inference_steps, res=height)
    inversion_latents = inversion_latents[::-1]
    batch_size = images.shape[0]
    uncond_embeddings = init_uc(model, 1) # attack single vid clip
    register_attention_control(model, controller)
    cross_entro = torch.nn.CrossEntropyLoss()
    controller.loss = 0
    torch.cuda.empty_cache()
    controller.reset()
    latent = inversion_latents[start_step - 1].to(device)
    latent_prev = latent
    if args.retome:
        print('use retome!')
        retome.apply_patch(model, align_batch=True, merge_ratio=merge_ratio, merge_to=merge_to) 

    with torch.enable_grad():
        for ind, t in enumerate(model.scheduler.timesteps[1 + start_step - 1:]):
            latent = latent_prev.detach()
            latent.requires_grad_(True)
            original_latent = inversion_latents[start_step - 1 + ind].to(device)
            optimizer = optim.AdamW([latent], lr=1e-2)
            pbar = tqdm(range(iterations+ind*2+1), desc="Iterations")
            for pidx, _ in enumerate(pbar):
                latents = torch.cat([original_latent, latent])
                out_image, latent_prev = diffusion_step2img(
                                        model, 
                                        latents, 
                                        torch.cat([uncond_embeddings.expand(batch_size, *uncond_embeddings.shape[1:])] * 2),
                                        t)
                
                vid = make_vid(out_image).to(device)
                pred = classifier(vid)
                p_label = pred.argmax(1)
                if p_label != label:
                    success = True
                attack_loss = - cross_entro(pred, label) * args.attack_loss_weight
                self_attn_loss = controller.loss * args.self_attn_loss_weight
                loss = self_attn_loss + attack_loss
                if verbose:
                    pbar.set_postfix_str(
                        f"at_loss: {attack_loss:.3f} "
                        f"self_loss: {self_attn_loss.item():.3f} "
                        )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                controller.loss = 0
                controller.reset()


    register_attention_control(model, EmptyControl())
    image = latent2image(model.vae, latent_prev.detach()) 
    perturbed = image.astype(np.float32) / 255
    if args.retome:
        retome.remove_patch(model)
    return perturbed, success, None
