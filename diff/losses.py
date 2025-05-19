import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import graph_lib
from model import utils as mutils



def get_loss_fn(noise, graph, train, sampling_eps=1e-3, lv=False, pad_token_id=None):
    def loss_fn(model, batch, cond=None, t=None, perturbed_batch=None):
        """
        Batch shape: [B, L] int. D given from graph
        """

        if t is None:
            if lv:
                raise NotImplementedError("Yeah I gotta do this later")
            else:
                t = (1 - sampling_eps) * torch.rand(batch.shape[0], device=batch.device) + sampling_eps
            
        sigma, dsigma = noise(t)
        
        # Add stability check
        # if torch.any(sigma < 1e-5):
        #     print("Warning: Very small sigma detected")
        #     sigma = torch.clamp(sigma, min=1e-5)
        
        if perturbed_batch is None:
            perturbed_batch = graph.sample_transition(batch, sigma[:, None])

        # Create attention mask for pad tokens
        attention_mask = None
        if pad_token_id is not None:
            # print("entering attentionmask at attentionmask computation")
            # from remote_pdb import RemotePdb; RemotePdb('127.0.0.1', 4444).set_trace()
            attention_mask = (batch != pad_token_id).float()
            # print(f"atten mask in loss_fn is {attention_mask}")

        log_score_fn = mutils.get_score_fn(model, train=train, sampling=False)
        
        
         
        # Pass attention mask to model through score function
        log_score = log_score_fn(perturbed_batch, sigma, attention_mask=attention_mask)
        
        
        # print("this is the output of log_score", log_score)
        
        # if torch.isnan(log_score).any():
        #     print("NaN at log_score\n")
        
        loss = graph.score_entropy(log_score, sigma[:, None], perturbed_batch, batch)

        # if torch.isnan(loss).any():
        #     print("NaN at loss_0\n")
        
        # Mask out loss for pad tokens - TODO uncomment this
        if attention_mask is not None:
            loss = loss * attention_mask

        loss = (dsigma[:, None] * loss).sum(dim=-1)

        # if torch.isnan(loss).any():
        #     print("NaN at loss_1\n")

        
        # Average loss only over non-pad tokens
        if attention_mask is not None:
            # print("entering attentionmask at loss computation")
            loss = loss.sum() / attention_mask.sum() # - TODO uncomment this 
            # loss = loss.mean()
        else:
            loss = loss.mean()

        # if torch.isnan(loss).any():
        #     print("NaN at loss_2\n")
        
        return loss

    return loss_fn



# def get_loss_fn(noise, graph, train, sampling_eps=1e-3, lv=False): # modified by foobar

#     def loss_fn(model, batch, cond=None, t=None, perturbed_batch=None):
#         """
#         Batch shape: [B, L] int. D given from graph
#         """

#         if t is None:
#             if lv:
#                 raise NotImplementedError("Yeah I gotta do this later")
#             else:
#                 t = (1 - sampling_eps) * torch.rand(batch.shape[0], device=batch.device) + sampling_eps
            
#         sigma, dsigma = noise(t)
        
#         if perturbed_batch is None:
#             perturbed_batch = graph.sample_transition(batch, sigma[:, None])

#         log_score_fn = mutils.get_score_fn(model, train=train, sampling=False)
#         log_score = log_score_fn(perturbed_batch, sigma)
#         loss = graph.score_entropy(log_score, sigma[:, None], perturbed_batch, batch)

#         loss = (dsigma[:, None] * loss).sum(dim=-1)

#         return loss

#     return loss_fn


def get_optimizer(config, params):
    if config.optim.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'AdamW':
        optimizer = optim.AdamW(params, lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!')

    return optimizer


def optimization_manager(config):
    """Returns an optimize_fn based on `config`."""

    def optimize_fn(optimizer, 
                    scaler, 
                    params, 
                    step, 
                    lr=config.optim.lr,
                    warmup=config.optim.warmup,
                    grad_clip=config.optim.grad_clip):
        """Optimizes with warmup and gradient clipping (disabled if negative)."""
        scaler.unscale_(optimizer)
        
        # total_norm = 0.
        # first_nan = True  # Flag to catch first NaN

        # # Get model from optimizer's parameter groups
        # model = None
        # for group in optimizer.param_groups:
        #     for param in group['params']:
        #         # Find the model by checking parameters
        #         for name, potential_model in optimizer.state.items():
        #             if isinstance(potential_model, torch.nn.Module):
        #                 model = potential_model
        #                 break
        #         if model is not None:
        #             break
        #     if model is not None:
        #         break

        # # Create parameter to name mapping
        # param_to_name = {}
        # if model is not None:
        #     for name, param in model.named_parameters():
        #         param_to_name[param] = name

        # for p in params:
        #     if p.grad is not None:
        #         # Check for NaN in gradients
        #         if torch.isnan(p.grad).any():
        #             if first_nan:
        #                 print("\n" + "="*50)
        #                 print("Found first NaN gradient!")
                        
        #                 # Try to get parameter name
        #                 param_name = param_to_name.get(p, "Unknown parameter")
        #                 print(f"Parameter name: {param_name}")
        #                 print(f"Parameter shape: {p.shape}")
                        
        #                 print("\nGradient stats:")
        #                 print(f"- Number of NaNs: {torch.isnan(p.grad).sum().item()}")
        #                 print(f"- Total elements: {p.grad.numel()}")
        #                 print(f"- Percentage of NaNs: {100 * torch.isnan(p.grad).sum().item() / p.grad.numel():.2f}%")
                        
        #                 # Print non-NaN gradient statistics if any exist
        #                 valid_grads = p.grad[~torch.isnan(p.grad)]
        #                 if len(valid_grads) > 0:
        #                     print(f"- Valid gradient range: [{valid_grads.min().item():.4f}, {valid_grads.max().item():.4f}]")
        #                     print(f"- Valid gradient mean: {valid_grads.mean().item():.4f}")
                        
        #                 # Get positions of some NaNs
        #                 nan_indices = torch.where(torch.isnan(p.grad))
        #                 if len(nan_indices[0]) > 0:
        #                     print("\nFirst few NaN positions:", 
        #                           [tuple(idx[:3].tolist()) for idx in nan_indices])
                        
        #                 # If it's a 2D parameter (like weight matrix), print more details
        #                 if len(p.shape) == 2:
        #                     print(f"\nParameter appears to be a weight matrix:")
        #                     print(f"- Input dimension: {p.shape[1]}")
        #                     print(f"- Output dimension: {p.shape[0]}")
                            
        #                     # Check if NaNs are concentrated in certain rows/columns
        #                     nan_rows = torch.isnan(p.grad).any(dim=1).sum()
        #                     nan_cols = torch.isnan(p.grad).any(dim=0).sum()
        #                     print(f"- Rows containing NaNs: {nan_rows}")
        #                     print(f"- Columns containing NaNs: {nan_cols}")
                            
        #                 print("="*50 + "\n")
        #                 # import pdb; pdb.set_trace()
        #                 first_nan = False
                
        #         param_norm = p.grad.data.norm(2)
        #         if not torch.isnan(param_norm):  # Only add to total if not NaN
        #             total_norm += param_norm.item() ** 2

        # if not torch.isnan(torch.tensor(total_norm)):
        #     total_norm = total_norm ** (1. / 2)
        #     print("Gradient norm:", total_norm)
        # else:
        #     print("Gradient norm is NaN")
        
        
        
        
#         """
#         total_norm = 0.
#         first_nan = True  # Flag to catch first NaN

#         def get_param_info(param):
#             """Get detailed information about a parameter's location in the model"""
#             if param.grad_fn is None:
#                 return "Parameter has no grad_fn"
            
#             # Get the complete backward graph name
#             grad_fn_str = str(param.grad_fn)
            
#             # Try to identify the layer/block
#             if 'block' in grad_fn_str.lower():
#                 block_info = "Block detected in: " + grad_fn_str
#             else:
#                 block_info = "No block info found"
                
#             # Try to identify the operation
#             if 'attention' in grad_fn_str.lower():
#                 op_type = "Attention Layer"
#             elif 'mlp' in grad_fn_str.lower():
#                 op_type = "MLP Layer"
#             elif 'norm' in grad_fn_str.lower():
#                 op_type = "Normalization Layer"
#             elif 'embedding' in grad_fn_str.lower():
#                 op_type = "Embedding Layer"
#             else:
#                 op_type = "Other Layer"
                
#             return f"""
# Parameter Info:
# - Operation Type: {op_type}
# - Block Info: {block_info}
# - Full grad_fn: {grad_fn_str}
# """

#         for p in params:
#             if p.grad is not None:
#                 # Check for NaN in gradients
#                 if torch.isnan(p.grad).any():
#                     if first_nan:
#                         print("\n" + "="*50)
#                         print("Found first NaN gradient!")
#                         print("Parameter shape:", p.shape)
#                         print(get_param_info(p))
#                         print("Gradient stats:")
#                         print("- Number of NaNs:", torch.isnan(p.grad).sum().item())
#                         print("- Total elements:", p.grad.numel())
#                         print("- Percentage of NaNs: {:.2f}%".format(
#                             100 * torch.isnan(p.grad).sum().item() / p.grad.numel()
#                         ))
                        
#                         # Print non-NaN gradient statistics if any exist
#                         valid_grads = p.grad[~torch.isnan(p.grad)]
#                         if len(valid_grads) > 0:
#                             print("- Valid gradient range: [{:.4f}, {:.4f}]".format(
#                                 valid_grads.min().item(), 
#                                 valid_grads.max().item()
#                             ))
#                             print("- Valid gradient mean: {:.4f}".format(
#                                 valid_grads.mean().item()
#                             ))
                        
#                         # Get positions of some NaNs
#                         nan_indices = torch.where(torch.isnan(p.grad))
#                         if len(nan_indices[0]) > 0:
#                             print("\nFirst few NaN positions:", 
#                                   [tuple(idx[:3].tolist()) for idx in nan_indices])
                            
#                         print("="*50 + "\n")
#                         import pdb; pdb.set_trace()
#                         first_nan = False
                
#                 param_norm = p.grad.data.norm(2)
#                 if not torch.isnan(param_norm):  # Only add to total if not NaN
#                     total_norm += param_norm.item() ** 2

#         if not torch.isnan(torch.tensor(total_norm)):
#             total_norm = total_norm ** (1. / 2)
#             print("Gradient norm:", total_norm)
#         else:
#             print("Gradient norm is NaN")
        
#         """
        
        
        
        # # Check gradient norms
        # total_norm = 0.
        # for p in params:
        #     if p.grad is not None:
        #         param_norm = p.grad.data.norm(2)
        #         total_norm += param_norm.item() ** 2
        # total_norm = total_norm ** (1. / 2)
        # print("Gradient norm:", total_norm)
        
        
        if warmup > 0:
            for g in optimizer.param_groups:
                g['lr'] = lr * np.minimum(step / warmup, 1.0)
        if grad_clip >= 0:
            # print("entering clipping")
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)

        scaler.step(optimizer)
        scaler.update()

    return optimize_fn


# def get_step_fn(noise, graph, train, optimize_fn, accum): # modified by foobar
def get_step_fn(noise, graph, train, optimize_fn, accum, config):
    loss_fn = get_loss_fn(noise, graph, train, pad_token_id=config.pad_token_id)

    accum_iter = 0
    total_loss = 0

    def step_fn(state, batch, cond=None):
        nonlocal accum_iter 
        nonlocal total_loss

        model = state['model']

        if train:
            # print("currently in training mode")
            optimizer = state['optimizer']
            scaler = state['scaler']
            loss = loss_fn(model, batch, cond=cond).mean() / accum
            
            scaler.scale(loss).backward()

            accum_iter += 1
            total_loss += loss.detach()
            if accum_iter == accum:
                accum_iter = 0

                state['step'] += 1
                optimize_fn(optimizer, scaler, model.parameters(), step=state['step'])
                state['ema'].update(model.parameters())
                optimizer.zero_grad()
                
                loss = total_loss
                total_loss = 0
        else:
            # print("currently in evaluation mode")
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                loss = loss_fn(model, batch, cond=cond).mean()
                ema.restore(model.parameters())

        return loss

    return step_fn