# stochasticity adaptive optimizer

# seperate the gradient noise into two parts - stochastic noise due to random batch and loss curvature
# estimate the stochastic noise by the difference in gradients between batches, by calculating the gradients at a single point with two different batches
# N_s = g_1 - g_2
# estimate the variance due to loss curvature by calculating the total variance and subtracting the variance due to stochasticity
# all variances are calculated using moving averages
# use the stochastic variance to determine the EMA coefficient for the curvature variance
# initialize the loss variance at a high initial value

# note on estimating the stochastic noise:
# ===================================
# If the variance is \sigma_s then the variance for a batch of size B would be \sigma_s / sqrt(B)
# therefore to estimate the variance we can subdivide the batch into two of size B/2 and calculate the moving average of the square difference from the mutual mean
# (g1-g2)^2 ~= 2*\sigma_s^2 / B ===> \sigma_s ~= sqrt(B/2) * |g1-g2|

# note on estimating the curvature
# ===================================
# absent stochastic noise, the curvature is approximated (when it is small) by the second derivative: k~=f''
# in the presence of stochastic noise calculating an approximate second derivative will give noise of variance sqrt(2) * \sigma_s / dw
# f'' (in practive) ~ N(f'', sqrt(2) * \sigma_s / sqrt(B) / \alpha)
# the absolute value of the second derivative can be approximated using the EMA of (<f'>-f')^2 divided by \alpha^2 step size
# then in the mean,  E[(<f'>_t - f'_(t+1))^2 / \alpha^2] = E[(<f'>- /mu_t + /mu_t - f'_(t+1))^2 / \alpha^2] ~= d\mu^2 / \alpha^2 + K^2 ~ \sigma_s^2 / \alpha^2 + K^2
# with a variance of:    2 * \sigma_s^2 / B / \alpha^2
# The probability of having 
# || E[(<f'>-f')^2 / \alpha^2] ||< 2*sqrt(2) * 2 * \sigma_s^2 / B / \alpha^2
# is approx 1-5e-3 (using table of erf function)
# Therefore if > 2*sqrt(2) * 2 * \sigma_s^2 / B / \alpha^2
# then we can say the curvature is > 0 with very good confidence and we can rely on the value
# when the variance is very small we are able to measure small curvatures
# E[(<f'>-f')^2 / \alpha^2]


# the aim is to adapt the learning rate per param as:
# small stochastic noise \sigma_s -> higher lr     <<< learn in reliable directions and gather information to make it reliable (EMA)
# small curvature K (large radius of curvature) -> smaller lr   <<< stay in wide minima
# large gradient (momentum) \mu -> large lr <<< go where the gradient says

# idea 1:
# dw ~ \mu * (\alpha * (1-p) + <K> * p )  *  tanh(\mu / \sigma_s / \tau)
#                  ^                                     ^
#            K << \alpha => \alpha,         small SNR => 0,    ; SNR = \mu / \sigma_s
#            large K => K                   large SNR => 1
# <K> = (K * (1-exp(-K))+\alpha*exp(-K))   note that <K> >= K
# p = erf(K / 2 / \sigma_s) ~= tanh(sqrt(pi)log(2) * K / 2 / \sigma_s)
# see https://math.stackexchange.com/questions/321569/approximating-the-error-function-erf-by-analytical-functions
# \tau -> "temperature"

import math
import torch
from torch.nn import functional as F
from torch.optim import Optimizer
from torch import Tensor
from typing import List, Optional

def erf_approx(x: torch.Tensor):
    
    """
    approximation of erf function.
    see https://math.stackexchange.com/questions/321569/approximating-the-error-function-erf-by-analytical-functions
    """
    return torch.tanh(math.sqrt(math.pi) * math.log(2) * x)

def smoothstep(x: torch.Tensor, edge0: float, edge1: float):
   # convert [edge0,edge1] to [-1,1] range
   x = (x-edge0)/(edge1-edge0) # [0, 1]
   x = torch.clamp(x, 0, 1)
   return x * x * (3 - 2 * x) # apply hermite interpolation in middle region.


def sag(params: List[Tensor],
        grads: List[Tensor],
        grads_prev: List[Tensor],
        ema_s_vars: List[Tensor],
        ema_avgs: List[Tensor],
        ema_vars: List[Tensor],
        curvatures: List[Tensor],
        adaptation_factors: List[Tensor],
        state_steps: List[int],
        *,
        beta1: float,
        beta2: float,
        lr: float,
        weight_decay: float,
        tau: float,
        maximize: bool):
    # any_nans = any([x.isnan().any() for x in ema_s_vars])
    # print(f"ema_s_vars has nans: {any_nans}")
    # print(max([v.norm() for v in ema_s_vars]))
    for i, param in enumerate(params):

        grad = grads[i] if not maximize else -grads[i]
        grad_prev = grads_prev[i] if not maximize else -grads_prev[i]
        ema_avg = ema_avgs[i]
        ema_var = ema_vars[i]
        ema_s_var = ema_s_vars[i]
        curvature = curvatures[i]
        adaptation_factor = adaptation_factors[i]
        
        step = state_steps[i]
        ## TODO: Test the idea that the bigger the ema_avg the faster the vars should change
        # beta2 = 2*(torch.sigmoid(ema_avg.abs() / ema_var.sqrt()) -1/2)
        #
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        # estimate stochastic noise
        grad_s_diff = grad.sub(grad_prev)
        ema_s_var.mul_(beta2).addcmul_(grad_s_diff, grad_s_diff.conj(), value=1 - beta2)
        ema_s_var.div_(bias_correction2)

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # mean and variance running averages
        ema_avg.mul_(beta1).add_(grad, alpha=1 - beta1).div_(bias_correction1)
        # estimate curvature
        grad_diff = grad.sub(ema_avg)
        ema_var.mul_(beta2).addcmul_(grad_diff, grad_diff.conj(), value=1 - beta2)
        ema_var.div_(bias_correction2)
        
        step_size = lr * ema_avg.abs()
        # K = [ema_var - ema_s_var].sqrt() / step_size
        torch.div(ema_var.sub(ema_s_var).relu().sqrt(), step_size, out=curvature)
        p = erf_approx(curvature / (2 * ema_s_var.sqrt()+1e-16)) # probability that -2 * ema_s_var < <k> - k < 2 * ema_s_var
        # l = torch.exp(-(ema_avg.abs() / curvature) / step_size)
        # k_eff = (curvature * l+ 1/ lr * (1-l)) # >= k
        # curvature_factor = (1 / lr * (1-p) + k_eff * p)
        curvature_factor = (curvature * p + 1/ lr * (1-p)) # >= k

        snr = ema_avg.abs().div(ema_s_var.sqrt()+1e-16)
        noise_factor = torch.tanh(snr / tau)
        
        torch.div(noise_factor, curvature_factor, out=adaptation_factor)
        # if torch.isnan(adaptation_factor).any():
        #     print("detected nans")
        param.addcdiv_(ema_avg * noise_factor, curvature_factor, value=-1)

def sag_no_curvature(params: List[Tensor],
        grads: List[Tensor],
        grads_prev: List[Tensor],
        ema_s_vars: List[Tensor],
        ema_avgs: List[Tensor],
        state_steps: List[int],
        *,
        beta_min: float,
        beta_max: float,
        lr: float,
        weight_decay: float,
        tau: float,
        maximize: bool):
    # any_nans = any([x.isnan().any() for x in ema_s_vars])
    # print(f"ema_s_vars has nans: {any_nans}")
    # print(max([v.norm() for v in ema_s_vars]))
    for i, param in enumerate(params):

        grad = grads[i] if not maximize else -grads[i]
        grad_prev = grads_prev[i] if not maximize else -grads_prev[i]
        ema_avg = ema_avgs[i]
        ema_s_var = ema_s_vars[i]
        
        step = state_steps[i]
        
        # estimate stochastic noise. inspired by KAMA https://school.stockcharts.com/doku.php?id=technical_indicators:kaufman_s_adaptive_moving_average
        grad_s_diff = grad.sub(grad_prev)
        
        bias_correction2 = 1 - beta_max ** step
        ema_s_var.mul_(beta_max).addcmul_(grad_s_diff, grad_s_diff.conj(), value=1 - beta_max).div_(bias_correction2)

        # if weight_decay != 0:
            # grad = grad.add(param, alpha=weight_decay)

        # mean and variance running averages
        volatility_ratio = torch.clamp(torch.sub(grad, ema_avg).abs().div(ema_s_var.sqrt() / (1-beta_max)+1e-16), 0, 1) # change / volatility
        beta_eff = (volatility_ratio*(beta_max-beta_min)+beta_min) # change var faster if incoming value is big
        bias_correction = 1 - beta_eff ** step
        ema_avg.mul_(beta_eff).add_(grad*(1 - beta_eff)).div_(bias_correction)
        
        step_size = lr

        # snr = ema_avg.abs().div(ema_s_var.sqrt()+1e-16)
        # noise_factor = smoothstep(snr, 3, 4) # snr needs to be at least 3

        noise_factor = 1-smoothstep(volatility_ratio, 0.5, 0.6) # step in the direction if it is not changing rapidly now
        # if torch.isnan(adaptation_factor).any():
        #     print("detected nans")
        param.addcmul_(ema_avg, noise_factor, value=-step_size)

class SAG(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), tau=1,
                 weight_decay=0, *, maximize: bool = False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= tau:
            raise ValueError("Invalid tau value: {}".format(tau))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, tau=tau,
                        weight_decay=weight_decay, maximize=maximize)
        super(SAG, self).__init__(params, defaults)
        self.prev_grads = None
        

    def __setstate__(self, state):
        super(SAG, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('maximize', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        if self.prev_grads is None:
            # gather gradients and wait for next step
            self.prev_grads = []
            for group_idx, group in enumerate(self.param_groups):
                self.prev_grads.append([])
                for p in group['params']:
                    p: torch.nn.parameter.Parameter
                    if p.grad is not None:
                        if p.grad.is_sparse:
                            raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                        self.prev_grads[-1].append(p.grad.clone().detach())
                    else:
                        self.prev_grads[-1].append(None)
            return loss

        for group_idx, group in enumerate(self.param_groups):
            params_with_grad = []
            grads = []
            ema_avgs = []
            ema_s_vars = []
            ema_vars = []
            curvature = []
            adaptation_factor = []
            state_steps = []
            beta1, beta2 = group['betas']

            for p_idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                if self.prev_grads[group_idx][p_idx] is None:
                    continue

                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                grads.append(p.grad)
                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['ema_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of gradient variance values
                    state['ema_var'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['ema_s_var'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['curvature'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['adaptation_factor'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                ema_avgs.append(state['ema_avg'])
                ema_vars.append(state['ema_var'])
                ema_s_vars.append(state['ema_s_var'])
                curvature.append(state['curvature'])
                adaptation_factor.append(state['adaptation_factor'])
                # update the steps for each param group update
                state['step'] += 1
                # record the step after step update
                state_steps.append(state['step'])

            sag(params_with_grad,
                   grads,
                   self.prev_grads[group_idx],
                   ema_s_vars,
                   ema_avgs,
                   ema_vars,
                   curvature,
                   adaptation_factor,
                   state_steps,
                   beta1=beta1,
                   beta2=beta2,
                   lr=group['lr'],
                   weight_decay=group['weight_decay'],
                   tau=group['tau'],
                   maximize=group['maximize'])

        # clean up for next step
        self.prev_grads = None

        return loss

class SAG_NoCurvature(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), tau=1,
                 weight_decay=0, *, maximize: bool = False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= tau:
            raise ValueError("Invalid tau value: {}".format(tau))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, tau=tau,
                        weight_decay=weight_decay, maximize=maximize)
        super(SAG_NoCurvature, self).__init__(params, defaults)
        self.prev_grads = None
        

    def __setstate__(self, state):
        super(SAG, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('maximize', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        if self.prev_grads is None:
            # gather gradients and wait for next step
            self.prev_grads = []
            for group_idx, group in enumerate(self.param_groups):
                self.prev_grads.append([])
                for p in group['params']:
                    p: torch.nn.parameter.Parameter
                    if p.grad is not None:
                        if p.grad.is_sparse:
                            raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                        self.prev_grads[-1].append(p.grad.clone().detach())
                    else:
                        self.prev_grads[-1].append(None)
            return loss

        for group_idx, group in enumerate(self.param_groups):
            state_keys = []
            params_with_grad = []
            grads: List[torch.Tensor] = []
            ema_avgs: List[torch.Tensor] = []
            ema_s_vars: List[torch.Tensor] = []
            state_steps = []
            beta1, beta2 = group['betas']

            for p_idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                if self.prev_grads[group_idx][p_idx] is None:
                    continue

                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                grads.append(p.grad)
                state = self.state[p]
                state_keys.append(p)
                # Lazy state initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['ema_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of gradient variance values
                    # state['ema_s_var'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['ema_s_var'] = 0 * torch.ones_like(p, memory_format=torch.preserve_format)
                ema_avgs.append(state['ema_avg'])
                ema_s_vars.append(state['ema_s_var'])
                # update the steps for each param group update
                state['step'] += 1
                # record the step after step update
                state_steps.append(state['step'])

            sag_no_curvature(params_with_grad,
                   grads,
                   self.prev_grads[group_idx],
                   ema_s_vars,
                   ema_avgs,
                   state_steps,
                   beta_min=beta1,
                   beta_max=beta2,
                   lr=group['lr'],
                   weight_decay=group['weight_decay'],
                   tau=group['tau'],
                   maximize=group['maximize'])
            
            # update auxilary state variables meant for debugging
            for p, g, mu, sigma in zip(state_keys, grads, ema_avgs, ema_s_vars):
                state = self.state[p]
                state["residual"] = g-mu

        # clean up for next step
        self.prev_grads = None

        return loss