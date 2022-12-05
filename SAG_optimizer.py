# stochasticity adaptiev optimizer

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
# then in the mean,      E[(<f'>-f')^2 / \alpha^2] ~= K^2
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
from torch.optim import Optimizer
from torch import Tensor
from typing import List, Optional

def erf_approx(x: torch.Tensor):
    
    """
    approximation of erf function.
    see https://math.stackexchange.com/questions/321569/approximating-the-error-function-erf-by-analytical-functions
    """
    return torch.tanh(math.sqrt(math.pi) * math.log(2) * x)


def sag(params: List[Tensor],
        grads: List[Tensor],
        grads_prev: List[Tensor],
        ema_s_vars: List[Tensor],
        ema_avgs: List[Tensor],
        ema_vars: List[Tensor],
        state_steps: List[int],
        *,
        beta1: float,
        beta2: float,
        lr: float,
        weight_decay: float,
        tau: float,
        B: int,
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
        step = state_steps[i]
        grad_diff = grad.sub(ema_avg)
        grad_s_diff = grad.sub(grad_prev)

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # mean and variance running averages
        ema_avg.mul_(beta1).add_(grad, alpha=1 - beta1).div_(bias_correction1)
        # estimate stochastic noise
        ema_var.mul_(beta2).addcmul_(grad_diff, grad_diff.conj(), value=1 - beta2)
        ema_s_var.mul_(beta2).addcmul_(grad_s_diff, grad_s_diff.conj(), value=1 - beta2)
        ema_var.div_(bias_correction2)
        ema_s_var.div_(bias_correction2)

        # estimate curvature
        step_size = lr * tau
        k = ema_var.sqrt() / step_size
        p = erf_approx(k / (2 * ema_s_var.sqrt())) # probability that -2 * ema_s_var < <k> - k < 2 * ema_s_var
        l = torch.exp(-k/step_size)
        k_eff = (k * (1-l)+ step_size * l) # >= k

        snr = ema_avg.abs().div(100 * ema_s_var.sqrt())
        noise_factor = torch.tanh(snr / tau)
        curvature_factor = (step_size * (1-p) + k_eff * p)
        param.addcdiv_(ema_avg * noise_factor, curvature_factor, value=-step_size)
        # param.addcmul_(ema_avg * noise_factor, curvature_factor, value=-step_size)

class SAG(Optimizer):

    def __init__(self, params, B, lr=1e-3, betas=(0.9, 0.999), tau=1,
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
        self.B = B
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
                ema_avgs.append(state['ema_avg'])
                ema_vars.append(state['ema_var'])
                ema_s_vars.append(state['ema_s_var'])
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
                   state_steps,
                   beta1=beta1,
                   beta2=beta2,
                   lr=group['lr'],
                   weight_decay=group['weight_decay'],
                   tau=group['tau'],
                   B=self.B, # batch size
                   maximize=group['maximize'])

        # clean up for next step
        self.prev_grads = None

        return loss