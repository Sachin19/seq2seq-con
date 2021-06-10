import torch
import scipy.special
import numpy as np
from torch.autograd import Variable

from onmt.utils.logging import logger

eps = np.finfo("float64").eps
# def ratio(v, z):
#     # return z/(v-0.5+torch.sqrt((v-0.5)*(v-0.5) + z*z))
#     return z/(v-1+torch.sqrt((v+1)*(v+1) + z*z))


class Logcmk(torch.autograd.Function):
    """
    The exponentially scaled modified Bessel function of the first kind
    """

    @staticmethod
    def forward(ctx, k, m=300):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        # m = 300
        ctx.m = m
        ctx.save_for_backward(k)

        k = k.double()
        k_cpu = k.data.cpu().numpy()
        logarg = scipy.special.ive(m / 2 - 1, k_cpu)#np.maximum(scipy.special.ive(m / 2 - 1, k_cpu) - eps, 0)  + eps
        # answer = (m/2-1)*torch.log(k) - torch.log(scipy.special.ive(m/2-1, k)).cuda() - k - (m/2)*np.log(2*np.pi)
        answer = (
            (m / 2 - 1) * torch.log(k)
            - torch.log(torch.from_numpy(logarg)).to(
                k.device
            )
            - k
            - (m / 2) * np.log(2 * np.pi)
        )
        answer = answer.float()
        return answer

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        (k,) = ctx.saved_tensors
        m = ctx.m
        # m = 300
        # x = -ratio(m/2, k)
        k = k.double()
        k_cpu = k.data.cpu().numpy()
        den = scipy.special.ive(m / 2 - 1, k_cpu)#np.maximum(scipy.special.ive(m / 2 - 1, k_cpu)- eps, 0) + eps
        x = -(
            torch.from_numpy(scipy.special.ive(m / 2, k_cpu))
            / torch.from_numpy(den)
        ).to(k.device)
        # x = -((scipy.special.ive(m/2, k))/(scipy.special.ive(m/2-1,k))).cuda()
        x = x.float()

        return grad_output * x, None


logcmk = Logcmk.apply
