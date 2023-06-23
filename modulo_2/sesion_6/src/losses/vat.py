import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(ConditionalEntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = b.sum(dim=1)
        return -1.0 * b.mean(dim=0)


class VAT(nn.Module):
    def __init__(self, network_f, network_h, radius=3.5, n_power=1):
        super(VAT, self).__init__()
        self.n_power = n_power # number of gradient iterations
        self.radius = radius
        self.network_f = network_f
        self.network_h = network_h
        self.epsilon = 3.5

    def forward(self, x, logit):
        vat_loss = self.virtual_adversarial_loss(x, logit)
        return vat_loss

    def virtual_adversarial_loss(self, x, logit):
        # generate perturbation based on a radius
        # r_vadv contains a tensor, which is an adversarial perturbation
        r_vadv = self.generate_virtual_adversarial_perturbation(x, logit)

        logit_p = logit.detach()  # for backpropagating the gradient
        # get predictions for perturbated samples
        logit_m = self.network_h(self.network_f(x + r_vadv))
        # apply Kullback Leibler divergence
        loss = self.kl_divergence(logit_p, logit_m)
        return loss

    def generate_virtual_adversarial_perturbation(self, x, logit):
        # generate a gaussian perturbation
        d = torch.randn_like(x, device='cuda')

        for _ in range(self.n_power):
            # L2-normalization vector and multiply by a radius
            d = self.radius * self.get_normalized_vector(d).requires_grad_()

            # generate prediction
            logit_m = self.network_h(self.network_f(x + d))

            # [Compute adversarial perturbation]
            # KL divergence to measure difference
            dist = self.kl_divergence(logit, logit_m)
            # torch.autograd.grad: Computes and returns the sum of gradients of outputs with respect to the inputs.
            grad = torch.autograd.grad(dist, [d])[0]
            # copy the content of grad in a new tensor
            d = grad.detach()

            # L2-normalization gradients and multiply by the constant 'epsilon' (radius)
            perturbation = self.epsilon * self.get_normalized_vector(d)

        return perturbation

    def kl_divergence(self, q_logit, p_logit):
        q = F.softmax(q_logit, dim=1)
        qlogq = torch.mean(torch.sum(q * F.log_softmax(q_logit, dim=1), dim=1))
        qlogp = torch.mean(torch.sum(q * F.log_softmax(p_logit, dim=1), dim=1))
        return qlogq - qlogp

    def get_normalized_vector(self, d):
        return F.normalize(d.view(d.size(0), -1), p=2, dim=1).reshape(d.size())

