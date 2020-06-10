# Adapted from https://github.com/fungtion/DANN_py3/blob/master/functions.py

from torch.autograd import Function


class GradReverseLayerFunction(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


