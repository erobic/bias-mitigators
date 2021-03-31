import torch


def build_bias_retriever(bias_variable_name):
    if bias_variable_name == 'color':
        return ColorRetriever()
    else:
        return VariableRetriever(bias_variable_name)


class ColorRetriever:

    def retrieve(self, batch):
        x = batch['x']
        return x.view(x.size(0), x.size(1), -1).max(2)[0]

    def __call__(self, batch, main_out):
        return self.retrieve(batch)


class VariableRetriever:
    def __init__(self, var_name):
        self.var_name = var_name

    def __call__(self, batch, main_out):
        if self.var_name in batch:
            ret = batch[self.var_name]
        else:
            ret = main_out[self.var_name]
        if isinstance(ret, list):
            ret = torch.FloatTensor(ret).cuda()
        return ret
