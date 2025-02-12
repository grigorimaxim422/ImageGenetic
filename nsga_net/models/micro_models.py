from models.micro_operations import *
from misc.utils import drop_path
from torch.autograd import Variable
from  torch.cuda import FloatTensor
import torch.nn.functional as F
import torch
import logging

DEFAULT_PADDINGS = {
    'none': 1,
    'skip_connect': 1,
    'avg_pool_3x3': 1,
    'max_pool_3x3': 1,
    'sep_conv_3x3': 1,
    'sep_conv_5x5': 2,
    'sep_conv_7x7': 3,
    'dil_conv_3x3': 2,
    'dil_conv_5x5': 4,
    'conv_7x1_1x7': 3,
}

class Cell3(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev, SE=False):
        super(Cell3, self).__init__()
        print(C_prev_prev, C_prev, C)

        self.se_layer = None

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

        if SE:
            self.se_layer = SELayer(channel=self.multiplier * C)
            
    def drop_path(self, x, drop_prob):        
        if drop_prob > 0.:
            keep_prob = 1. - drop_prob
            # mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
            mask = torch.bernoulli(torch.full((x.size(0), 1, 1, 1), keep_prob, device=x.device))
            x.div_(keep_prob)
            x.mul_(mask)
        return x
    
    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)
        # logging.info(f"steps={self._steps}")
        self._ops = nn.ModuleList()        
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices
        
    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        # logging.info(f"indicies={self._indices}")
        states = [s0, s1]
        #Repeat 5 times, few...
        h1 = states[self._indices[0]]
        h2 = states[self._indices[1]]
        op1 = self._ops[0]
        op2 = self._ops[1]
        h3 = op1(h1)
        h4 = op2(h2)        
        s10 = h3 + h4
        states += [s10]
        # logging.info(f"len={len(states)}")
                
        
        h11 = states[self._indices[2]]
        h12 = states[self._indices[3]]
        op11 = self._ops[2]
        op12 = self._ops[3]
        h13 = op11(h11)
        h14 = op12(h12)
        s20 = h13 + h14
        states += [s20]
        # logging.info(f"len.01={len(states)}")
        
        h21 = states[self._indices[4]]
        h22 = states[self._indices[5]]
        op21 = self._ops[4]
        op22 = self._ops[5]
        h23 = op21(h21)
        h24 = op22(h22)
        s30 = h23 + h24
        states += [s30]
        # logging.info(f"len.02={len(states)}")
        
        
        h31 = states[self._indices[6]]
        h32 = states[self._indices[7]]
        op31 = self._ops[6]
        op32 = self._ops[7]
        h33 = op31(h31)
        h34 = op32(h32)
        s40 = h33 + h34
        states += [s40]
        # logging.info(f"len.03={len(states)}")
        
        h41 = states[self._indices[8]]
        h42 = states[self._indices[9]]
        op41 = self._ops[8]
        op42 = self._ops[9]
        h43 = op41(h41)
        h44 = op42(h42)
        s50 = h43 + h44
        states += [s50]
        # logging.info(f"len.04={len(states)}")
        
        # for i in range(self._steps):
        #     h1 = states[self._indices[2 * i]]
        #     h2 = states[self._indices[2 * i + 1]]
        #     op1 = self._ops[2 * i]
        #     op2 = self._ops[2 * i + 1]
        #     h1 = op1(h1)
        #     h2 = op2(h2)
        #     if self.training and drop_prob > 0.:
        #         if not isinstance(op1, Identity):
        #             h1 = drop_path(h1, drop_prob)
        #         if not isinstance(op2, Identity):
        #             h2 = drop_path(h2, drop_prob)
        #     s = h1 + h2
        #     states += [s]

        if self.se_layer is None:
            return torch.cat([states[i] for i in self._concat], dim=1)
        else:
            return self.se_layer(torch.cat([states[i] for i in self._concat], dim=1))


class Cell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev, SE=False):
        super(Cell, self).__init__()
        print(C_prev_prev, C_prev, C)

        self.se_layer = None

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

        if SE:
            self.se_layer = SELayer(channel=self.multiplier * C)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]

        if self.se_layer is None:
            return torch.cat([states[i] for i in self._concat], dim=1)
        else:
            return self.se_layer(torch.cat([states[i] for i in self._concat], dim=1))


class AuxiliaryHeadCIFAR(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),  # image size = 2 x 2
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class AuxiliaryHeadImageNet(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 14x14"""
        super(AuxiliaryHeadImageNet, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            # NOTE: This batchnorm was omitted in my earlier implementation due to a typo.
            # Commenting it out for consistency with the experiments in the paper.
            # nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class NetworkCIFAR(nn.Module):

    def __init__(self, C, num_classes, layers, auxiliary, genotype, SE=False):
        super(NetworkCIFAR, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary

        stem_multiplier = 3
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False

            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, SE=SE)

            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.droprate)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux


class PyramidNetworkCIFAR(nn.Module):
    def __init__(self, C, num_classes, layers, auxiliary, genotype, increment=4, SE=False):
        super(PyramidNetworkCIFAR, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary

        stem_multiplier = 3
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                # C_curr *= 2
                reduction = True
            else:
                reduction = False

            cell = Cell3(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, SE=SE)

            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

            C_curr += increment

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        # logits_aux = None
        logits_aux = torch.zeros(0)            
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        # if logits_aux is None:
        #     logits_aux = torch.zeros(x.size(0))            
        return logits, logits_aux


class NetworkImageNet(nn.Module):

    def __init__(self, C, num_classes, layers, auxiliary, genotype):
        super(NetworkImageNet, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary

        self.stem0 = nn.Sequential(
            nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        self.stem1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        C_prev_prev, C_prev, C_curr = C, C, C

        self.cells = nn.ModuleList()
        reduction_prev = True
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        s0 = self.stem0(input)
        s1 = self.stem1(s0)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.droprate)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux


if __name__ == '__main__':
    import validation.utils as utils
    import models.micro_genotypes as genotypes

    genome = genotypes.NSGANet
    # model = AlterPyramidNetworkCIFAR(30, 10, 20, True, genome, 6, SE=False)
    model = PyramidNetworkCIFAR(48, 10, 20, True, genome, 22, SE=True)
    # model = NetworkCIFAR(34, 10, 20, True, genome, SE=True)
    # model = GradPyramidNetworkCIFAR(34, 10, 20, True, genome, 4)
    model.droprate = 0.0

    # calculate number of trainable parameters
    print("param size = {}MB".format(utils.count_parameters_in_MB(model)))
