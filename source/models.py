import torch.nn as nn

class ShallowModel(nn.Module):

    def __init__(self, model):
        
        super().__init__()
        self.shallow1 = model.conv_stem
        self.shallow2 = model.bn1
        self.shallow3 = model.blocks[0]
        self.shallow4 = model.blocks[1]
        self.layers = [self.shallow1, self.shallow2, self.shallow3, self.shallow4]

    def forward(self, x):
        x = self.shallow1(x)
        x = self.shallow2(x)
        x = self.shallow3(x)
        x = self.shallow4(x)

        return x


class DeepModel(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.blocks = model.blocks[2:]
        self.conv_head = model.conv_head
        self.bn2 = model.bn2
        self.global_pool = model.global_pool
        self.classifier = model.classifier
    
    def forward(self, input):
        
        x = self.blocks(input)
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.global_pool(x)
        x = self.classifier(x)

        return x
    
class SplitModel(nn.Module):

    def __init__(self, model):
        super().__init__()
        
        self.shallow = ShallowModel(model)
        self.deep = DeepModel(model)

    def forward(self, input):

        x = self.shallow(input)
        x = self.deep(x)

        return x