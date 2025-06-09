import torch.nn as nn

class ShallowModel(nn.Module):

    def __init__(self, model, model_type):
        super().__init__()
        if(model_type == "efficientnet"):
            self.shallow1 = model.conv_stem
            self.shallow2 = model.bn1
            self.shallow3 = model.blocks[0]
            self.shallow4 = model.blocks[1]
        else:
            self.shallow1 = model.stem
            self.shallow2 = model.stages[0]
            self.shallow3 = model.stages[1]
            self.shallow4 = model.stages[2]


    def forward(self, x):
        x = self.shallow1(x)
        x = self.shallow2(x)
        x = self.shallow3(x)
        x = self.shallow4(x)

        return x


class DeepModel(nn.Module):
    def __init__(self, model, model_type):
        super().__init__()

        self.model_type = model_type 

        if(model_type == "efficientnet"):
            self.blocks = model.blocks[2:]
            self.conv_head = model.conv_head
            self.bn2 = model.bn2
            self.global_pool = model.global_pool
            self.classifier = model.classifier
        else:
            self.layer = model.stages[3:]
            self.id = model.norm_pre
            self.classifier = model.head
    
    def forward(self, input):
        
        if(self.model_type == "efficientnet"):
            x = self.blocks(input)
            x = self.conv_head(x)
            x = self.bn2(x)
            x = self.global_pool(x)
            x = self.classifier(x)
        else:
            x = self.layer(input)
            x = self.id(x)
            x = self.classifier(x)

        return x
    
class SplitModel(nn.Module):

    def __init__(self, model, model_type):
        super().__init__()

        if(model_type not in ("efficientnet", "convnext")):
            raise Exception("Unknown model type")
        
        self.shallow = ShallowModel(model, model_type)
        self.deep = DeepModel(model, model_type)

    def forward(self, input):

        x = self.shallow(input)
        x = self.deep(x)

        return x