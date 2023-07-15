"""
Create torchvision models and reshape their classification layers for transfer learning
"""
import torch
import torch.nn
import torchvision.models as models

# architecture families whose output layers reshape_model() knows how to replace
SUPPORTED_ARCHS = ("resnet", "resnext", "wide_resnet", "regnet", "shufflenet",
                   "alexnet", "vgg", "mobilenet", "efficientnet", "mnasnet", "convnext",
                   "densenet", "squeezenet", "inception", "googlenet")


def create_model(arch, num_classes, pretrained=True, dropout=0.0, freeze=False):
    """
    Create a torchvision model (with its ImageNet weights if pretrained) and reshape it for num_classes.
    train.py and export_onnx.py both use it so that the exported model is built exactly like the trained one.
    """
    if pretrained:
        print(f"=> using pre-trained model '{arch}'")
        model = models.get_model(arch, weights="DEFAULT")
    else:
        print(f"=> creating model '{arch}'")
        model = models.get_model(arch)

    # freeze the pre-trained layers, the new layers created by reshape_model() stay trainable
    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    return reshape_model(model, arch, num_classes, dropout)


def output_layer(in_features, num_classes, dropout=0.0):
    """
    Create a new fully-connected output layer, preceded by a dropout layer if dropout > 0
    """
    layer = torch.nn.Linear(in_features, num_classes)

    if dropout > 0:
        return torch.nn.Sequential(torch.nn.Dropout(p=dropout), layer)

    return layer


#
# reshape the model for N classes
#
def reshape_model(model, arch, num_classes, dropout=0.0):
    """Reshape a model's output layers for the given number of classes"""

    # reshape output layers for the dataset
    if arch.startswith(("resnet", "resnext", "wide_resnet", "regnet", "shufflenet")):
        model.fc = output_layer(model.fc.in_features, num_classes, dropout)
        print(f"=> reshaped fully-connected layer with: {model.fc}")

    elif arch.startswith(("alexnet", "vgg", "mobilenet", "efficientnet", "mnasnet", "convnext")):
        model.classifier[-1] = output_layer(model.classifier[-1].in_features, num_classes, dropout)
        print(f"=> reshaped classifier layer with: {model.classifier[-1]}")

    elif arch.startswith("densenet"):
        model.classifier = output_layer(model.classifier.in_features, num_classes, dropout)
        print(f"=> reshaped DenseNet classifier layer with: {model.classifier}")

    elif arch.startswith("squeezenet"):
        model.classifier[1] = torch.nn.Conv2d(model.classifier[1].in_channels, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model.num_classes = num_classes
        print(f"=> reshaped SqueezeNet classifier layer with: {model.classifier[1]}")

    elif arch.startswith("inception"):
        # the auxiliary classifier only exists when the model is created with aux_logits (the default)
        if model.AuxLogits is not None:
            model.AuxLogits.fc = torch.nn.Linear(model.AuxLogits.fc.in_features, num_classes)
            print(f"=> reshaped Inception aux-logits layer with: {model.AuxLogits.fc}")

        model.fc = output_layer(model.fc.in_features, num_classes, dropout)
        print(f"=> reshaped Inception fully-connected layer with: {model.fc}")

    elif arch.startswith("googlenet"):
        if model.aux_logits:
            from torchvision.models.googlenet import InceptionAux

            model.aux1 = InceptionAux(512, num_classes)
            model.aux2 = InceptionAux(528, num_classes)

            print("=> reshaped GoogleNet aux-logits layers with: ")
            print(f"      {model.aux1}")
            print(f"      {model.aux2}")

        model.fc = output_layer(model.fc.in_features, num_classes, dropout)
        print(f"=> reshaped GoogleNet fully-connected layer with:  {model.fc}")

    else:
        raise ValueError(f"classifier reshaping not supported for {arch} "
                         f"(supported architectures: {', '.join(SUPPORTED_ARCHS)})")

    return model
