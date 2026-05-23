import torch.nn as nn

from src.decoder.MLDecoder import MLDecoder


def add_ml_decoder_head(model, num_classes=-1, num_of_groups=-1, decoder_embedding=768, zsl=0):
    if num_classes == -1:
        num_classes = model.num_classes
    num_features = model.num_features
    if hasattr(model, 'global_pool') and hasattr(model, 'fc'):  # resnet50
        model.global_pool = nn.Identity()
        del model.fc
        model.fc = MLDecoder(num_classes=num_classes, initial_num_features=num_features, num_of_groups=num_of_groups,
                             decoder_embedding=decoder_embedding, zsl=zsl)
    elif hasattr(model, 'head'):  # tresnet
        if hasattr(model, 'global_pool'):
            model.global_pool = nn.Identity()
        del model.head
        model.head = MLDecoder(num_classes=num_classes, initial_num_features=num_features, num_of_groups=num_of_groups,
                               decoder_embedding=decoder_embedding, zsl=zsl)
    else:
        print("model is not suited for ml-decoder")
        exit(-1)

    return model
