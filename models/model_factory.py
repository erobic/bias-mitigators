from models.fc_models import *
from models.cnn_models import *
from models.vqa.updn.net import *
from models.vqa.mcan.net import MCAN
# from models.vqa.ban.ban import BAN, BANNoDropout


def build_model(option,
                model_name,
                in_dims=None,
                hid_dims=None,
                out_dims=None,
                freeze_layers=None,
                dropout=None
                ):
    if 'updn' in model_name.lower() or 'mcan' in model_name.lower() or 'ban' in model_name.lower():
        m = eval(model_name)(option.dataset_info.pretrained_emb,
                             option.dataset_info.token_size,
                             option.dataset_info.ans_size)
    else:
        if in_dims is None and hid_dims is None:
            m = eval(model_name)(num_classes=out_dims)
        elif hid_dims is None:
            m = eval(model_name)(in_dims=in_dims, num_classes=out_dims)
        elif dropout is None:
            m = eval(model_name)(in_dims=in_dims, hid_dims=hid_dims, num_classes=out_dims)
        else:
            m = eval(model_name)(in_dims=in_dims, hid_dims=hid_dims, num_classes=out_dims, dropout=dropout)
        if freeze_layers is not None:
            m.freeze_layers(freeze_layers)
    return m
