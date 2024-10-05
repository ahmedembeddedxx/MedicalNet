import torch
from torch import nn
from MedicalNet.models import resnet

def generate_model(opt):
    assert opt.model in ['resnet'], "Only 'resnet' model is supported."

    model_depth_map = {
        10: resnet.resnet10,
        18: resnet.resnet18,
        34: resnet.resnet34,
        50: resnet.resnet50,
        101: resnet.resnet101,
        152: resnet.resnet152,
        200: resnet.resnet200
    }

    assert opt.model_depth in model_depth_map, f"Invalid depth. Choose from: {list(model_depth_map.keys())}"

    model = model_depth_map[opt.model_depth](
        sample_input_W=opt.input_W,
        sample_input_H=opt.input_H,
        sample_input_D=opt.input_D,
        shortcut_type=opt.resnet_shortcut,
        no_cuda=opt.no_cuda,
        num_seg_classes=opt.n_seg_classes
    )

    if not opt.no_cuda:
        model = model.cuda()
        if len(opt.gpu_id) > 1:
            model = nn.DataParallel(model, device_ids=opt.gpu_id)
        else:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id[0])
            model = nn.DataParallel(model)

    if opt.phase != 'test' and opt.pretrain_path:
        print(f'Loading pretrained model from {opt.pretrain_path}')
        pretrain = torch.load(opt.pretrain_path)

        pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in model.state_dict()}
        model.load_state_dict({**model.state_dict(), **pretrain_dict})

        new_parameters = [
            p for pname, p in model.named_parameters() 
            if any(layer_name in pname for layer_name in opt.new_layer_names)
        ]
        new_parameters_id = list(map(id, new_parameters))
        base_parameters = [p for p in model.parameters() if id(p) not in new_parameters_id]

        parameters = {'base_parameters': base_parameters, 'new_parameters': new_parameters}
        return model, parameters

    return model, model.parameters()
