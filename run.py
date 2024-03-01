import os
import sys
import torch
from itertools import chain
from torch.utils.data import DataLoader
from datasets.cont_data import *
from common import parse_args
import torch.nn as nn
import numpy as np

from datasets.cont_data import get_data, generate_random_cl, StandardCL
from utils.utils import Logger, load_deit_pretrain, Criterion
from training import train
from networks.net import Net
from copy import deepcopy
from datetime import datetime
import clip
import timm

if __name__ == '__main__':
    args = parse_args()
    args.logger = Logger(args, args.folder)
    args.logger.now()

    args.device = f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu"

    # Assign None to feature extractors
    args.model_clip, args.model_vit = None, None

    if args.dynamic is not None:
        args.n_components = args.dynamic

    np.random.seed(args.seed)

    #################################################
    # LOAD DATASET
    #################################################
    train_data, test_data = get_data(args)
    if args.task_type == 'standardCL_randomcls':
        task_list = generate_random_cl(args)
        train_data = StandardCL(train_data, args, task_list)
        test_data = StandardCL(test_data, args, task_list)

    # Obtain the list of unique labels from all tasks
    args.sup_labels = []
    for task in task_list:
        args.logger.print(task)
        for name in task[0]:
            if name not in args.sup_labels:
                args.sup_labels.append(name)

    args.logger.print('\n\n',
                        os.uname()[1] + ':' + os.getcwd(),
                        'python', ' '.join(sys.argv),
                      '\n\n')

    # number of heads after final task
    args.out_size = len(args.sup_labels)
    args.num_cls_per_task = int(args.out_size // args.n_tasks)
    args.logger.print('\n', args, '\n')

    ############## transformer; Deit or ViT ############
    if 'adapter' in args.model:
        if 'vitadapter' in args.model:
            model_type = 'vit_base_patch16_224'
            if '_more' in args.model:
                from networks.my_vit_hat import vit_base_patch16_224 as transformer
            else:
                from networks.my_vit import vit_base_patch16_224 as transformer
        elif 'deitadapter' in args.model:
            model_type = 'deit_small_patch16_224'
            if '_more' in args.model:
                from networks.my_vit_hat import deit_small_patch16_224 as transformer
            # elif 'owm_' in args.model:
            #     from networks.my_vit_owm import deit_small_patch16_224 as transformer
            # todo: networks.my_vit_owm does not exist
            else:
                from networks.my_vit import deit_small_patch16_224 as transformer
        
        if 'pass' in args.model:
            num_classes = args.total_cls * 4
        else:
            num_classes = args.total_cls

        if '_hat' in args.model and args.use_buffer:
            num_classes = args.num_cls_per_task + 1     # single head, is the +1 for the OOD class?

        # Instantiate the transformer
        args.net = transformer(pretrained=True,
                               num_classes=num_classes,
                               latent=args.adapter_latent,
                               args=args)
        args.net.to(args.device)
        
        if args.distillation:
            teacher = timm.create_model(model_type, pretrained=False, num_classes=num_classes).cuda()

        if 'deitadapter' in args.model:
            load_deit_pretrain(args, args.net)  # Load the pretrained weights in the transformer

        if args.model == 'vitadapter_more' or args.model == 'deitadapter_more':
            args.model_clip, args.clip_init = None, None
            from apprs.vitadapter import ViTAdapter as Model

    args.criterion = Criterion(args, args.net)
    model = Model(args) #todo: this gives error if NOT (args.model == 'vitadapter_more' or args.model == 'deitadapter_more')

    # This is the simplified version without 1000000 if-else options
    if args.load_dir is None:
        ################################################
        # TRAINING
        ################################################
        args.train = True
        train(task_list, args, train_data, test_data, model)
    else:
        if args.train_clf:
            args.train = True
            from train_clf import train
            train(task_list, args, train_data, test_data, model)
        else:
            # THIS IS THE TEST_ONLY PART
            args.train = False
            from testing import test
            test(task_list, args, train_data, test_data, model)


    # todo: teacher_net unreferenced, and 'clipadapter_hat appear in both two if statements
    # if args.distillation:
    #     if args.model in ['vitadapter', 'clipadapter', 'clipadapter_hat']:
    #         args.logger.print("Load teacher")
    #         model.teacher = teacher
    #     if args.model in ['clipadapter', 'clipadapter_hat']:
    #         args.logger.print("Load teacher net")
    #         model.teacher_net = teacher_net

    # todo: lot of unreferenced modules: train_calibration, train_ebd, save_val_outputs, val_output_comp
    # if args.load_dir is None:
    #     ################################################
    #     # TRAINING
    #     ################################################
    #     args.train = True
    #     train(task_list, args, train_data, test_data, model)
    # else:
    #     # If not classic training check if need to do other staff post-training
    #     if args.calibration and not args.train_clf and not args.obtain_val_outputs and not args.obtain_val_outputs_comp and not args.train_ebd:
    #         args.train = True
    #         from train_calibration import train
    #         train(task_list, args, train_data, test_data, model)
    #     elif args.train_clf:
    #         args.train = True
    #         from train_clf import train
    #         train(task_list, args, train_data, test_data, model)
    #     elif args.train_ebd:
    #         args.train = True
    #         from train_ebd import train
    #         train(task_list, args, train_data, test_data, model)
    #     elif args.obtain_val_outputs:
    #         from save_val_outputs import test
    #         test(task_list, args, train_data, test_data, model)
    #     elif args.obtain_val_outputs_comp:
    #         from val_output_comp import test
    #         test(task_list, args, train_data, test_data, model)
    #     else:
    #         # THIS IS THE TEST_ONLY PART
    #         args.train = False
    #         from testing import test
    #         test(task_list, args, train_data, test_data, model)

    args.logger.now()
