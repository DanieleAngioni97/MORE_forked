import timm
from numpy.linalg import svd
from torch.optim import SGD, Adam
from collections import Counter
from itertools import chain
from utils.utils import *
import torch
import clip
import matplotlib.pyplot as plt
from torch.autograd import Variable
from numpy.random import multivariate_normal
from tqdm import tqdm
from utils.my_ipca import MyIPCA as IPCA
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

def train(task_list, args, train_data, test_data, model):
    # noise cannot be used without use_md
    if args.noise: assert args.use_md

    zeroshot = Zeroshot(args.model_clip, args)

    cil_tracker = Tracker(args)
    til_tracker = Tracker(args)
    cal_cil_tracker = Tracker(args)
    auc_softmax_tracker = AUCTracker(args)
    auc_md_tracker = AUCTracker(args)
    openworld_softmax_tracker = OWTracker(args)

    # cil_correct, til_correct are for cumulative accuracy throughout training
    cil_correct, til_correct, total = 0, 0, 0

    c_correct, c_total, p_correct, p_total = 0, 0, 0, 0
    cum_acc_list, total_loss_list, iter_list, total_iter = [], [], [], 0

    train_loaders, test_loaders, calibration_loaders = [], [], []

    args.mean, args.cov, args.cov_inv = {}, {}, {}
    args.mean_task, args.cov_noise, args.cov_inv_noise = {}, {}, {}

    param_copy = None

    combined_sigma = 0

    if args.task_type == 'concept':
        if_shift = []

    ########################################################################################################
    ########################################################################################################
    # TASKS LOOP
    ########################################################################################################
    ########################################################################################################
    for task_id in range(len(task_list)):
        task_loss_list = []

        if args.validation is None:
            t_train = train_data.make_dataset(task_id)
            t_test = test_data.make_dataset(task_id)
        else:
            # todo: here if t_test is used as validation CALL IT T_VAL, NOT T_TEST, THIS IS CONFUSING!
            t_train, t_test = train_data.make_dataset(task_id)

        if args.calibration:
            assert args.cal_batch_size > 0
            assert args.cal_epochs > 0
            assert args.cal_size > 0
            t_train, t_cal = calibration_dataset(args, t_train)
            calibration_loaders.append(make_loader(t_cal, args, train='calibration')) #todo: check when this is used

        train_loaders.append(make_loader(t_train, args, train='train'))
        test_loaders.append(make_loader(t_test, args, train='test'))

        # For some technical reason, create current_train_loader, a copy of train_loader
        current_train_loader = deepcopy(train_loaders[-1])

        if args.resume is not None:
            """
            How to load e.g.
            CUDA_VISIBLE_DEVICES=0 python run.py --model deitadapter_more --n_tasks 20 --dataset cifar100 --adapter_latent 128 --optim sgd --compute_md --compute_auc --buffer_size 2000 --n_epochs 40 --lr 0.005 --batch_size 64 --calibration --folder final_deitadapter_hat_cifar100_20t/class_order=2 --use_buffer --class_order 2 --resume_id 2 --n_epochs 1 --resume logs/final_deitadapter_hat_cifar100_20t/class_order=2/saving_buffer
            """
            saving_buffer = torch.load(args.resume)
            resume_id = saving_buffer['task_id']
            print("resume_id", resume_id)
            if task_id <= resume_id:
                if hasattr(model, 'preprocess_task'):
                    model.preprocess_task(names=train_data.task_list[task_id][0],
                                          labels=train_data.task_list[task_id][1],
                                          task_id=task_id,
                                          loader=current_train_loader)
                args.logger.print("Loading from:", args.logger.dir() + f'/model_task_{task_id}')
                state_dict = torch.load(args.logger.dir() + f'/model_task_{task_id}')
                model.net.load_state_dict(state_dict)

                # Load statistics for MD
                args = load_MD_stats(args, task_id)
                # End task
                if hasattr(model, 'end_task'):
                    if args.calibration:
                        model.end_task(calibration_loaders, test_loaders, train_loader=train_loaders[-1])
                    else:
                        model.end_task(task_id + 1, train_loader=train_loaders[-1])

                if args.use_buffer:
                    args.logger.print("Loading memory from:", args.logger.dir() + f'/memory_{resume_id}')
                    memory = torch.load(args.logger.dir() + f'/memory_{resume_id}')
                    model.buffer_dataset.data = memory[0]
                    model.buffer_dataset.targets = memory[1]
                    model.buffer_dataset.transform = train_loaders[-1].dataset.transform

                saving_buffer = torch.load(args.logger.dir() + f'/saving_buffer')
                model.p_mask = saving_buffer['p_mask']
                model.mask_back = saving_buffer['mask_back']
                cil_tracker = saving_buffer['cil_tracker']
                til_tracker = saving_buffer['til_tracker']
                continue

        # Here the model add the embeddings and the classification head for the specific tasks
        if hasattr(model, 'preprocess_task'):
            # todo: here the model knows already the final number of classes, we can fix it
            model.preprocess_task(names=train_data.task_list[task_id][0],
                                  labels=train_data.task_list[task_id][1],
                                  task_id=task_id,
                                  loader=current_train_loader)
            print(Counter(current_train_loader.dataset.targets))

        if args.distillation:
            raise NotImplementedError("model name not matching")

        if args.task_type == 'concept':
            if 'shifted' in train_data.current_labels:
                args.logger.print(train_data.current_labels)
                if_shift.append(True)
                init = int(train_data.current_labels.split('shifted: ')[-1].split(' -> ')[0])

                test_loaders[init].dataset.update()
                args.logger.print(len(test_loaders[init].dataset.targets))
            else:
                if_shift.append(False)

        if args.modify_previous_ood and task_id > 0:
            assert args.model == 'oe' or args.model == 'oe_fixed_minibatch'
            param_copy = model.net.fc.weight.detach()
            print(param_copy.sum(1))

        ########################################################################################################
        ########################################################################################################
        # EPOCHS LOOP
        ########################################################################################################
        ########################################################################################################
        for epoch in range(args.n_epochs):
            iters = []
            model.reset_eval()
            # orig is the original data (mostly likely numpy for CIFAR, and indices for ImageNet)
            for b, (x, y, f_y, names, orig) in tqdm(enumerate(current_train_loader)):
                if b > 5:
                    break
                # for b, (x, y) in tqdm(enumerate(current_train_loader)):
                # for simplicity, consider that we know the labels ahead
                f_y = f_y[:, 1]
                x, y = x.to(args.device), y.to(args.device)
                with torch.no_grad():
                    if args.model_clip:
                        x = args.model_clip.encode_image(x).type(torch.FloatTensor).to(args.device)
                    elif args.model_vit:
                        x = args.model_vit.forward_features(x)
                    if args.zero_shot:
                        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in train_data.seen_names]).to(args.device)
                        zeroshot.evaluate(x, text_inputs, y)
                loss = model.observe(x, y, names, x, f_y, task_id=task_id, b=b, B=len(train_loaders[-1]))

                total_loss_list.append(loss)
                task_loss_list.append(loss)
                cum_acc = model.correct / model.total * 100
                cum_acc_list.append(cum_acc)
                iters.append(total_iter)
                total_iter += 1
                args.logger.print(f"e:[{epoch + 1}/{args.n_epochs}], b:[{b}/{len(current_train_loader)}]"\
                                   f"-- Loss: {loss}, Cum. Acc.: {cum_acc}")
                
            iter_list.append(iters)

            if epoch == 0 and args.zero_shot:
                args.logger.print("Train Data | Task {}, Zero-shot Acc: {:.2f} | ".format(task_id, zeroshot.acc()['cil_acc']), end='')

            if args.n_epochs == 1:
                cil_correct += model.correct
                til_correct += model.til_correct
                total += model.total
                metrics = model.acc()
                args.logger.print("Task {}, CIL Cumulative Acc: {:.2f}".format(task_id, metrics['cil_acc']))
                args.logger.print("Task {}, TIL Cumulative Acc: {:.2f}".format(task_id, metrics['til_acc']))
                args.logger.print("All seen classes, CIL Cumulative Acc: {:.2f}".format(cil_correct / total * 100))
                args.logger.print("All seen classes, TIL Cumulative Acc: {:.2f}".format(til_correct / total * 100)) # NOT COMPUTED

            if args.modify_previous_ood and task_id > 0:
                assert args.model == 'oe' or args.model == 'oe_fixed_minibatch'
                out_dim, _ = param_copy.size()
                model.net.fc.weight.data[:out_dim] = param_copy.data
            
            ########################################################################################################
            ########################################################################################################
            # PER-EPOCH EVALUATION
            ########################################################################################################
            ########################################################################################################
            #TODO: where is eval_every?
            args.logger.print("Evaluating CIL and TIL of current task at eval_every")
            model.reset_eval()
            for batch in test_loaders[-1]:
                x, y, *_ = batch
                x, y = x.to(args.device), y.to(args.device)
                with torch.no_grad():
                    if args.model_clip:
                        x = args.model_clip.encode_image(x).type(torch.FloatTensor).to(args.device)
                    elif args.model_vit:
                        x = args.model_vit.forward_features(x)
                    if args.zero_shot:
                        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in train_data.seen_names]).to(args.device)
                        zeroshot.evaluate(x, text_inputs, y)
                model.evaluate(x, y, task_id, report_cil=True, total_learned_task_id=task_id, ensemble=args.pass_ensemble)
            metrics = model.acc()
            args.logger.print(f"Task {task_id}, Epoch {epoch + 1}/{args.n_epochs}, Total Loss: {np.mean(task_loss_list):.4f}, "\
                              f"CIL Acc: {metrics['cil_acc']:.2f}, TIL Acc: {metrics['til_acc']:.2f}")

            # if compute_AUC is true, compute its AUC at eval_every
            if args.compute_auc:
                args.logger.print("Evaluating AUC current task at eval_every")
                in_scores = metrics['scores']
                # if args.compute_md: 
                if args.use_md:
                    in_scores_md = metrics['scores_md']
                auc_list, auc_list_md = [], []
                auc_total_in_list, auc_total_out_list, out_id_list = [metrics['scores_total']], [], []
                for task_out in range(args.n_tasks):
                    if task_out != task_id:
                        if args.validation is None:
                            t_test = test_data.make_dataset(task_out)
                        else:
                            _, t_test = train_data.make_dataset(task_out)
                        ood_loader = make_loader(t_test, args, train='test')
                        for x, y, _, _, _ in ood_loader:
                            x, y = x.to(args.device), y.to(args.device)
                            with torch.no_grad():
                                model.evaluate(x, y, task_id=task_id, report_cil=True, total_learned_task_id=task_id, ensemble=args.pass_ensemble)
                        metrics = model.acc()

                        out_scores = metrics['scores']
                        auc = compute_auc(in_scores, out_scores)
                        auc_list.append(auc * 100)
                        args.logger.print("Epoch {}/{} | in/out: {}/{} | Softmax AUC: {:.2f}".format(epoch + 1, args.n_epochs, task_id, task_out, auc_list[-1]), end=' ')
                        auc_softmax_tracker.update(auc_list[-1], task_id, task_out)

                        # if args.compute_md:
                        if args.use_md:
                            out_scores_md = metrics['scores_md']
                            auc_md = compute_auc(in_scores_md, out_scores_md)
                            auc_list_md.append(auc_md * 100)
                            args.logger.print("| MD AUC: {:.2f}".format(auc_list_md[-1]))
                            auc_md_tracker.update(auc_list_md[-1], task_id, task_out)
                        else:
                            args.logger.print('')

                        if task_out <= task_id:
                            auc_total_in_list.append(metrics['scores_total'])
                        else:
                            auc_total_out_list.append(metrics['scores_total'])
                            out_id_list.append(task_out)

                args.logger.print("Epoch {}/{} | Average Softmax AUC: {:.2f}".format(epoch + 1, args.n_epochs, np.array(auc_list).mean()), end=' ')
                if args.compute_md:
                    args.logger.print("| Average MD AUC: {:.2f}".format(np.array(auc_list_md).mean()))
                else:
                    args.logger.print('')

                for task_out, out_scores in zip(out_id_list, auc_total_out_list):
                    auc = compute_auc(auc_total_in_list, out_scores)
                    args.logger.print("Epoch {}/{} | total in/out: {}/{} | AUC: {:.2f}".format(epoch + 1, args.n_epochs, task_id, task_out, auc * 100))
                    openworld_softmax_tracker.update(auc * 100, task_id, task_out)
                if len(auc_total_in_list) > 0 and len(auc_total_out_list) > 0:
                    auc = compute_auc(auc_total_in_list, auc_total_out_list)
                    args.logger.print("Epoch {}/{} | total in | AUC: {:.2f}".format(epoch + 1, args.n_epochs, auc * 100))

            # Save model elements required for resuming training
            if hasattr(model, 'save'):
                model.save(state_dict=model.net.state_dict(),
                            optimizer=model.optimizer,
                            task_id=task_id,
                            epoch=epoch + 1,
                            cil_tracker=cil_tracker,
                            til_tracker=til_tracker,
                            auc_softmax_tracker=auc_softmax_tracker,
                            auc_md_tracker=auc_md_tracker)
        # Save
        torch.save(model.net.state_dict(),
                    os.path.join(args.logger.dir(), f'model_task_{task_id}'))
        

        # If compute_md is true, obtain the features and compute/save the statistics for MD
        if args.compute_md:
            args.logger.print("Computing Mahalanobis distance on the current training set...")
            # First obtain the features
            model.reset_eval()
            for x, y, _, _, _ in train_loaders[-1]:
                x, y = x.to(args.device), y.to(args.device)
                with torch.no_grad():
                    if args.model_clip:
                        x = args.model_clip.encode_image(x).type(torch.FloatTensor).to(args.device)
                    elif args.model_vit:
                        x = args.model_vit.forward_features(x)
                    if args.zero_shot:
                        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in train_data.seen_names]).to(args.device)
                        zeroshot.evaluate(x, text_inputs, y)
                model.evaluate(x, y, task_id, report_cil=False, total_learned_task_id=task_id, ensemble=args.pass_ensemble)
            feature_list = np.concatenate(model.feature_list)
            label_list = np.concatenate(model.label_list)
            torch.save(feature_list,
                        os.path.join(args.logger.dir(), f"feature_task_{task_id}"))
            torch.save(label_list,
                        os.path.join(args.logger.dir(), f"label_task_{task_id}"))
            cov_list = []
            ys = list(sorted(set(label_list)))
            # Compute/save the statistics for MD
            for y in ys:
                idx = np.where(label_list == y)[0]
                f = feature_list[idx]
                cov = np.cov(f.T)
                cov_list.append(cov)
                mean = np.mean(f, 0)
                np.save(args.logger.dir() + f'mean_label_{y}', mean)
                args.mean[y] = mean
            cov = np.array(cov_list).mean(0)
            np.save(args.logger.dir() + f'cov_task_{task_id}', cov)
            args.cov[task_id] = cov
            args.cov_inv[task_id] = np.linalg.inv(cov)
            # For MD-noise
            mean = np.mean(feature_list, axis=0)
            np.save(args.logger.dir() + f'mean_task_{task_id}', mean)
            # args.mean_task[task_id] = mean
            cov = np.cov(feature_list.T)
            np.save(args.logger.dir() + f'cov_task_noise_{task_id}', cov)
            # args.cov_noise[task_id] = cov
            # args.cov_inv_noise[task_id] = np.linalg.inv(cov)
            if args.noise:
                args.mean_task[task_id] = mean
                args.cov_noise[task_id] = cov
                args.cov_inv_noise[task_id] = np.linalg.inv(cov)

            args.logger.print("End task...")
            # End task
            if hasattr(model, 'end_task'):
                if args.calibration:
                    model.end_task(calibration_loaders, test_loaders, train_loader=train_loaders[-1])
                else:
                    model.end_task(task_id + 1, train_loader=train_loaders[-1])

        if args.calibration:
            if model.w is not None:
                torch.save(model.w.data,
                            args.logger.dir() + f'calibration_w_task_{task_id}')
                torch.save(model.b.data,
                            args.logger.dir() + f'calibration_b_task_{task_id}')

        # Save statistics e.g. mean, cov, cov_inv
        if args.save_statistics:
            np.save(os.path.join(args.logger.dir(), 'statistics'), 
                    model.statistics)

        args.logger.print("######################")
        ########################################################################################################
        ########################################################################################################
        # PER TASK EVALUATION
        ########################################################################################################
        ########################################################################################################
        true_lab, pred_lab = [], []
        for p_task_id, loader in enumerate(test_loaders):
            model.reset_eval()
            for batch in loader:
                x, y, *_ = batch
                x, y = x.to(args.device), y.to(args.device)
                with torch.no_grad():
                    if args.model_clip:
                        x = args.model_clip.encode_image(x).type(torch.FloatTensor).to(args.device)
                    elif args.model_vit:
                        x = args.model_vit.forward_features(x)
                model.evaluate(x, y, task_id=p_task_id, report_cil=True, total_learned_task_id=task_id, ensemble=args.pass_ensemble)
            if args.save_output:
                
                # np.save(args.logger.dir() + 'output_learned_{}_task_{}'.format(task_id, p_task_id),
                #                                         np.concatenate(model.output_list))
                np.save(os.path.join(f"{args.logger.dir()}output_learned_{task_id}_task_{p_task_id}"),
                                                        np.concatenate(model.output_list))
                # np.save(args.logger.dir() + 'label_learned_{}_task_{}'.format(task_id, p_task_id),
                #                                         np.concatenate(model.label_list))
                np.save(os.path.join(f"{args.logger.dir()}label_learned_{task_id}_task_{p_task_id}"),
                                                        np.concatenate(model.label_list))

            metrics = model.acc()
            cil_tracker.update(metrics['cil_acc'], task_id, p_task_id)
            til_tracker.update(metrics['til_acc'], task_id, p_task_id)

            if args.tsne:
                tsne(np.concatenate(model.output_list),
                     np.concatenate(model.label_list),
                     logger=args.logger)
            if args.confusion:
                true_lab_ = np.concatenate(model.true_lab)
                pred_lab_ = np.concatenate(model.pred_lab)

                plot_confusion(true_lab_, pred_lab_, model.seen_names, task_id, p_task_id,
                                logger=args.logger, num_cls_per_task=args.num_cls_per_task)

                true_lab.append(true_lab_)
                pred_lab.append(pred_lab_)

                if p_task_id == len(test_loaders) - 1:
                    true_lab_ = np.concatenate(true_lab)
                    pred_lab_ = np.concatenate(pred_lab)
                    plot_confusion(true_lab_, pred_lab_, model.seen_names,
                                    name='confusion mat task {}'.format(p_task_id),
                                    logger=args.logger, num_cls_per_task=args.num_cls_per_task)

        args.logger.print()
        if args.compute_auc:
            args.logger.print("Softmax AUC result")
            auc_softmax_tracker.print_result(task_id, type='acc')
            args.logger.print("Open World result")
            openworld_softmax_tracker.print_result(task_id, type='acc')
        if args.compute_md:
            args.logger.print("MD AUC result")
            auc_md_tracker.print_result(task_id, type='acc')
        args.logger.print("CIL result")
        cil_tracker.print_result(task_id, type='acc')
        cil_tracker.print_result(task_id, type='forget')
        args.logger.print("TIL result")
        til_tracker.print_result(task_id, type='acc')
        til_tracker.print_result(task_id, type='forget')
        args.logger.print()

        if task_id == 0 and args.calibration:
            model.cil_acc_mat_test = deepcopy(cil_tracker.mat)

        # Save model elements required for resuming training
        if hasattr(model, 'save'):
            model.save(state_dict=model.net.state_dict(),
                        optimizer=model.optimizer,
                        task_id=task_id,
                        epoch=epoch + 1,
                        cil_tracker=cil_tracker,
                        til_tracker=til_tracker,
                        auc_softmax_tracker=auc_softmax_tracker,
                        auc_md_tracker=auc_md_tracker,
                        openworld_softmax_tracker=openworld_softmax_tracker)
        torch.save(cil_tracker.mat, os.path.join(args.logger.dir(), 'cil_tracker'))
        torch.save(til_tracker.mat, os.path.join(args.logger.dir(), 'til_tracker'))
        torch.save(auc_softmax_tracker.mat, os.path.join(args.logger.dir(), 'auc_softmax_tracker'))
        torch.save(auc_md_tracker.mat, os.path.join(args.logger.dir(), 'auc_md_tracker'))
        torch.save(openworld_softmax_tracker.mat, os.path.join(args.logger.dir(), 'openworld_softmax_tracker'))

    plt.plot(cum_acc_list)
    xticks = [l[0] for l in iter_list]
    xticks.append(iter_list[-1][-2])
    plt.xticks(xticks)
    plt.xlabel('Training Time')
    plt.ylabel('Cumulative Accuracy')
    plt.title('Cumulative Accuracy over Training Time')
    plt.savefig(os.path.join(args.logger.dir(), 'cumulative_acc.png'))
    plt.close()

def load_MD_stats(args, task_id):
    if os.path.exists(args.logger.dir() + f'/cov_task_{task_id}.npy'):
        args.compute_md = True
        args.logger.print("*** Load Statistics for MD ***")
        cov = np.load(args.logger.dir() + f'/cov_task_{task_id}.npy')
        args.cov[task_id] = cov
        args.cov_inv[task_id] = np.linalg.inv(cov)
        if args.noise:
            args.logger.print("Importing Noise Stats")
            mean = np.load(args.logger.dir() + f'/mean_task_{task_id}.npy')
            args.mean_task[task_id] = mean
            cov = np.load(args.logger.dir() + f'/cov_task_noise_{task_id}.npy')
            args.cov_noise[task_id] = cov
            args.cov_inv_noise[task_id] = np.linalg.inv(cov)
        for y in range(task_id * args.num_cls_per_task, (task_id + 1) * args.num_cls_per_task):
            mean = np.load(args.logger.dir() + f'/mean_label_{y}.npy')
            args.mean[y] = mean
        args.logger.print("Means for classes:", args.mean.keys())
        args.logger.print("Means for classes:", args.cov_inv_noise.keys())
    else:
        args.logger.print("*** No MD ***")
    return args