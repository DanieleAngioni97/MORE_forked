from utils.utils import *
from utils.ood_func import *
from argparse import ArgumentParser
from sys import argv

# python /homes/sgupta/projects/more_openmax/resources/MORE/run.py --model deitadapter_more --n_tasks 5 --dataset cifar10 --adapter_latent 64 --optim sgd --my_ood_det --ood_det msp --use_md --compute_auc --buffer_size 200 --folder cifar10_ood --load_dir logs/cifar10 --print_filename testing_ood_det.txt --use_buffer --load_task_id 4 --class_order 0 

# removing --test_model_name model_task_clf_epoch=10_ 

def clood_msp(logits):
    sm_scores = torch.softmax(logits, dim = 1)
    conf, pred = torch.max(sm_scores, dim = 1)
    return pred, conf

def clood_maxlogit(logits):
    conf, pred = torch.max(logits, dim = 1)
    return pred, conf

def clood_ebo(logits, temp):
    sm_scores = torch.softmax(logits, dim = 1)
    _, pred = torch.max(sm_scores, dim = 1)
    conf = temp * torch.logsumexp(logits/temp, dim = 1)
    return pred, conf

def clood_scale(feature, percentile, fc_layer):
    thresh_logits = thresholded_activations(feature, percentile, fc_layer, det = 'scale')
    _, pred = torch.max(thresh_logits, dim = 1)
    energy_conf = torch.logsumexp(thresh_logits, dim = 1)       # ref scale
    return pred, energy_conf

def clood_ash(feature, percentile, fc_layer):
    thresh_logits = thresholded_activations(feature, percentile, fc_layer, det = 'ash')
    _, pred = torch.max(thresh_logits, dim = 1)
    energy_conf = torch.logsumexp(thresh_logits, dim = 1)       # ref scale
    return pred, energy_conf

def clood_gradnorm(w, b, logits, feature, num_cls):
    with torch.enable_grad():
        scores = gradnorm(feature, w, b, num_cls)
        _, pred = torch.max(logits, dim = 1)
    return pred, torch.from_numpy(scores)
    

def clood_react():
    pass





def get_features():
    pass

def get_logits():
    pass

def get_fc_layer():
    pass

def get_fc_w_b():
    return w, b
    




if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = ArgumentParser()
    ood_detectors = ['max_logits', 'msp', 'ebo', 'scale', 'ash', 'gradnorm']
    
    parser.add_argument('--model_task', dest='model_task', type=int, choices=[0,1, 2, 3, 4, 5], help='select the trained model of the task_id(0,1,2,3,4)')
    parser.add_argument('--ood_detector', dest='ood_det', type=str, default='max_logit', choices=ood_detectors, help='name of the posthoc ood detector from the choices')
    parser.add_argument('--ood_log_folder', dest='log_folder', type=str, default=None, help='directory NAME e.g., save under logs/NAME')
    parser.add_argument('--temperature', dest='ebo_temp', default= 1, required=(ood_detectors[2] in argv))
    parser.add_argument('--percentile', dest='percentile', default = 85, choices = [65, 70, 75, 80, 85, 90, 95], required=(ood_detectors[3] in argv))   # scale: 85, ash: 65

    args = parser.parse_args()
    log_folder = args.log_folder
    ood_det = args.ood_det
    model_task = args.model_task
    num_cls = 2                             # CHECK
    logger = Logger(args, log_folder)
    logger.now()

    logits = get_logits()
    features = get_features()
    fc_layer = get_fc_layer(model_task)
    

    if ood_det == "max_logit":
        pred, conf = clood_maxlogit(logits)

    elif ood_det == "msp":
        pred, conf = clood_msp(logits)

    elif ood_det == "ebo":
        temperature = args.ebo_temp
        pred, conf = clood_ebo(logits, temperature)

    elif ood_det == "scale":
        # fc_layer = get fc layer of the model model_task.fc_layer 
        percentile = args.percentile
        pred, conf = clood_scale(features, percentile, fc_layer)

    elif ood_det == "ash":
        percentile = args.percentile
        pred, conf = clood_ash(features, percentile, fc_layer)

    elif ood_det == "gradnorm":
        w, b = get_fc_w_b(model_task)
        pred, conf = clood_gradnorm(w, b, logits, features, num_cls)




# ref
# 1. scale pp: https://github.com/Jingkang50/OpenOOD/blob/main/openood/postprocessors/scale_postprocessor.py
# 2. gradnorm: https://github.com/Jingkang50/OpenOOD/blob/main/openood/postprocessors/gradnorm_postprocessor.py