from utils.fm import my_load
from os.path import join
from utils.utils import Logger, Tracker
from common import parse_args
from openood.networks import ResNet18_32x32
args = parse_args()
args.folder = 'cifar10_epoch20_original'
args.logger = Logger(args, args.folder, filename='check_perf')
tracker_fname_list = [
    'cil_tracker_train_clf_equal_test',
    'til_tracker_train_clf_equal_test',
    'auc_softmax_tracker_train_clf_equal_test'
]
metrics = {}
for metric_name in tracker_fname_list:
    print(f"#######################")
    print(f"# {metric_name}")
    print(f"#######################")
    path = join(args.logger.dir(), metric_name)
    mat = my_load(path)
    tracker = Tracker(args)
    tracker.mat = mat
    tracker.print_result(args.n_tasks - 1, 'acc')
    tracker.print_result(args.n_tasks - 1, 'forget')
    print("")

print("")