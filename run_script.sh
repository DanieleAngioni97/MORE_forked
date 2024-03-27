# # Training
# cmd="nohup python run.py --model deitadapter_more --n_tasks 5 --dataset cifar10 --adapter_latent 64 --optim sgd --compute_md --compute_auc --buffer_size 200 --n_epochs 20 --lr 0.005 --batch_size 64 --use_buffer --class_order 0 --folder cifar10"

# Back-Update
cmd="nohup python run.py --model deitadapter_more --n_tasks 5 --dataset cifar10 --adapter_latent 64 --optim sgd --compute_auc --buffer_size 200 --folder MORE_EXP --load_dir logs/MORE_EXP--ds-cifar10--model-deitadapter_more --n_epochs 10 --print_filename train_clf_epoch=10.txt --use_buffer --load_task_id 4 --train_clf --train_clf_save_name model_task_clf_epoch=10 --class_order 0"



# # Testing
# # e.g., For CIFAR10-5T,
# # if back-update is used,
# python run.py --model deitadapter_more --n_tasks 5 --dataset cifar10 --adapter_latent 64 --optim sgd --use_md --compute_auc --buffer_size 200 --folder cifar10 --load_dir logs/cifar10 --print_filename testing_train_clf_useMD.txt --use_buffer --load_task_id 4 --test_model_name model_task_clf_epoch=10_ --class_order 0
	
# # if back-update is not used.
python run.py --model deitadapter_more --n_tasks 5 --dataset cifar10 --adapter_latent 64 --optim sgd --use_md --compute_auc --buffer_size 200 --folder cifar10 --load_dir logs/cifar10 --print_filename testing_useMD.txt --use_buffer --load_task_id 4 --class_order 0

$cmd

