python main.py --dataset datasets/tum/rgbd_dataset_freiburg1_desk/ --config config/calib.yaml --no-viz > output.log 2>&1 
python main_roma.py --dataset datasets/tum/rgbd_dataset_freiburg1_desk/ --config config/calib.yaml --no-viz > output.log 2>&1 

evo_ape tum datasets/tum/rgbd_dataset_freiburg1_desk/groundtruth.txt logs/rgbd_dataset_freiburg1_desk.txt -as