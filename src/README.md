requirement: python3, pip
cd src

1. Set up library: 	pip install -r requirements.txt	

2. Download Omniglot dataset at: 
https://drive.google.com/file/d/1eHf3Dw3q9_OPOkFqR5mLfNpOc__9cise/view?usp=sharing

3. Run experiment
python3 --experiment [dataset] --approach [method] --drop_type Gauss --droprate [Giá trị init_alpha] ...

- Detail about all arguments: arguments.py
- Run for 5 different seeds: grid_run.py

- Eg:
+ Experiment with EWC + ALV for Split Mnist: 
python3 main.py --experiment split_mnist --approach ewc_vd --drop_type Gauss --droprate 0.5 --KL_weight 0.0001 --lamb 40000

+ Experiment with VCL + ALV for Split Mnist: 
python3 main.py --experiment split_mnist --approach vcl_vd --drop_type Gauss --droprate 0.1 --KL_weight 0.01 --local_trick  --num_samples 1 --test_sample 100

+ Experiment UCL + ALV for Split Mnist: 
python3 main.py --experiment split_mnist --approach ucl_vd --drop_type Gauss --droprate 0.1 --KL_weight 0.001 --local_trick --beta 0.0001 --ratio 0.5 --lr_rho 0.001 --alpha 0.01 --num_samples 1 --test_sample 100
