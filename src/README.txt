Yêu cầu: python3, pip
Vào thư mục src: cd src

1. Cài đặt các thư viện cần thiết: 	pip install -r requirements.txt	

2: Tải bộ dữ liệu Omniglot tại: 
https://drive.google.com/file/d/1eHf3Dw3q9_OPOkFqR5mLfNpOc__9cise/view?usp=sharing

3. Chạy thử nghiệm
python3 --experiment [Tên bộ dữ liệu] --approach [Tên phương pháp] --drop_type Gauss --droprate [Giá trị init_alpha] ...

- Chi tiết tham số xem trong tệp: arguments.py
- Ví dụ câu lệnh chạy xem trong: grid_run.py (chạy cho nhiều seed)

- Ví dụ:
+ Thử nghiệm EWC với ALV cho Split Mnist: 
python3 main.py --experiment split_mnist --approach ewc_vd --drop_type Gauss --droprate 0.5 --KL_weight 0.0001 --lamb 40000

+ Thử nghiệm VCL với ALV cho Split Mnist: 
python3 main.py --experiment split_mnist --approach vcl_vd --drop_type Gauss --droprate 0.1 --KL_weight 0.01 --local_trick  --num_samples 1 --test_sample 100

+ Thử nghiệm UCL với ALV cho Split Mnist: 
python3 main.py --experiment split_mnist --approach ucl_vd --drop_type Gauss --droprate 0.1 --KL_weight 0.001 --local_trick --beta 0.0001 --ratio 0.5 --lr_rho 0.001 --alpha 0.01 --num_samples 1 --test_sample 100