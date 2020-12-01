# How Robust are Randomized Smoothing based Defenses to Data Poisoning?

### The codes used to run the experiments to report the results in the paper are present in this folder. The code is divided into contains 3 main folders. 
	
  #### 1. The "attack_generation" folder contains the code to run the data poisning experiments which generate the poisoned data against different robust training procedures. Each subfolder in this folder contains code to generate poison data against a robust training procedure on MNIST and CIFAR10 datasets. The code for ApproxGrad algorithm to solve the bilevel optimization is present in the folder "bilevel_optimizer".
	
  #### 2. The "evaluation" folder contains the code that is used to train models with different robust training procedures on clean and poisoned data. The folder also contains the code to certify the models using randomized smoothing. The core.py file in "randomized_smoothing_certification" contains the code from Cohen et. al, which is used to obatin the certified radius and certified accuracy for points in the test set. 
	
  #### 3. The "models" folder contains code for the keras based models used in the work.
