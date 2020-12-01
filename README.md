## How Robust are Randomized Smoothing based Defenses to Data Poisoning?

### The codes used to run the experiments to report the results in the paper are present in this repository. The code is divided into contains 3 main folders. 
<ol>
<li> The "attack_generation" folder contains the code to run the data poisning experiments which generate the poisoned data against different robust training procedures. Each subfolder in this folder contains code to generate poison data against a robust training procedure on MNIST and CIFAR10 datasets. The code for ApproxGrad algorithm to solve the bilevel optimization is present in the folder "bilevel_optimizer".</li>

<li> The "evaluation" folder contains the code that is used to train models with different robust training procedures on clean and poisoned data. The folder also contains the code to certify the models using randomized smoothing. The core.py file in "randomized_smoothing_certification" contains the code from Cohen et. al, which is used to obatin the certified radius and certified accuracy for points in the test set. </li>
	
<li> The "models" folder contains code for the keras based models used in the work.</li>

</ol>

### Generating the datasets:
For MNIST we randomly sample 55000 points and 5000 points for the training and validation sets and save them as "X_train.npy", "Y_train.npy" and "X_val.npy" ,"Y_val.npy" in the folder name "data". Similarly for CIFAR10 we randomly samle 45000 and 5000 points as training and validation data. The test sets of both the datasets contain 10000 points and are the standard test sets. These datasets are used in all the experiements.

### Poison Generation
To generate the poison data against a particular robust training procedure and the dataset, navigate to the appropriate folder in "attack_generation" folder and run the python file. For example, to generate poisoned data against Gaussian data augmentation on MNIST, use the following commands.<br>
	<ol>
	<li>Navigate to "attack_generation\gaussian_data_augmentation\MNIST".</li>
	<li>Create a folder called "data" and place the generated data in the folder.</li>
	<li>Run the command "python3 poison_radius_gaussianaug.py".</li>
	</ol>
The poison data will be generated and placed in the folder "data". This data will generated considering the class 8 as the target class. To target a differnet class, change the parameter "target_class" in poison_radius_gaussianaug.py.<br><br>
*Poison data for other training procedures and datasets can be generated similarly.*

### Evaluation
To evaluate the effect of poisoning, a model trained from scratch on the generated poison data and robust training procedure is used. To obtain this model, navidate to the appropriate folder based on the training method and dataset to be used in the "evaluation" folder. For example to test the effect of poisoning against Gaussian data augmentation on MNIST, use the following steps.<br><br>
	<ol>
	<li>Navigate to "evaluation\MNIST".</li>
	<li>Create a folder called "data" and place the generated poisoned data along with clean daat in the folder.</li>
	<li>Run the command "python3 gaussian_augmented_training.py".</li>
	<li>This command will generate a model trained on the poisoned data and place it in the folder "Models". To generate data trained only on clean data, change the parameter "dataset" in the file "gaussian_augmented_training.py".</li>
	<li>Finally to obtain the certified radius and certified accuracy, run the code "python3 certify_mnist.py", present in "randomized_smoothing_certification" by changing the model path to the location of the generated poisoned model. The certification code will return the average certified radius and approximate certified test accuracy of the poisoned model on 500 randomly smapled points of the target class from the test set.</li>
	</ol>
*Evaluation of models trained on different datasets and using different robust training procedures can be obtained similarly.* 
