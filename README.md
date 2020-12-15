## How Robust are Randomized Smoothing based Defenses to Data Poisoning?

### Abstract

<p align = justify>
The prediction of certifiably robust classifiers remains constant around a neighborhood of a point, making them resilient to test-time attacks with a guarantee. 
In this work, we present a previously unrecognized threat to robust machine learning models that highlights the importance of training-data quality in achieving high certified robustness. Specifically, we propose a novel bilevel optimization based data poisoning attack that degrades the robustness guarantees of certifiably robust classifiers.
Unlike other data poisoning attacks that reduce the accuracy of the poisoned models on a small set of target points, our attack reduces the average certified radius of an entire target class in the dataset. Moreover, our attack is effective even when the victim trains the models from scratch using state-of-the-art robust training methods such as <a href="https://arxiv.org/pdf/1902.02918.pdf">Gaussian data augmentation</a>, <a href="https://arxiv.org/pdf/2001.02378.pdf">MACER</a>, and <a href="https://arxiv.org/abs/1906.04584">SmoothAdv</a>.
To make the attack harder to detect we use clean-label poisoning points with imperceptibly small distortions. The effectiveness of the proposed method is evaluated by poisoning MNIST and CIFAR10 datasets and training deep neural networks using the previously mentioned robust training methods and certifying their robustness using randomized smoothing. 
For the models trained with these robust training methods our attack points reduce the average certified radius of the target class by more than 30% and are transferable to models with different architectures and models trained with different robust training methods.
	
<hr>

### The codes used to report the results in the paper <b>"[How Robust are Randomized Smoothing based Defenses to Data Poisoning?](https://arxiv.org/abs/2012.01274)"</b> are present in this repository. The code is divided into contains 3 main folders. 

<ol>
<li> <p align = justify> The "attack_generation" folder contains the code to run the data poisning experiments which generate the poisoned data against different robust training procedures. Each subfolder in this folder contains code to generate poison data against a robust training procedure on MNIST and CIFAR10 datasets. The code for ApproxGrad algorithm to solve the bilevel optimization is present in the folder "bilevel_optimizer".</li>

<li> <p align = justify> The "evaluation" folder contains the code that is used to train models with different robust training procedures on clean and poisoned data. The folder also contains the code to certify the models using randomized smoothing. The core.py file in "randomized_smoothing_certification" contains the code from Cohen et. al, which is used to obatin the certified radius and certified accuracy for points in the test set. </li>
	
<li> <p align = justify> The "models" folder contains code for the keras based models used in the work.</li>

</ol>

<hr>

### Generating the datasets:
<p align = justify>For MNIST we randomly sample 55000 points and 5000 points for the training and validation sets and save them as "X_train.npy", "Y_train.npy" and "X_val.npy" ,"Y_val.npy" in the folder name "data". Similarly for CIFAR10 we randomly samle 45000 and 5000 points as training and validation data. The test sets of both the datasets contain 10000 points and are the standard test sets. These datasets are used in all the experiements.

<hr>

### Poison Generation
<p align = justify>To generate the poison data against a particular robust training procedure and the dataset, navigate to the appropriate folder in "attack_generation" folder and run the python file. For example, to generate poisoned data against Gaussian data augmentation on MNIST, use the following commands.<br>
	<ol>
	<li>Navigate to "attack_generation\gaussian_data_augmentation\MNIST".</li>
	<li>Create a folder called "data" and place the generated data in the folder.</li>
	<li>Run the command "python3 poison_radius_gaussianaug.py".</li>
	</ol>
<p align = justify>The poison data will be generated and placed in the folder "data". This data will generated considering the class 8 as the target class. To target a differnet class, change the parameter "target_class" in poison_radius_gaussianaug.py.

<i>Poison data for other training procedures and datasets can be generated similarly.</i>

<hr>

### Evaluation
<p align = justify>To evaluate the effect of poisoning, a model trained from scratch on the generated poison data and robust training procedure is used. To obtain this model, navidate to the appropriate folder based on the training method and dataset to be used in the "evaluation" folder. For example to test the effect of poisoning against Gaussian data augmentation on MNIST, use the following steps.
	<ol>
	<li>Navigate to "evaluation\MNIST".</li>
	<li>Create a folder called "data" and place the generated poisoned data along with clean daat in the folder.</li>
	<li>Run the command "python3 gaussian_augmented_training.py".</li>
	<li><p align = justify>This command will generate a model trained on the poisoned data and place it in the folder "Models". To generate data trained only on clean data, change the parameter "dataset" in the file "gaussian_augmented_training.py".</li>
	<li><p align = justify>Finally to obtain the certified radius and certified accuracy, run the code "python3 certify_mnist.py", present in "randomized_smoothing_certification" by changing the model path to the location of the generated poisoned model. The certification code will return the average certified radius and approximate certified test accuracy of the poisoned model on 500 randomly smapled points of the target class from the test set.</li>
	</ol>
<i>Evaluation of models trained on different datasets and using different robust training procedures can be obtained similarly.</i>

<hr>

#### Citing

If you find this useful for your work, please consider citing
<pre>
<code>
@misc{mehra2020robust,
      title={How Robust are Randomized Smoothing based Defenses to Data Poisoning?}, 
      author={Akshay Mehra and Bhavya Kailkhura and Pin-Yu Chen and Jihun Hamm},
      year={2020},
      eprint={2012.01274},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
</code>
</pre>
