The official implementation of the paper **"[Time-varying Preference Bandits for Robot behavior Personalization]()"**.

Please visit our [webpage](https://jenior97.github.io/timevaryng_pbl/) for more details.





## Installation
DPB is tested with [Ubuntu 20.04](https://releases.ubuntu.com/focal/). 
- [mjpro 1.5](https://www.roboti.us/download.html)
	```bash
	wget https://www.roboti.us/download/mjpro150_linux.zip
	wget https://www.roboti.us/file/mjkey.txt

	mkdir ~/.mujoco

	unzip mjpro150_linux.zip
	mv mjpro150 ./.mujoco/
	mv mjkey.txt ./.mujoco/

	cd ./.mujoco
	cp mjkey.txt ./mjpro150/bin/

	sudo apt-get install libosmesa6-dev libglew-dev libgl-dev
	sudo apt install -y patchelf

	echo 'export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$HOME/.mujoco/mjpro150/bin' >> ~/.bashrc
	echo 'export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so' >> ~/.bashrc

	# Test mjpro 150
	cd ~/.mujoco/mjpro150/bin
	./simulate ../model/humanoid.xml
	```
- Clone DPB
	```bash
	git clone git@github.com:jenior97/DPB.git
	```
- Anaconda environment
	```bash
	conda create -n DPB python=3.8
	```
- [mujoco-py 1.50.1.0](https://github.com/openai/mujoco-py/releases) 
	``` bash
	sudo apt-get install python3-dev build-essential libssl-dev libffi-dev libxml2-dev
	sudo apt-get install libxslt1-dev zlib1g-dev libglew1.5 libglew-dev python3-pip
	sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3
	
	cd DPB/mujoco-py-1.50.1.0

	pip3 install -r requirements.txt
	pip3 install -r requirements.dev.txt
	pip3 install -e . --no-cache

	```
- Requirements
	```bash
	cd ..

	pip3 install -r requirements.txt 
	conda install pymc
	```
- Dependencies
	```bash
	pip3 install numpy==1.21.3
	pip3 install arviz==0.12.1
	pip3 install scipy==1.7.3
	```


## Time-varying scenarios
The manipulation of time-varying scenarios can be effectively achieved by utilizing the "timevarying_true_w" function located in run_algo/algo_utils.py.
By default, this function uniformly samples the optimal weights within the range of [-1, 1].
Interesting time-varying scenarios can be acquired by managing the ranges effectively.

## Run
- Interpolated time-varying scenario
	```bash
	python3 run_experiment_interpolation.py 
	```
- Abrupt time-varying scenario
	```bash
	python3 run_experiment.py 
	```
- Static scenario
	```bash
	python3 run_experiment_time_invarying.py 
	```
- Hyper-parameters
	- ```--task-env``` : tosser, driver, avoid
	- ```--algo``` : DPB, batch_active_PBL
		- if DPB,
			please refer to the paper for a comprehensive understanding of the parameters below.
			- ```--exploration-weight``` : constant multiplied to $\alpha_{t-1}$ to effectively generate query sets.
			- ```--discounting-factor``` : $\gamma$ $\in$ (0,1]
			- ```--delta``` : probability to hold the inequality
			- ```--regularized-lambda``` : regularizer $\lambda$ 
		- if batch_active_PBL,
			- ```--BA-method``` : greedy, medoids, dpp, random, information, max_regret
	- ```--num-iteration``` : # of iteration
	- ```--num-batch``` : # of batch
	- ```--seed``` : random seed

## Acknowldegements
We would like to express our gratitude to the contributors upon whose work our code is based on:

E Bıyık and D Sadigh. **"[Batch Active Preference-Based Learning of Reward Functions](https://github.com/Stanford-ILIAD/batch-active-preference-based-learning)"**. *Conference on Robot Learning (CoRL)*, Zurich, Switzerland, Oct. 2018.


E Bıyık, K Wang, N Anari, D Sadigh, **"[Batch Active Learning using Determinantal Point Processes](https://github.com/Stanford-ILIAD/DPP-Batch-Active-Learning)"**. *arXiv preprint arXiv:1906.07975*, Dec. 2019.

E Bıyık, M Palan, N Landolfi, D Losey and D Sadigh. **"[Asking Easy Questions: A User-Friendly Approach to Active Reward Learning](https://github.com/Stanford-ILIAD/easy-active-learning)"**. *Conference on Robot Learning (CoRL)*,  Oct. 2020.