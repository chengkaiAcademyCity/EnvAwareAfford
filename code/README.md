# Experiments
This folder includes the codebase for our simulator and experiments.

## Before start
To train the models, please first go to the `../data` folder and download PartNet-Mobility Dataset and the pre-processed ShapeNet models at https://sapien.ucsd.edu/downloads and http://download.cs.stanford.edu/orion/o2oafford/sapien_dataset.zip. 


## Dependencies
This code has been tested on Ubuntu 20.04 with Cuda 11.7, Python 3.7, and PyTorch 1.13.1.

First, install SAPIEN following

    pip install http://download.cs.stanford.edu/orion/where2act/where2act_sapien_wheels/sapien-0.8.0.dev0-cp36-cp36m-manylinux2014_x86_64.whl

For other Python versions, you can use one of the following

    pip install http://download.cs.stanford.edu/orion/where2act/where2act_sapien_wheels/sapien-0.8.0.dev0-cp35-cp35m-manylinux2014_x86_64.whl
    pip install http://download.cs.stanford.edu/orion/where2act/where2act_sapien_wheels/sapien-0.8.0.dev0-cp37-cp37m-manylinux2014_x86_64.whl
    pip install http://download.cs.stanford.edu/orion/where2act/where2act_sapien_wheels/sapien-0.8.0.dev0-cp38-cp38-manylinux2014_x86_64.whl

Please do not use the default `pip install sapien` as SAPIEN is being actively updated.

Then, if you want to run the 3D experiment, this depends on PointNet++.

    git clone --recursive https://github.com/erikwijmans/Pointnet2_PyTorch
    cd Pointnet2_PyTorch
    # [IMPORTANT] comment these two lines of code:
    #   https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/pointnet2_ops_lib/pointnet2_ops/_ext-src/src/sampling_gpu.cu#L100-L101
    pip install -r requirements.txt
    pip install -e .

Finally, run the following to install other packages.
   
    # make sure you are at the repository root directory
    pip install -r requirements.txt

to install the other dependencies.

For visualization, please install blender v2.79 and put the executable in your environment path.
Also, the prediction result can be visualized using MeshLab or the *RenderShape* tool in [Thea](https://github.com/sidch/thea).

## Simulator
You can run the following command to test and visualize a random interation in the simulation environment.

    python collect_data.py --out_dir results/ 33810 Table 0 Null 0 pushing --data_split test_cat --max_num_occlusions 8

Change the shape id to other ids for testing other shapes, 
and modify the primitive action type to : *pushing or pulling*. 

After you ran the code, you will find a record for this interaction trial under `./results/33810_Table_0_Null_0_pulling_0`, from where you can see the full log, 2D image, 3D depth and interaction outcome.

If you want to run on a headless server, simple put `xvfb-run -a ` before any code command that runs the SAPIEN simulator.
Install the `xvfb` tool on your server if not installed.

## Generate Offline Training Data
Before training the network, we need to collect a large set of interaction trials via random exploration, using the script `scripts/run_gen_offline_data.sh`.

Generating enough offline interaction trials is necessary for a successful learning, and it may require many CPU hours (e.g. 10,000 hrs or more) for the data collection.
So, this offline data collection script is designed for you to parallelize the data generation on different machines and many CPU cores, by setting the proper `--starting_epoch`, `--num_epochs`, `--out_fn` and `--num_processes` parameters.
After the data generation, you need to move all the data to the same folder and create one `data_tuple_list.txt` file merging all output data index files.
Check the parameters for more information.

    python gen_offline_data.py --help

In our experiments, we train one network per primitive action.

## Train the Network

Using the batch-generated interaction data, run the following to train the network

    bash scripts/train.sh

Change to other environment names to train for other tasks.

The training will generate a log directory under `./logs/[exp-folder]/`.

## Evaluate the Network

To test and evaluate the results, run

    bash scripts/eval.sh
    
to visualize the predicted affordance maps.
The results are generated under `logs/[exp-folder]/[result-folder]/`.

## External Libraries

This code uses the following external libraries (all are free to use for academic purpose):
   * https://github.com/sidch/Thea
   * https://github.com/erikwijmans/Pointnet2_PyTorch
   * https://github.com/haosulab/SAPIEN-Release
   * https://github.com/KieranWynn/pyquaternion
   * https://github.com/daerduoCarey/where2act

We use the data from ShapeNet and PartNet, which are both cited in the main paper.