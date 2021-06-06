# bottleneck-rewiring
This code is to replicate the curvature-based rewiring method and figures found in the paper "Graph bottlenecks and curvature-based rewiring". The synthetic dataset is generated and used in `replicate_figures.ipynb`.

# Environment setup
Before running the Jupyter notebook for the first time, run these commands to set up the conda enviroment as needed. PyTorch Geometric is only needed for its implementation of GDC.
```
conda create -n env_gbcbr python=3.6 numpy pandas networkx matplotlib seaborn jupyterlab
conda activate env_gbcbr

conda install pytorch torchvision torchaudio cpuonly -c pytorch
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.1+cpu.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.1+cpu.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.1+cpu.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.1+cpu.html
pip install torch-geometric
```
