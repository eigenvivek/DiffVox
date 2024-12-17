DiffVox
================
[![Paper shield](https://img.shields.io/badge/arXiv-2411.19224-red.svg)](https://arxiv.org/abs/2411.19224)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

DiffVox is a self-supervised framework for Cone-Beam Computed Tomography (CBCT) reconstruction by directly optimizing a voxelgrid representation using physics-based differentiable X-ray rendering.


## Install

To install the latest release use [Conda](https://docs.conda.io/en/latest/miniconda.html):

```bash
git clone https://github.com/hossein-momeni/DiffVox.git
cd DiffVox
conda env create -f environment.yml
conda activate diffvox
pip install .
```

## Prepare the Dataset
To download the dataset, run the `data.sh` script located in the `data/` directory:

```bash
pip install zenodo_get
cd data
./data.sh
```
**Note**: This dataset is sourced from the study ["A cone-beam X-ray computed tomography data collection designed for machine learning"](https://www.nature.com/articles/s41597-019-0235-y). It comprises 48 walnuts, each with approximately 3,600 high-resolution X-ray projections. The download requires around 300 GB of storage and may take approximately 10 hours, depending on your internet speed.


After downloading the dataset you can reconstruct the ground truth volumes using [`slurm`](https://slurm.schedmd.com/):

```bash
srun python utils/construct_ground_truth.py -d data
```
This runs in about ~4 min / walnut on an NVIDIA TITAN Xp.

## Reconstruction

To reconstruct the walnuts using `diffvox`, you can use the script `walnut_recon.py`. For example, to reconstruct walnut ID `3` with `15` views using trilinear interpolation, run the following command:

```bash
python walnut_recon.py --walnut_id 3 --n_views 15
```
You can customize the reconstruction experiments further by using the following flags:
<details> <summary>Click to reveal parameters</summary>


1. `--walnut_id (int)`: ID of the walnut dataset to use for reconstruction. Default: `3`.
2. `--n_views (int)`: Number of X-ray views to use for reconstruction. Increasing this can improve reconstruction quality but increases computation time. Default: `15`.
3. `--downsample (int)`: Factor by which to downsample the supplied X-ray images. Use this to reduce computational load. Default: `1` (no downsampling).
4. `--batch_size (int)`: Number of rays loaded into memory for each gradient step. Adjust based on your GPU memory capacity:
   - Example: For an NVIDIA RTX A6000 with 48GB memory:
     - Trilinear method: Up to `1,800,000` rays.
     - Siddon's method: Up to `500,000` rays.
   Default: `1_800_000`.
5. `--n_itr (int)`: Number of optimization iterations to perform. Default: `50`.
6. `--lr (float)`: Learning rate for the optimizer. Default: `0.01`.
7. `--tv_coeff (float)`: Weight coefficient for the total variation (TV) norm. Used to regularize the density map. Higher values encourage smoother reconstructions. Default: `15`.
8. `--shift (float)`: Shift parameter applied to the input before regularization using the density regulator. 
   - This modifies the input to `softplus` as `x - shift`, allowing fine-tuning of the density's baseline value. 
   - Useful for controlling where the density values start in the optimization process.
   Default: `0`.9. `--beta (float)`: Smoothing parameter for density regularization. Default: `20`.
9. `--beta (float)`: Smoothing parameter for the `softplus` function in the density regulator. 
   - A higher `beta` makes the `softplus` function sharper, approaching the behavior of a ReLU. 
   - Lower values smooth the transition, which can help with optimization stability.
   Default: `20`.
   **Usage:**  
   The density regularizer is defined as:
   ```python
   torch.nn.functional.softplus(x - shift, beta=self.beta, threshold=20)
10. `--loss_fn (str)`: Loss function to use for optimization. Options include:
    - `"l1"`: L1 loss
    - `"l2"`: L2 loss
    - `"pcc"`: Pearson Correlation Coefficient loss (*work in progress*) 
    - `"ncc"`: Normalized Cross-Correlation Loss
    - Default: `"l1"`.
11. `--renderer (str)`: Rendering method to use for generating the DRRs (Digitally Reconstructed Radiographs). Options include:
    - `"trilinear"`: Faster but less accurate.
    - `"siddon"`: Physics-based rendering method, slower but more accurate.
    - Default: `"trilinear"`.
12. `--n_points (int)`: Number of sampling points per ray in the volume. 
    - **Relevance:** This parameter is used **only** with the `trilinear` renderer to determine the number of points sampled along each ray.
    - **Ignored:** This parameter is ignored when using the `siddon` renderer since Siddon's method inherently calculates ray intersections based on the voxel grid structure.
    - A higher number of points may improve reconstruction quality for `trilinear` but increases memory and computational costs.
    Default: `500`.

13. `--drr_params (dict)`: Dictionary of parameters for the DRR generator(`DiffDrr`). Keys include:
    - `sdd` (float): Source-to-detector distance. Default: `199.006188`.
    - `height` (int): Height of the DRR image. Default: `768`.
    - `width` (int): Width of the DRR image. Default: `972`.
    - `delx` (float): Detector pixel spacing. Default: `0.074800`.
    - **Note**: These default values are calibrated specifically for walnut dataset reconstruction.

14. `--density_regulator (str)`: Regularization method for the density function. Options include:
    - `"softplus"`: Applies a softplus transformation.
    - `"sigmoid"`: Applies a sigmoid transformation.
    Default: `"softplus"`.

15. `--tv_type (str)`: Type of total variation regularization to apply. Options include:
    - `"vl1"`: Variation L1 norm.
    - `"vl2"`: Variation L2 norm.
    Default: `"vl1"`.

16. `--half_orbit (bool)`: Whether to use a half-orbit of X-ray views for reconstruction instead of a full orbit. Reduces the number of views required. Default: `False`.

17. `--drr_scale (float)`: Scale factor to apply to the generated DRRs. Default: `1.0`.

18. `--proj_name (str)`: Project name for organizing experiments, particularly when logging with WandB. Default: `"walnut_recon"`.

19. `--initialize_alg (str)`: Initialization algorithm for the voxel grid. Options include:
    - `"None"`: No specific initialization; the grid is initialized to zeros.
    - `"fdk"`: Use Filtered Back Projection (FDK) for initialization. Commonly used in CT reconstruction for quick, approximate results.
    - `"cgls"`: Use Conjugate Gradient Least Squares (CGLS) for initialization, an iterative reconstruction method.
    - `"sirt"`: Use Simultaneous Iterative Reconstruction Technique (SIRT) for initialization, known for its robust iterative refinement.
    - `"nesterov"`: Use Nesterov-accelerated gradient descent for initialization, providing faster convergence in optimization.
    Default: `"None"`.

20. `--log_wandb (bool)`: Whether to log experiment results to WandB. Default: `False`.
</details>

## Using your own dataset

To use your own dataset with DiffVox, you can create a subclass of `Dataset_DiffVox` and make a constructor (`__init__()`) that would handle your data. The Dataset should have the following parameters defined:

1. `gt_projs`: Ground truth projections.
2. `sources`: Source positions for the projections (in world coordinates).
3. `targets`: Target positions for the projections (in world coordinates).
4. `subject`: An instance of `torchio.Subject` representing the dataset subject.

By defining these attributes, you ensure that your dataset is compatible with DiffVox's processing pipeline.

## Citing `DiffVox`

If you find DiffVox useful in your work, please cite our [paper](https://arxiv.org/abs/2411.19224v2):

    @article{momeni2024voxel,
      title={Voxel-based Differentiable X-ray Rendering Improves Self-Supervised 3D CBCT Reconstruction},
      author={Momeni, Mohammadhossein and Gopalakrishnan, Vivek and Dey, Neel and Golland, Polina and Frisken, Sarah},
      booktitle={Machine Learning and the Physical Sciences, NeurIPS 2024},
      year={2024}
    }