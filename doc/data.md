## ExPose Model - Documentation
For suggestions on improving documentation, please contact [expose@tue.mpg.de](mailto:expose@tue.mpg.de).

Once you download and extract the zip with the pre-trained model you should have the following files:
* all_means.pkl : The mean pose parameters, which are used as the initial point for the iterative regression, in different pose representations ( axis-angle, PCA for the hands only, etc).
* shape_mean.npy: The mean shape parameters used to initialize the iterative regressor.
* SMPLX_to_J14.pkl: A linear regressor that computes the 14 LSP-like joints used to compute the mean per-joint point error (MPJPE).
* conf.yaml: Contains all the arguments needed to run ExPose.
* checkpoints: The pre-trained checkpoint.
* ExPose Dataset - Documentation

### Curated fits
Downloading and extracting the curated fits zip should give you the following
two files:
* train.npz
    * img_fns: The name of the image to read.
    * betas: A Nx10 numpy array with the shape coefficients of each instance.
    * expression: A Nx10 numpy array with the expression coefficients of each instance.
    * keypoints2D: The OpenPose keypoints used to generate the fits.
    * pose: A numpy array that contains the estimated SMPL-X pose vector in axis-angle format.
* val.npz
    * img_fns: The name of the image to read.
    * betas: A Nx10 numpy array with the shape coefficients of each instance.
    * expression: A Nx10 numpy array with the expression coefficients of each instance.
    * keypoints2D: The OpenPose keypoints used to generate the fits.
    * pose: A numpy array that contains the estimated SMPL-X pose vector in axis-angle format.
    * vertices: A numpy array that contains the estimated SMPL-X vertices.
    * joints: The 14 LSP-like joints used to compute the mean per-joint point error metric.

### SPIN in SMPL-X

The data format is exactly the same as the one in SPIN, see the [original page](https://github.com/nkolot/SPIN#final-fits) for more details.
