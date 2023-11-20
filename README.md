# DiffSmooth: Certifiably Robust Learning via Diffusion Models and Local Smoothing

   ## About

   This repository houses the code and other resources for the paper titled [**DiffSmooth: Certifiably Robust Learning via Diffusion Models and Local Smoothing**](https://www.usenix.org/system/files/usenixsecurity23-zhang-jiawei.pdf), accepted at the 32nd USENIX Security Symposium, 2023.

   ## Introduction

   DiffSmooth improves the robustness of machine learning models against adversarial attacks. Utilizing a two-fold strategy, it first applies diffusion models for adversarial purification and then enhances robustness via local smoothing. Our SOTA results include a certified accuracy boost from $36.0\% to $53.0\%$ under $\ell_2$ radius $1.5$ on ImageNet.

   ### Parameters
    - denoising_sigma: specify the magnitude for added Gaussian noise (can take in range 0.00~0.50)
    - local_sigma: specify the magnitude of the local smoothing Gaussian noise after purification (can take in 0.06, 0.12, 0.25), no need to be the same with denoising_sigma
    - num_samples: the n0 parameter in randomized smoothing (i.e., the ensemble for prediction)
    - local_vote: the number of local smoothing noise (the `m' in the paper, default: 5)

## Citation

   If you find our work beneficial, please consider citing our paper:

   ```
@inproceedings {287372,
  author = {Jiawei Zhang and Zhongzhu Chen and Huan Zhang and Chaowei Xiao and Bo Li},
  title = {{DiffSmooth}: Certifiably Robust Learning via Diffusion Models and Local Smoothing},
  booktitle = {32nd USENIX Security Symposium (USENIX Security 23)},
  year = {2023},
  isbn = {978-1-939133-37-3},
  address = {Anaheim, CA},
  pages = {4787--4804},
  url = {https://www.usenix.org/conference/usenixsecurity23/presentation/zhang-jiawei},
  publisher = {USENIX Association},
  month = aug,
}
   ```

## Contact

Thank you for your interest in DiffSmooth!

If you have any questions or encounter any errors while running the code, feel free to contact [jiaweiz7@illinois.edu](mailto:jiaweiz7@illinois.edu)!
