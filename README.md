# Recurrent ConFormer for WiFi Activity Recognition

<div align="justify">
  This is the official implementation of our IEEE/CAA JAS 2023 paper "Recurrent ConFormer for WiFi Activity Recognition".
In this paper,  we propose a lightweight and efficient Recurrent model of CONvolution blocks and transFORMER encoders (ReConFormer) for HAR using CSI signals. The proposed recurrent model not only combines the advantages of CNN and transformer but also builds a deep enough structure with a fixed number of parameters. Results of comparative experiments on ARIL and UT-HAR indicate the superiority of the proposed method in both accuracy and efficiency. 
</div>



**Recurrent ConFormer for WiFi Activity Recognition**<br>
Miao Shang, Xiaopeng Hong. IEEE/CAA Journal of Automatica Sinica. (JAS 23). <br>
[[Paper]] (DOI：10.1109/JAS.2023.123291)


## Introduction

<div align="justify">
We focus on Human Activity Recognition (HAR) task by using WiFi signals. A new lightweight and efficient deep learning framework namely Recurrent ConFormer is proposed and facilitates the study of Channel State Information (CSI)-based HAR methods by making the following contributions.
Firstly, we incorporate the recurrent mechanism into the transformer encoder and propose the recurrent transformer module,  which allows for building the deep architecture with a fixed number of parameters. Secondly, we propose to cascade the recurrent convolution and transformer modules. This architecture captures the local variation by convolution blocks and models the long-range dependencies among local features by transformer encoders. To the best of our knowledge, this is the first time to combine the recurrent mechanism with the cascaded CNN and transformer for CSI-based recognition.

Recurrent ConFormer consists of three main components: recurrent CNN, recurrent Transformer, and classifier head. We implement the three components on two public CSI-based HAR datasets, i.e. ARIL and UT-HAR. Extensive experimental results demonstrate the effectiveness of our Recurrent ConFormer framework.

</div>

<img src=".\figs\fig2.png" alt="fig" width="500px"/>       <img src=".\figs\fig3.png" alt="fig" width="500px" />                                                                               

<center><p>Fig. The overall architecture of recurrent CNN and recurrent Transformer.</p></center>



## Requirements
- python 3.9 (We recommend to use Anaconda, since many python libs like numpy and sklearn are needed in our code.)

- [PyTorch 1.12.0](https://pytorch.org/) (we run the code under version 1.12.0 with **gpu**)  

  

- More requirements can be seen in **requirements.txt**, use the following command to install.

  ```
  pip install -r requirements.txt
  ```

  

- The **virtual environment** for Recurrent ConFormer is also provided **for convenience**. (optional)

  Create the virtual environment for Recurrent ConFormer.

  ```python
  conda env create -f environment.yaml
  ```

  After this, you will get a new environment that can conduct Recurrent ConFormer experiments.  
  Run `conda activate ` to activate.

  Note that only NVIDIA GPUs are supported for the code, and we use  NVIDIA GeForce RTX 3060. 



## Dataset preparation
Please refer to the following links to download two standard WiFi human activity recognition datasets and put them in the folder "./data". 

- **ARIL**

  The authors of [ARIL](https://github.com/geekfeiw/ARIL) provided the processed data as well as the original data, you can train the Recurrent ConFormer by them.  

  Meanwhile, we download the original dataset and perform the linear interpolation and split the dataset by ourselves. **The processed data is available at [ARIL_RConFormer](https://drive.google.com/file/d/1h_F0_JRQ4Tx1IXKd9Kk9ZpMzx7QjT4VI/view).**

- **UT-HAT**

  The original dataset is available at [UT-HAR](https://github.com/ermongroup/Wifi_Activity_Recognition).

  We follow the preprocessing procedure in [THAT](https://github.com/windofshadow/THAT). For your convenience, **we provide the processed data followed by  [THAT](https://github.com/windofshadow/THAT) at [UT-HAR_RConFormer](https://drive.google.com/file/d/1-3cbFdlLuJdmdWCfbj2BE6VJf36nvjha/view).**

The folder of the datasets is as follows:
```
data
├── ARIL
│   ├── linear_train_data.mat
│   └── linear_test_data.mat
├── UT_HAT
│   └── Data.pt
├── samples
... ...
```



## Training

Please change the `data_path` in the config files to the locations of the datasets。  

Currently, there are two options for `dataset_type` in the config files: `ARIL` and `UT-HAR`.    

Feel free to change the parameters in the config files, and run  `main.py`  to reproduce the main results in our paper:

```python
# for ARIL dataset
python main.py --config configs/ARIL.json

# for UT-HAR dataset
python main.py --config configs/UT-HAR.json
```

## Evaluation

Please refer to 
[[Evaluation Code]](https://github.com/infinite0522/Recurrent-ConFormer-for-WiFi-HAR/blob/main/inference.py).


## Results

<img src=".\figs\result1.png" alt="results1.png" width="500px"/>

<img src=".\figs\result2.png" alt="results2.png" width="500px"/>



## License

Please check the MIT  [license](./LICENSE) that is listed in this repository.

## Acknowledgments

We thank the following repos providing helpful components/functions in our work.

- [ResNet1D](https://github.com/geekfeiw/ARIL)  
- [THAT](https://github.com/windofshadow/THAT)

## References

[1] F. Wang, J. Feng, Y. Zhao, X. Zhang, S. Zhang, and J. Han, “Joint activity recognition and indoor localization with wifi fingerprints,” IEEE Access, vol. 7, pp. 80 058–80 068, 2019.

[2] S. Yousefi, H. Narui, S. Dayal, S. Ermon, and S. Valaee, “A survey on behavior recognition using wifi channel state information,” IEEE Commun. Mag., vol. 55, no. 10, pp. 98–104, 2017.

[3] B. Li, W. Cui, W. Wang, L. Zhang, Z. Chen, and M. Wu, “Two-stream convolution augmented transformer for human activity recognition,” in Proc. AAAI Conf. Artificial Intelligence (AAAI), vol. 35, no. 1, 2021, pp. 286–293. 

## Citation

If you use any content of this repo for your work, please cite the following bib entry:
```
@inproceedings{shang2023RConFormer,
  title={Recurrent ConFormer for WiFi Activity Recognition},
  author={Shang, Miao and Hong, Xiaopeng},
  booktitle={IEEE/CAA Journal of Automatica Sinica. (JAS)},
  year={2023}
}
```
