## Dataset preparation

Please refer to the following links to download two standard WiFi human activity recognition datasets and put them in the folder "./data". 

- **ARIL**

  The authors of [ARIL](https://github.com/geekfeiw/ARIL) provided the processed data as well as the original data, you can train the Recurrent ConFormer by them.  

  Meanwhile, we download the original dataset and perform the linear interpolation and split the dataset by ourselves. The processed data is available at [ARIL_RConFormer](https://drive.google.com/file/d/1h_F0_JRQ4Tx1IXKd9Kk9ZpMzx7QjT4VI/view).

- **UT-HAT**

  The original dataset is available at [UT-HAR](https://github.com/ermongroup/Wifi_Activity_Recognition).

  We follow the preprocessing procedure in [THAT](https://github.com/windofshadow/THAT). For your convenience, we provide the processed data followed by  [THAT](https://github.com/windofshadow/THAT) at [UT-HAR_RConFormer](https://drive.google.com/file/d/1-3cbFdlLuJdmdWCfbj2BE6VJf36nvjha/view).

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

