### RelightVid
生成 Recamera 结果：
输入固定取前 81 帧（前 81 个相机位姿）、832x480 分辨率。
如输入长度不足使用 `change_camera_framecount.py` 和 `change_video_framecount.py` 进行插帧：
```bash
python3 change_camera_framecount.py -i test_data_acc/cameras.json -o test_data_acc/cameras_resampled.json -m 81
python3 change_video_framecount.py -i /home/csm/ReCamMaster/test_data_acc/luo.mp4 -o ./test_data_acc/luo2.mp4 -m 81
```

数据路径格式：
```
ReCamMaster/
├── data/
│   ├── cameras.json # 相机参数
│   ├── metadata.csv # 数据集信息表
│   ├── video_0.mp4
│   ├── video_1.mp4
│   └── ...
```
metadata.csv内容如下（文件名 + Video Caption， Video Caption 可使用语言模型生成）：
```csv
mask,prompt
luo.mp4,"A video scene showing two men ..."
```


使用 `inference_recammaster_acc.py` 进行生成，使用 `--cam_type custom`。
```bash
python3 inference_recammaster_acc.py --dataloader_num_workers 0 --cam_type custom --dataset_path ./test_data_acc
```



模型路径：
```
ReCamMaster/
├── models/
│   ├── ReCamMaster/
│   │   └──  checkpoints/
│   │        ├── Put ReCamMaster ckpt file here.txt
│   │        └── step20000.ckpt
│   └── Wan-AI/
│       └── Wan2.1-T2V-1.3B/          
```