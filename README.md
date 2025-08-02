<p align="center">
  <h1 align="center">ESMStereo-RS: Enhanced ShuffleMixer Disparity Upsampling with RealSense Inference Support</h1>
  <p align="center">
    Custom Implementation based on ESMStereo by Mahmoud Tahmasebi, Saif Huq, Kevin Meehan, Marion McAfee
  </p>
  <h3 align="center"><a href="https://arxiv.org/abs/2506.21091">Original Pre-print</a>
</p>

<p align="center">
  <img src="https://github.com/M2219/ESMStereo/blob/main/imgs/Graphical_abstract.png" alt="Logo" width="80%">
</p>

💡 Real-time stereo depth estimation with Intel RealSense support  
💡 Inference using `latest.py` script  
💡 Maintains high performance on KITTI and SceneFlow  
💡 Optimized for Jetson AGX and high-end GPUs  

---

## 🆕 RealSense Inference (New Feature)

### Requirements
- Intel RealSense camera (e.g., D435i)
- `pyrealsense2` library
- Replace inference script with `latest.py`

### Run Inference
```bash
python3 latest.py
