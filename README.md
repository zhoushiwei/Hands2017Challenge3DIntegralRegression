# Hands2017Challenge3DIntegralRegression

This is the implementation of 3D Integral Regression (including coarse to fine 3D heatmap) for the Hands2017 challenge dataset.

The dataset is released in the workshop of ICCV 2017.

The backbone network is 2-stage Stacked Hourglass. Many thanks to Guanghan Ning's caffe Hourglass version
  https://github.com/Guanghan/GNet-pose. 
  
About 3D integral regression (originally devised for human pose estimation):

@article{sun2017integral,
  title={Integral Human Pose Regression},
  author={Sun, Xiao and Xiao, Bin and Liang, Shuang and Wei, Yichen},
  journal={arXiv preprint arXiv:1711.08229},
  year={2017}
}

About caffe Hourglass

@article{ning2017knowledge,
  title={Knowledge-Guided Deep Fractal Neural Networks for Human Pose Estimation},
  author={Ning, Guanghan and Zhang, Zhi and He, Zhiquan},
  journal={IEEE Transactions on Multimedia},
  year={2017},
  publisher={IEEE}
}