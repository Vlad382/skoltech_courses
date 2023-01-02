## Final Grade: A (88.70%)

### Homework 1. Image Classification. 12.45/15 pts

Tiny Imagenet classification task. The score depended on final accurcay (up to 50.00 %). Achieved accuracy: 42.72 %

### Homework 2. 17/20 pts

-3 pts: submitted notebook had no log of transformers' training, only result metrics. 🙃

### Homework 3. 19.5/20 pts

-0.5 pts: one of the validation metrics (FID) produced wrong test accurcay.

### Homework 4. 15/15 pts

### Final Project. 24.75/30 pts

[GitHub repo of the project](https://github.com/Vlad382/vo_exploring)

Deep Learning for Visual Odometry

*Enhance classical visual odometry approach (ORB SLAM) by masking transparent objects and picture deblurring. Enhance deep learning by introducing new (for this task) dense map -- CoordConv. Comapre the approaches with each other and with baselines*

**Results**

 - Deblurring preprocess did not enhance the performance of Visual Odometry in each case. However, it added a remarkable amount of extra running time to a whole pipeline
 - Transparent object masking did not work on outdoor dataset (Kitti). On indoor dataset (TUM) the metrics did not suggest a significant improvement
 - The Deep Learning for Visual Odometry with CoordConv need more further investigation. Fine-tuning did not provide any improvement (no matter of weights initialization approach). Complete re-train needed to check the hypothesis.