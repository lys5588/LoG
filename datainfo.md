
### dataset info

sample feicuiwan 
images:180
points:132070
image_size:6144*1096

yingrenshi_downsample
image:855
points:~660000
image_size:2736 × 1824


### large scene econfiguration
to run the large scene calibration 
imutils should be installed
timm also

the large scene preprocessing contain 5 step(dataset do not contain the gps)
1. feature extraction
2. use vocal tree to match the feature
3. mapper
4. calibration the coordinate
5. use midas model to calculate the depth map