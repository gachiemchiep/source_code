# 3次元点軍解析に関する色々

##  資料一覧

[UCI_3DL_talk.pdf ](https://drive.google.com/u/0/uc?export=download&confirm=Gfo4&id=19aOuyX7-nMSdF1fZ7E2QtNtXGV10vTaU)
 [source](https://github.com/timzhang642/3D-Machine-Learning/issues/65#issuecomment-617471300) :  
    -> good introduction for beginner

```bash
3d deep learning framework : 
    TensorFlow Graphics
    Kaolin (by Nvidia)
    Pytorch 3d 
    PyTorch Geometric

# Some keywords
Shape representation : RGB image, depth image, voxel, SDF, Points, Mesh

Object classification : 3DShapeNet. PointNet
Segmentation : PointNet, PointNet++, Dynamic GraphCNN
Single-View 3D Reconstruction : 3D-R2N2, OctNet, AtlasNet, Pixel2Mesh, MeshRCNN
ShapeAbstraction 
```


[CreativeAI: Deep Learning for Computer Graphics](https://geometry.cs.ucl.ac.uk/creativeai/)

```bash
• Provide an overview of the popular ML algorithms used in CG
• Provide a quick overview of theory and CG applications
• Many extra slides in the course notes + example code
• Summarize progress in the last 3-5 years

-> DL have a lot of application in CG too 

Part 3: Supervised Learning in CG
Style Transfer Applications
Sketch Simplification
Denoising Renderings
Image Decomposition
3D CNN: Object Recognition  
VoxNet: Object Recognition 
Multi-view CNN for 3D
Mesh Labeling / Segmentation
Colorization
Single-image SVBRDF Capture
Realistic Reconstructions
Differentiable Rendering: Rendering in the Loop
Design Options
Learning Volumetric Deformation
Single Image Facial Relighting

Part 4: Unsupervised Learning in CG
GAN : StyleGAN, Conditional GAN: Pix2Pix, ... 
-> there are few good points at the summary slide

Part 5: Learning on Unstructued Data
    -> point cloud, surface mesh, volume metric, multi-view images
    -> PointNet , PointNet++
    -> FlowNet3D,  ShapeNet

Part 6: Learning for Simulation/Animation
    -> deep learning can be used for simulation and animation too
```