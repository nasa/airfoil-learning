# Airfoil Learning
The goal of this repository is to outline a method of using graph neural networks and deep neural networks to predict the Lift and Drag of 2D Airfoils. Graph Neural Networks investigate the relationship between nodes through edges and edge attributes. Graph relationships are all around us, our social networks on Facebook, the products we purchase or are interested in on Amazon, molecular interactions. This project investigates using the connectivity of points that describe a 2D airfoil to predict performance. 

![image](https://user-images.githubusercontent.com/9328717/157048314-72a143ea-621b-405b-89b2-bdbb72d7ba33.png)

## Data Structure

* (Feature) Vertices (x,y,z)
* (Feature) Edge_Connectivity (edge_index1,edge_index2)
* (Feature) Reynolds - flow velocity
* (Feature) Ncrit - turbulence parameter
* (Feature) Alpha - angle of attack
* (Label) Cl - Coefficient of Lift
* (Label) Cd - Coefficient of Drag
* (Label) Cm - Coefficient of Moment 
* (Label) Cp - Pressure coefficient

Useful documentation on torch data object 
* [Data Class](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/data/data.html#Data)
* [In-Memory Datasets](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html#creating-in-memory-datasets)
* [Datasets](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html#creating-larger-datasets)

### Link to Dataset
**Unprocessed dataset** can be found at https://nasa-public-data.s3.amazonaws.com/plot3d_utilities/airfoil-learning-dataset.zip 
This dataset is not normalized and contains the geometry of each airfoil and the xfoil results. It's important to go through the process of normalizing the design. This will give you the scalars (scalers.pickle) used with the processed dataset. See tutorial (Normalization section). 

> Important note: Training can give you really low values for error and loss but all of that doesn't mean much until you do a santity check with a random design. You may find low error but a mistake in your normalization/unnormalization code will give you strange results and most definitely affects the training. Always do a santiy check and plot what it is you are interested in. 

**Processed and normalized dataset**:  https://nasa-public-data.s3.amazonaws.com/plot3d_utilities/dataset-processed.zip

> Note: This file may not give you the normalization scalars. It doesn't take long to run through the normalization process. The only parts that take a long time is Step4_CreateDataset.py (This is used to create the **Processed and normalized dataset**) AND training of the graph neural network. 

## Google Colab Tutorial
[Link to tutorial]()

### Regarding use of Jupyter Notebooks

Jupyter notebook is great from demonstrations, homework, interview questions, but it should never be used for real development work. I have witnessed interns who do development work in Jupyter because they do not know how to debug python with vscode, pycharm, or any editor. Their notebooks are massive 1000+ lines of code and output. They broke something in cell 20, ended up fixing it in cell 50 and when they run through the code from start to finish it's broken from cell 21 to 49. Weeks go by and still they couldn't figure it out. End of internship and the notebook is still broken. Notebooks are never mean for code development, only for show and tell. 

For python development, I reccomend the following:
- Visual studio code
    - Extensions:
        - autodocstring
        - better comments
        - docker
        - python
        - pylance
        - remote ssh 
        - restructuredtext
        - syntax highlighting 
- anaconda (learn how to make virtual environments) 
- Learn best practices of debugging python with visual studio code from youtube.


## Reporting Bugs 
To report bugs, add a github issue. Instructions for adding github issues: https://www.youtube.com/watch?v=TKJ4RdhyB5Y

## Repository Walk through
Below are links to two interactive walk throughs detailing how to reproduce the results of this work. 
1. Generating Airfoil Designs [Generate Xfoil](https://github.com/nasa/airfoil-learning/tree/main/generate_xfoil)
2. Training Graph Netwworks and MultiLayer Perception (MLP) [pytorch](https://github.com/nasa/airfoil-learning/tree/main/pytorch)

## Technical Reports
A link to the publication will be included here once it's been published. 

## Tutorial

# License
[NASA Open Source Agreement](https://opensource.org/licenses/NASA-1.3)
