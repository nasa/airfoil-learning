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


# Wiki links
[Getting Started with your Environment](https://gitlab.grc.nasa.gov/machine-learning/graph-networks/airfoil-learning/-/wikis/1.0-Getting-Started)

# Repository Walk through
1. Generating Airfoil Designs [Generate Xfoil](https://github.com/nasa/airfoil-learning/tree/main/generate_xfoil)
2. Training Graph Netwworks and MultiLayer Perception (MLP) [pytorch](https://github.com/nasa/airfoil-learning/tree/main/pytorch)

# Link to Dataset
Dataset can be found at https://nasa-public-data.s3.amazonaws.com/plot3d_utilities/airfoil-learning-dataset.zip 

# Reporting Bugs 
To report bugs, add a github issue. Instructions for adding github issues: https://www.youtube.com/watch?v=TKJ4RdhyB5Y

# License
[NASA Open Source Agreement](https://opensource.org/licenses/NASA-1.3)
