
![pull request image](https://github.com/Wolesword/Wine-Quality/edit/main/Wine.PNG)

# Wine Quality Prediction

## Technology and Resources Used

**Python Version**: 3.7.7

## Table of Contents
1) [Define the Problem](#1.-Define-the-Problem)<br>
2) [Gather the Data](#2.-Data-Set-Information)<br>
3) [Prepare Data for Consumption](#3.-Prepare-Data-for-Consumption)<br>
4) [Data Cleaning](#4.-Data-Cleaning)<br>
5) [Data Exploration](#5.-Data-Exploration)<br>
6) [Feature Engineering](#6.-Feature-Engineering)<br>
7) [Model Building](#7.-Model-Building)<br>
8) [Hyperparameter Tuning](#8.-More-Hyperparameter-Tuning)<br>
9) [End](#END)<br>

## 1. Define the Problem
Two datasets are included, related to red and white vinho verde wine samples, from the north of Portugal. The goal is to model wine quality based on physicochemical tests. In this study we evaluate the quality grade of Portugese white wine.
The Solution proposed classifies the white into four categories namely:
1. Excellent Wine - Cat 4
2. Very Good wine - Cat 3
3. Fairly good Wine - Cat 2
4. Bad Wine - Cat 1

## 2. Data Set Information
The two datasets are related to red and white variants of the Portuguese "Vinho Verde" wine. For more details, consult: [Web Link] or the reference [Cortez et al., 2009]. Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output) variables are available (e.g. there is no data about grape types, wine brand, wine selling price, etc.).

These datasets can be viewed as classification or regression tasks. The classes are ordered and not balanced (e.g. there are many more normal wines than excellent or poor ones). Outlier detection algorithms could be used to detect the few excellent or poor wines. Also, we are not sure if all input variables are relevant. So it could be interesting to test feature selection methods..

![image.png](attachment:image.png)

**Source:**: P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properties.
In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

Available at: https://archive.ics.uci.edu/ml/datasets/wine+quality
