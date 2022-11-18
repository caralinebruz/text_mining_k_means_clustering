# Text Mining & K-Means Clustering 

Predictive Analytics class assignment 1

Directions to run:
1. Install the required packages
`pip3 install requirements.txt`
2. Execute 
`python3 text_mining.py`



## Usage Notice
The images will render one after the other. They will not render at the same time. You can save the image and view them side by side for comparison.
When done viewing the images, close them out to continue the script.


Confusion matrix runs after the visualizations, so the images  need to be closed in order for the rest of the program to run.


## Known Issues
There is an issue with the consistency of my clusters. Initially, I pick three starting centroids at random, the centroids don't always converge to the same values each time I run my program. Sometimes the clusters are fully balanced (8 in each cluster) and other times there might be only one node in a cluster at the end! Not sure what to do about this.

_Need to figure out how to get consistency or how to pick better starting centroids._


### Example output
```
Confusion Matrix:
[[6 2 0]
 [0 7 1]
 [0 0 8]]

Precision: 0.8888888888888888
Recall: 0.875
F1 Score: 0.8818897637795274
```


![Alt text](https://github.com/caralinebruz/pa_py_assign1/blob/main/actual_3.png?raw=true)
![Alt text](https://github.com/caralinebruz/pa_py_assign1/blob/main/predicted_3.png?raw=true)
