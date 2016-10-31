# Bubble Size Analysis (BSA)

This repository contains a selection of tools ready to be used for bubble detection.
Here we want to provide what we experienced to be a functional sequence of commands that can be quickly aplied for image analysis in bubble detection.

### Packages used 
- opencv
- skimage

### Requirements
At the moment only fully compatible for Python 3.5

## The Idea
The user is provided with a number of tools that have been already adjusted to be applicable to the purpose of bubble detection, so to shorten the time expended in developing your own algorithm and parameter selection.
As a matter of fact, the time needed for this process is not negligible and this wants to be a starting platform to efficiently start to detect bubbles.

### Run a pipeline
The user can have a first idea of how the package works on her/his sample image by running one of the two default pipelines:

- Canny pipeline:
	apply a Cany filter for edge detection on the whole image and perform the following steps


- Adaptive threshold pipeline: 
	apply the adaptive threshold method of opencv with default or chosen parameters for the gaussian edge detection. In this case the sequece of steps becomes:

See example 1 in example_bubble.py

### Perform an individual sequence
You can set your object and directly operate the different functions available in BubbleKicker in order to find your own perfefct sequence of processing steps

Each function directly changes your object, but the function what_have_i_done helps in wrapping up what were the steps taken till that moment. 
In any case you can always come back to the origin with the reset_to_raw function.

See example 2 in example_bubble.py

### Running a pipeline on a bunch of files/directories
Normally the analysis of bubbles is taking place on tons of images, thus with batch bubblekicker one can find the possibility to run a complete pipeline with custom parameters on an entire folder. 

See expample 3 in example_bubble.py

### Define Bubbles properties
Once the detection of bubbles has come to a satisfying end, you can proceed on defining the interesting bubbles properties.

NOTE: here the last crucial step occurs, the filteriing of objects which are not possibly real bubbles

This step is considered as a post-processing of the imaging steps taken so fa, since blobls are labeled and characterized for specific properites that are known to be efficient filtering parameters.
In specific, the circularity reciprocal and convexity are used to recognize those objects that are not classifable as bubbels.

Once more, default and custom filter rules can be set.
By default, it was observed that circularity reciprocal {'min': 0.92} and convexity {'max': 1.6, 'min': 0.2} were very efficient parameters.

