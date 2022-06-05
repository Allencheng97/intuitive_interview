# intuitive_interview
Bob the bird keeper has a problem that he thinks you can help him with. His birds are constantly harassed by 
the neighbor’s cats who are looking for a quick dinner. To protect his birds, Bob has setup a camera triggered 
alarm that goes off when a cat is nearby. To help Bob, please design and implement an automated algorithm that 
triggers when a cat is in view of the camera. To reassure him that his birds are alive and well, please also 
detect if a bird is present in the frame of view.

This problem can summary into below:
1.For given picture, the model can classify whether this is a picture of cat or bird
2.For special case, it is a photo of birds and cats(edge case)

The given dataset is too dirty,  both the training and test sets contain incorrect images that need to be corrected manually.

sol1 Treat this problem as a classification problem that use a  modified resnet 18 as backbone.
Train on the given dataset.It can distingusih whether this is cat or bird.
If we want to achieve function  2, we just need to modify our outpit as logit.（1*2 array）
For a given input, we set a threshold Q, if both number of output logit is beyond the Q.
We think it is a photo of bird and cat

sol2 Treat this problem as a detection problem.Use yolov3 pretrained model,Since our data doesn't have
bounding box groud truth, we can't train unless manually label all data.
For detection problem, we don't need to care problem 2.

sol3 use Contrastive Language–Image learning.Set three text categories, 
a picture of a cat, a picture of a bird, a picture of a cat and a bird.
Then all test images are encoded and the class with the greatest similarity is selected 
for output. Easy to use, highly extensible, and at the same time up to 98% accuracy.
