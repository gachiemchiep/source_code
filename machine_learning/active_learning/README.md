A reddit post  : https://www.reddit.com/r/computervision/comments/l695g1/you_should_try_active_learning/

-> Roughly yeah, that's one way of doing it. Once you find the high loss / poorly performing examples in your dataset (the green cones example), go find more of those examples in your production datastream to label, retrain on the new dataset, and usually the new model should be better than the old one.

-> find the high performing examples inside dataset 
-> remove low performing examples and create new dataset
-> retrain model -> this is the new, better one

original artical at this : 

    https://medium.com/aquarium-learning/you-should-try-active-learning-37a86aab1afb

    https://www.aquariumlearning.com/

-> can this work on big dataset like imagenet
-> is this really useful

https://www.youtube.com/watch?t=513&v=hx7BXih7zx8&feature=youtu.be&ab_channel=Matroid
    -> HydraNet : shared backbone, multiple task

    -> have data labelling -> ...
    -> divide new task into sub division of old task
    -> map car into global 3d map : BEV Net (Bird eyey view network)
