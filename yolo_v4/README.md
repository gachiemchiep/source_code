# Yolov4 Related


## OpenCV inferencing



### Re-write postprocess

```bash
# Ver 0 : use opencv version
1 loop, best of 3: 27.4 s per loop
-> postprocess = 27.4 / 100 = 0.274 sec = 274 ms
	-> 91 ms per image

# Ver 1 : use numpy argmax (remove the opencv for loop)
1 loop, best of 3: 1.46 s per loop
-> postprocess = 1.46 / 100 = 0.0146 = 14.6 ms
	-> 5ms per image

# Ver 2 : merge features and do postprocess for each image
1 loop, best of 3: 1.54 s per loop
-> postprocess  = 1.54 / 100 = 0.0154 = 15.4 ms
	-> 5ms per image

# Ver 3: merge all features, use confidence threshold to remove invalid bbox, then do nms for each image
# note : each image has different valid bboxes. so in the last step we must use the for loop
1 loop, best of 3: 1.45 s per loop
-> post process = 1.45 / 100 = 0.0145 = 14.5 ms

-> 1, 2, 3 is very close
-> Ver 3 is fast but very hard to mainternance the code . 
So we will use version 2 instead

```