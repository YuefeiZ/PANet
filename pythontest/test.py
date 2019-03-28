import tensorflow as tf
import numpy as np
import tensorflow.contrib.eager as tfe

def log2_graph(x):
    """Implementation of Log2. TF doesn't have a native implementation."""
    return tf.log(x) / tf.log(2.0)

tf.enable_eager_execution()
#boxes: [batch, num_boxes, (y1, x1, y2, x2)]

#print(tf.constant([480,360,480,360,480,360,160,160],shape=[1,2,4],dtype=tf.float32))
p2 = tf.ones([4,128,128,3], dtype=tf.uint8)
p2 = p2
p3 = tf.ones([4,64,64,3], dtype=tf.uint8)
p4 = tf.ones([4,32,32,3], dtype=tf.uint8)
p5 = tf.ones([4,16,16,3], dtype=tf.uint8)
p5 = p5
feature_maps=[p2,p3,p4,p5]
boxes = tf.contrib.eager.Variable(tf.constant([480,360,480,360,480,360,160,160],shape=[1,2,4],dtype=tf.float32),name='boxes')
y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
h = y2
w = x2
#h = tf.contrib.eager.Variable(tf.constant([480,360,160,1,2,3],shape=[6,1,1],dtype=tf.float32),name='h')
#w = tf.contrib.eager.Variable(tf.constant([480,320,160,1,2,3], shape=[6, 1,1],dtype=tf.float32),name='w')
#print(y2)
#print(x2)
image_area = tf.cast(600 * 600, tf.float32)

roi_level = log2_graph(tf.sqrt(h*w) / (224.0 / tf.sqrt(image_area)))
print(roi_level)

roi_level = tf.minimum(2, tf.maximum(
            2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
print(roi_level)
roi_level = tf.squeeze(roi_level,2)
print(roi_level)
roi_level_pa = boxes = tf.contrib.eager.Variable(tf.constant([2,3,4,5],shape=[1,2,4],dtype=tf.float32),name='roi')

pooled = []
box_to_level = []
pool_shape = (2,2)
for i, level in enumerate(range(2, 6)):

    ix = tf.where(tf.equal(roi_level, level))

    #print(ix)
    level_boxes = tf.gather_nd(boxes, ix)
    #print(level_boxes)
    box_indices = tf.cast(ix[:, 0], tf.int32)
    #print(box_indices)
    box_to_level.append(ix)
    #print(box_to_level)
    #print(feature_maps[i])
    pooled.append(tf.image.crop_and_resize(
        feature_maps[i], level_boxes, box_indices, pool_shape,
        method="bilinear"))
    #print(pooled)
#for i in pooled:
#    print(i)
pooled = tf.concat(pooled, axis=0)
#print(pooled)
#print(box_to_level)
box_to_level = tf.concat(box_to_level, axis=0)
print(box_to_level)
box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
print(box_range)
box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                                 axis=1)
print(box_to_level[:, 0])
print(box_to_level[:, 1])
sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
print (sorting_tensor)

ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
            box_to_level)[0]).indices[::-1]
ix = tf.gather(box_to_level[:, 2], ix)
print(ix)
pooled = tf.gather(pooled, ix)
print(pooled)
# Re-add the batch dimension
shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)
print(shape)
pooled = tf.reshape(pooled, shape)
print(pooled)

