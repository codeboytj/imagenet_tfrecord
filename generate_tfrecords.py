import os.path
import csv
import contextlib2

import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.dataset_tools import tf_record_creation_util


flags = tf.app.flags
flags.DEFINE_string('box_csv_path', '', 'Path to box csv path')
flags.DEFINE_string('images_path', '', 'Path to box csv path')
flags.DEFINE_string('output_filebase', '', '')
flags.DEFINE_integer('num_shards', 10, '')

FLAGS = flags.FLAGS

class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()

    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_png(self._png_data, channels=3)
    self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that converts CMYK JPEG data to RGB JPEG data.
    self._cmyk_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_jpeg(self._cmyk_data, channels=0)
    self._cmyk_to_rgb = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def png_to_jpeg(self, image_data):
    return self._sess.run(self._png_to_jpeg,
                          feed_dict={self._png_data: image_data})

  def cmyk_to_rgb(self, image_data):
    return self._sess.run(self._cmyk_to_rgb,
                          feed_dict={self._cmyk_data: image_data})

  def decode_jpeg(self, image_data):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

def create_tf_example(images_path, image_basename, box):
  sysnet = image_basename.split('_')[0]
  fullname = os.path.join(images_path, sysnet, image_basename)
  if not os.path.exists(fullname):
      return None
  with tf.gfile.FastGFile(fullname, 'rb') as f:
      image_data = f.read()

  # Decode the RGB JPEG.
  coder = ImageCoder()
  image = coder.decode_jpeg(image_data)

  assert len(image.shape) == 3
  height = image.shape[0]
  width = image.shape[1]
  assert image.shape[2] == 3
  filename = bytes(image_basename,'utf8') # Filename of the image. Empty if image is not from file
  encoded_image_data = image_data # Encoded image bytes
  image_format = bytes(image_basename.split('.')[1],'utf8') # b'jpeg' or b'png'

  xmins = [float(box[0])] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [float(box[2])] # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = [float(box[1])] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [float(box[3])] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  # TODO(user): change this to what object you want detection.
  classes_text = [b'tennis ball'] # List of string class name of bounding box (1 per box)
  # TODO(user): imagenet always contains 1 box in 1 image, so set to 1
  classes = [1] # List of integer class id of bounding box (1 per box)

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example

def get_image_box_dict(box_csv_path):
  image_box_dict = {}
  with open(box_csv_path) as csvfile:
      boxreader = csv.reader(csvfile, delimiter=',')
      for row in boxreader:
          box = []
          image_box_dict[row[0]] = row[1:]
  return image_box_dict


def main(_):
  box_csv_path = FLAGS.box_csv_path
  num_shards=FLAGS.num_shards
  output_filebase=FLAGS.output_filebase
  images_path = FLAGS.images_path
  writer = tf.python_io.TFRecordWriter(output_filebase)

  image_box_dict = get_image_box_dict(box_csv_path)

 # with contextlib2.ExitStack() as tf_record_close_stack:
  #  output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
   #     tf_record_close_stack, output_filebase, num_shards)

  for image_basename in image_box_dict:
      box = image_box_dict[image_basename]
      tf_example = create_tf_example(images_path, image_basename, box)
      if not tf_example:
          continue;
  #    output_shard_index = index % num_shards
      writer.write(tf_example.SerializeToString())
  writer.close()


if __name__ == '__main__':
  tf.app.run()
