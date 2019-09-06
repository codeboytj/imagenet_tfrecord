# imagenet_tfrecord

Convert imagenet images to tfrecord file

## Recommended Directory Structure for Training and Evaluation

|--data/
     |--Annotation/
          |--n04409515/
               |--n04409515_7148.xml
               |--n04409515_6823.xml
               |--n04409515_6839.xml
               ....................
               |--n04409515_6862.xml
               |--n04409515_6900.xml
     |--train/
          |--n04409515/
               |--n04409515_7148.JPEG
               |--n04409515_6862.JPEG
               ....................
               |--n04409515_6900.JPEG
     |--val/
          |--n04409515/
               |--n04409515_6823.JPEG
               |--n04409515_6839.JPEG
               ....................


## convert annotation files to csv file

use process_bounding_boxes.py to convert annotation files in data directory to bounding_boxes.csv
```
./process_bounding_boxes.py data/Annotation/ > bounding_boxes.csv
```
process_bounding_boxes.py is copied from [tensorflow/models](https://github.com/tensorflow/models/tree/master/research/inception/inception/data)

## generate tfrecord according to images and annotation

Firstly, clone tensorflow models: `git clone https://github.com/tensorflow/models.git`.

Then, [install tensorflow object_detecion](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md).

Then, change classes_text variable in line 82 in generate_tfrecord.py to what object you want detection. For example, I want to train a tennis detection model:
```
  classes_text = [b'tennis ball']
```
After finish that, Copy generate_tfrecord.py to master/research/object_detection/

Finally, run follow command to generate train.record and val.record file.
```
python models/research/object_detection/generate_tfrecords.py  --box_csv_path=bounding_boxes.csv --images_path=data/train/ --output_filebase=data/tfrecords/train.record
python models/research/object_detection/generate_tfrecords.py  --box_csv_path=bounding_boxes.csv --images_path=data/val/ --output_filebase=data/tfrecords/val.record
```
Then you will see train.record and val.record are in data/tfrecords directory
