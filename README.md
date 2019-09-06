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

## convert annotation files to csv file

Firstly, use process_bounding_boxes.py to convert annotation files in data directory to bounding_boxes.csv
```
./process_bounding_boxes.py data/Annotation/ > bounding_boxes.csv
```
process_bounding_boxes.py is copied from [tensorflow/models](https://github.com/tensorflow/models/tree/master/research/inception/inception/data)
