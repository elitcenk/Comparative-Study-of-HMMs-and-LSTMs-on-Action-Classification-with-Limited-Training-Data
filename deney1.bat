python extractFeature.py -f Hu -exp deney1
python hmmTrain.py -s 3 -l2r -exp deney1 -tp person02 person03 person05 person06 person07 person08 person09 person10 person22
python hmmTest.py -exp deney1 -tp person02 person03 person05 person06 person07 person08 person09 person10 person22