python extractFeature.py -f Hu -exp deney10
python hmmTrain.py -s 7 -l2r -exp deney10 -tp person02 person03 person05 person06 person07 person08 person09 person10 person22
python hmmTest.py -exp deney10 -tp person02 person03 person05 person06 person07 person08 person09 person10 person22