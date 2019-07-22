REM python extractFeature.py -mhi -f Hu -exp deney13b
python hmmTrain.py -s 13 -exp deney13b -tp person02 person03 person05 person06 person07 person08 person09 person10 person22
python hmmTest.py -exp deney13b -tp person02 person03 person05 person06 person07 person08 person09 person10 person22