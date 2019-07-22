REM python extractFeature.py -mhi -f Hu -exp deney9a
python hmmTrain.py -s 9 -exp deney9a -tp person02 person03 person05 person06 person07 person08 person09 person10 person22
python hmmTest.py -exp deney9a -tp person02 person03 person05 person06 person07 person08 person09 person10 person22d