#!/bin/sh

echo "------------------------------------------"
echo "---- START CERVIX CLASSIFICATION TASK ----"
echo "------------------------------------------"

echo ""
echo "Step 1. Cervix/Os detection"
echo ""

python2 1_cervix_os_detection.py

sleep 10

echo ""
echo "Step 2a. Cervix classification with custom SqueezeNet"
echo ""

python3 2a_cervix_os_classification.py

sleep 10

echo ""
echo "Step 2b. Cervix classification with custom CNN"
echo ""

python3 2b_cervix_os_classification.py

echo ""
echo "Step 3. Merge results"
echo ""

python2 3_merge_classifications.py

echo ""
echo "Yeah! Everything is done"
echo ""



