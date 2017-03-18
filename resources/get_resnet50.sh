#!/bin/sh

echo "Get ResNet50..."

echo "-- deploy file"
wget https://iuxblw-bn1305.files.1drv.com/y3mZBWe7g-MEdxG-Y1DGxPvvAoh1IOVlpH6YeYej_qtAw48oing5DADGAbDQA2Yrri9QRaH973F6WlJ_-Pl6tAmI5mOoJFiLY7G7qfajUlaYXk2AAGzmwLZnKGOtC7jqzv81oHyUpNp4-yI0kLlbWBM7Hwq6auJsXMUsmgGZYbr71I/ResNet-50-deploy.prototxt

echo "-- mean binaryproto"
wget https://iuxblw-bn1305.files.1drv.com/y3mX31Zw54rki9I6Pn_On2yNw6g6_rJlsPzKj8WYretlCjK7PaeBHXIFr00_a2_D4HHgmtp0DGxa-v4xGJZeRlqgU9eywoJt6D6O5i_UsOW4vlrbCwlN3Dd1GMQXBpZ9ZeGDbDKPnRo-d3Fp7PquuNWjFSkNe_c60yjmVLit-XUToE/ResNet_mean.binaryproto

echo "-- model weights"
wget https://iuxblw-bn1305.files.1drv.com/y3mM29EbVJVpufOka4N0BPXgAANk3f1rS3lADKZAGuECSn_6Y7qpCzrlPtJHKGWHleavstr_rb9qDHuAAzGKnkeloABSCgELh6PAcBOv292IRcMkCMFboGOCg42Ma2FU-wRfh31cJATJyndfCWeQ1WrsH5Qyw2QKWUCoATUCLJ7lzs/ResNet-50-model.caffemodel
