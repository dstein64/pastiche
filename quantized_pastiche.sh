#!/usr/bin/env bash

DIR=quantized_pastiche
mkdir -p $DIR

for Q in {0..8}; do

    python -m pastiche \
        --num-steps 2000 \
        --quantization $Q \
        example/boston.jpg \
        example/vangogh_starry_night.jpg \
        $DIR/0.png

    python -m pastiche \
        --size 1024 \
        --quantization $Q \
        --num-steps 1000 \
        --init $DIR/0.png \
        example/boston.jpg \
        example/vangogh_starry_night.jpg \
        $DIR/1.png

    python -m pastiche \
        --size 2048 \
        --quantization $Q \
        --num-steps 500 \
        --init $DIR/1.png \
        example/boston.jpg \
        example/vangogh_starry_night.jpg \
        $DIR/2.png

    convert $DIR/2.png $DIR/q$Q.jpg
    rm $DIR/0.png $DIR/1.png $DIR/2.png

done
