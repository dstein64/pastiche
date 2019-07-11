#!/usr/bin/env bash

DIR=quantized_pastiche
mkdir -p $DIR

for Q in {0..8}; do

    python -m pastiche \
        --num-steps 2000 \
        --quantization $Q \
        example/elephant.jpg \
        example/the_scream.png \
        $DIR/0.png

    python -m pastiche \
        --size-pixels 1000000 \
        --quantization $Q \
        --num-steps 1000 \
        --init $DIR/0.png \
        example/elephant.jpg \
        example/the_scream.png \
        $DIR/1.png

    python -m pastiche \
        --size-pixels 3300000 \
        --quantization $Q \
        --num-steps 500 \
        --init $DIR/1.png \
        example/elephant.jpg \
        example/the_scream.png \
        $DIR/2.png

    convert $DIR/2.png $DIR/q$Q.jpg
    rm $DIR/0.png $DIR/1.png $DIR/2.png

done
