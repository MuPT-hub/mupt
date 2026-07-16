#!/bin/bash

iter=$1
next=$((iter+1))

cp equil/nvt-equil-iter${iter}.gro gro-files/iter${next}.gro
cp topology/iter${iter}.top topology/iter${next}.top
