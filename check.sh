#!/bin/bash

sstat --format=JobID,AveCPUFreq,AveDiskWrite,TresUsageInMax%80,TresUsageInMaxNode%80 -j $1 --allsteps
scontrol show jobid -dd $1