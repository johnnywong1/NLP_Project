#!/bin/bash
cd /u/rm5tx/nlp_project
screen -d -m jupyter notebook --no-browser --port=8092 --ip=*
