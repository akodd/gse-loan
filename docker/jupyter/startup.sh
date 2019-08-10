#!/bin/bash

#start SSH server
/usr/sbin/sshd

#start Jupyter Lab
jupyter lab --ip=0.0.0.0 --allow-root --no-browser

