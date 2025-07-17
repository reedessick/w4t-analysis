#!/bin/bash

# pick up PLASMAtools
export PYTHONPATH="$PWD/opt:$PYTHONPATH"

# pick up local installs (may live in different locations on different machines)
export PATH="$PWD/opt/bin:$PATH"
export PYTHONPATH="$PWD/opt/lib/python3.10/site-packages/:$PYTHONPATH"

export PATH="$PWD/opt/local/bin:$PATH"
export PYTHONPATH="$PWD/opt/local/lib/python3.10/dist-packages/:$PYTHONPATH"
