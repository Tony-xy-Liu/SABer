#!/bin/bash

HERE=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

export PYTHONPATH=$HERE/src:$PATH
python $HERE/src/saber/__main__.py $@
