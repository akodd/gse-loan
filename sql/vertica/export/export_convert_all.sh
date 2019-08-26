#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo "Export ACQ #############################################################"
${DIR}/export_acq.sh

echo "Export SEQ #############################################################"
${DIR}/export_seq.sh

echo "Covert Pandas to Numpy #################################################"
docker-compose exec jupyter /opt/conda/bin/python \
    /home/user/notebooks/sql/vertica/export/convert_to_numpy.py

echo "Complete"