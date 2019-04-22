#!/bin/bash

temp_location=/tmp/gseload
data_files=/opt/data/fnm/2018Q1
control_file_dir=/home/dbadmin/gse-loan/sql/vertica/init

mkdir -p $temp_location

CTL_FILE=fnm_prf_ctl.sql

SECONDS=0
TIMESTAMP=`date +'%Y-%m-%d %H:%M:%S'`
echo "$TIMESTAMP: Starting Performance file upload"

for f in `ls -1 $data_files/Performance_*.txt`
do
	TIMESTAMP=`date +'%Y-%m-%d %H:%M:%S'`
    echo "$TIMESTAMP: Uploading file $f"
	cp $control_file_dir/$CTL_FILE $temp_location
	sed -i -e "s|PERF_CURRENT|$f|g" $temp_location/$CTL_FILE
	/opt/vertica/bin/vsql -U dbadmin -f $temp_location/$CTL_FILE

	rm -rf $temp_location/*
done

duration=$SECONDS
TIMESTAMP=`date +'%Y-%m-%d %H:%M:%S'`
echo "$TIMESTAMP: Finished in $duration seconds."


