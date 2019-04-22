#!/bin/bash

temp_location=/tmp/gseload
data_files=/opt/data/fnm/2018Q1
control_file_dir=/home/dbadmin/gse-loan/sql/vertica/init

echo "Making $temp_location"
mkdir -p $temp_location

ACQ_CTL=fnm_acq_ctl.sql

mkdir -p $temp_location 

echo "Loading Acquistion files from $data_files"

for f in `ls -1 $data_files/Acquisition_*.txt`
do
    echo "Uploading file $f"
	cp -rf $control_file_dir/$ACQ_CTL $temp_location
	sed -i -e "s|ACQ_CURRENT|$f|g" $temp_location/$ACQ_CTL
	/opt/vertica/bin/vsql -U dbadmin -f $temp_location/$ACQ_CTL

	rm -rf $temp_location/*.txt
done

echo "Finished"
