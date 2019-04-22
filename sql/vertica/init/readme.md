# Loading data to Vertica database

Start docker-compose

## Upload Acquisition files

1. Create ```SQL fnm_acq``` table by running the following command against vertica container.

```shell
$ docker-compose exec vertica /opt/vertica/bin/vsql -U dbadmin -f /home/dbadmin/gse-loan/sql/vertica/init/fnm_acq.sql
CREATE TABLE
```
Test that ```SQL fnm_acq``` is created by running:
```shell
$ docker-compose exec vertica /opt/vertica/bin/vsql -U dbadmin -c "\d"
                List of tables
 Schema |  Name   | Kind  |  Owner  | Comment
--------+---------+-------+---------+---------
 public | fnm_acq | table | dbadmin |
(1 row)
```

2. Load Acquisition data

```shell
$ docker-compose exec vertica /home/dbadmin/gse-loan/sql/vertica/init/fnm_acq_load.sh
```

3. Verify that data is uploaded

```shell
$ docker-compose exec vertica /opt/vertica/bin/vsql -U dbadmin -c "select count(*) from fnm_acq"
  count
----------
 37955500
(1 row)
```

and matched the number of rows in CSV files

```shell
$ wc -l data-csv/fnm/2018Q1/Acquisition_*.txt
    246863 data-csv/fnm/2018Q1/Acquisition_2000Q1.txt
    274339 data-csv/fnm/2018Q1/Acquisition_2000Q2.txt
    333498 data-csv/fnm/2018Q1/Acquisition_2000Q3.txt
    363892 data-csv/fnm/2018Q1/Acquisition_2000Q4.txt
    471277 data-csv/fnm/2018Q1/Acquisition_2001Q1.txt
    847131 data-csv/fnm/2018Q1/Acquisition_2001Q2.txt
    790096 data-csv/fnm/2018Q1/Acquisition_2001Q3.txt
    896645 data-csv/fnm/2018Q1/Acquisition_2001Q4.txt
    968761 data-csv/fnm/2018Q1/Acquisition_2002Q1.txt
    669497 data-csv/fnm/2018Q1/Acquisition_2002Q2.txt
    751086 data-csv/fnm/2018Q1/Acquisition_2002Q3.txt
   1248944 data-csv/fnm/2018Q1/Acquisition_2002Q4.txt
   1427865 data-csv/fnm/2018Q1/Acquisition_2003Q1.txt
   1652967 data-csv/fnm/2018Q1/Acquisition_2003Q2.txt
   1738620 data-csv/fnm/2018Q1/Acquisition_2003Q3.txt
    840391 data-csv/fnm/2018Q1/Acquisition_2003Q4.txt
    452474 data-csv/fnm/2018Q1/Acquisition_2004Q1.txt
    614487 data-csv/fnm/2018Q1/Acquisition_2004Q2.txt
    389490 data-csv/fnm/2018Q1/Acquisition_2004Q3.txt
    361663 data-csv/fnm/2018Q1/Acquisition_2004Q4.txt
    303621 data-csv/fnm/2018Q1/Acquisition_2005Q1.txt
    339377 data-csv/fnm/2018Q1/Acquisition_2005Q2.txt
    440528 data-csv/fnm/2018Q1/Acquisition_2005Q3.txt
    378318 data-csv/fnm/2018Q1/Acquisition_2005Q4.txt
    253053 data-csv/fnm/2018Q1/Acquisition_2006Q1.txt
    291177 data-csv/fnm/2018Q1/Acquisition_2006Q2.txt
    271380 data-csv/fnm/2018Q1/Acquisition_2006Q3.txt
    280988 data-csv/fnm/2018Q1/Acquisition_2006Q4.txt
    253292 data-csv/fnm/2018Q1/Acquisition_2007Q1.txt
    287275 data-csv/fnm/2018Q1/Acquisition_2007Q2.txt
    314723 data-csv/fnm/2018Q1/Acquisition_2007Q3.txt
    391209 data-csv/fnm/2018Q1/Acquisition_2007Q4.txt
    380845 data-csv/fnm/2018Q1/Acquisition_2008Q1.txt
    444387 data-csv/fnm/2018Q1/Acquisition_2008Q2.txt
    353277 data-csv/fnm/2018Q1/Acquisition_2008Q3.txt
    342129 data-csv/fnm/2018Q1/Acquisition_2008Q4.txt
    617658 data-csv/fnm/2018Q1/Acquisition_2009Q1.txt
    744709 data-csv/fnm/2018Q1/Acquisition_2009Q2.txt
    563161 data-csv/fnm/2018Q1/Acquisition_2009Q3.txt
    393048 data-csv/fnm/2018Q1/Acquisition_2009Q4.txt
    323174 data-csv/fnm/2018Q1/Acquisition_2010Q1.txt
    334204 data-csv/fnm/2018Q1/Acquisition_2010Q2.txt
    496357 data-csv/fnm/2018Q1/Acquisition_2010Q3.txt
    672768 data-csv/fnm/2018Q1/Acquisition_2010Q4.txt
    505196 data-csv/fnm/2018Q1/Acquisition_2011Q1.txt
    282035 data-csv/fnm/2018Q1/Acquisition_2011Q2.txt
    328628 data-csv/fnm/2018Q1/Acquisition_2011Q3.txt
    587766 data-csv/fnm/2018Q1/Acquisition_2011Q4.txt
    637560 data-csv/fnm/2018Q1/Acquisition_2012Q1.txt
    542027 data-csv/fnm/2018Q1/Acquisition_2012Q2.txt
    715316 data-csv/fnm/2018Q1/Acquisition_2012Q3.txt
    721613 data-csv/fnm/2018Q1/Acquisition_2012Q4.txt
    681364 data-csv/fnm/2018Q1/Acquisition_2013Q1.txt
    662920 data-csv/fnm/2018Q1/Acquisition_2013Q2.txt
    623648 data-csv/fnm/2018Q1/Acquisition_2013Q3.txt
    421734 data-csv/fnm/2018Q1/Acquisition_2013Q4.txt
    274821 data-csv/fnm/2018Q1/Acquisition_2014Q1.txt
    326157 data-csv/fnm/2018Q1/Acquisition_2014Q2.txt
    394356 data-csv/fnm/2018Q1/Acquisition_2014Q3.txt
    400030 data-csv/fnm/2018Q1/Acquisition_2014Q4.txt
    436347 data-csv/fnm/2018Q1/Acquisition_2015Q1.txt
    498253 data-csv/fnm/2018Q1/Acquisition_2015Q2.txt
    503774 data-csv/fnm/2018Q1/Acquisition_2015Q3.txt
    425663 data-csv/fnm/2018Q1/Acquisition_2015Q4.txt
    404588 data-csv/fnm/2018Q1/Acquisition_2016Q1.txt
    528482 data-csv/fnm/2018Q1/Acquisition_2016Q2.txt
    623483 data-csv/fnm/2018Q1/Acquisition_2016Q3.txt
    678805 data-csv/fnm/2018Q1/Acquisition_2016Q4.txt
    480198 data-csv/fnm/2018Q1/Acquisition_2017Q1.txt
    460092 data-csv/fnm/2018Q1/Acquisition_2017Q2.txt
  37955500 total
```

## Upload Performance files

1. Create ```SQL fnm_prf``` table by running the following command against vertica container.

```shell
$ docker-compose exec vertica /opt/vertica/bin/vsql -U dbadmin -f /home/dbadmin/gse-loan/sql/vertica/init/fnm_prf.sql
CREATE TABLE
```
Test that ```SQL fnm_prf``` is created by running:
```shell
$ docker-compose exec vertica /opt/vertica/bin/vsql -U dbadmin -c "\d"
                List of tables
 Schema |  Name   | Kind  |  Owner  | Comment
--------+---------+-------+---------+---------
 public | fnm_acq | table | dbadmin |
 public | fnm_prf | table | dbadmin |
(2 rows)
```
2. Load Performance data

```shell
$ docker-compose exec vertica /home/dbadmin/gse-loan/sql/vertica/init/fnm_prf_load.sh

$ docker-compose exec vertica /home/dbadmin/gse-loan/sql/vertica/init/fnm_prf_load.sh
2019-04-22 01:07:19: Starting Performance file upload
2019-04-22 01:07:19: Uploading file /opt/data/fnm/2018Q1/Performance_2000Q1.txt
 Rows Loaded
-------------
     9102336
(1 row)

2019-04-22 01:07:30: Uploading file /opt/data/fnm/2018Q1/Performance_2000Q2.txt
 Rows Loaded
-------------
     8215385
(1 row)

2019-04-22 01:07:40: Uploading file /opt/data/fnm/2018Q1/Performance_2000Q3.txt
 Rows Loaded
-------------
     8734051
(1 row)

2019-04-22 01:07:51: Uploading file /opt/data/fnm/2018Q1/Performance_2000Q4.txt
 Rows Loaded
-------------
    10171514
(1 row)

2019-04-22 01:08:03: Uploading file /opt/data/fnm/2018Q1/Performance_2001Q1.txt
 Rows Loaded
-------------
    15555760
(1 row)

2019-04-22 01:08:21: Uploading file /opt/data/fnm/2018Q1/Performance_2001Q2.txt
 Rows Loaded
-------------
    31570805
(1 row)

2019-04-22 01:08:55: Uploading file /opt/data/fnm/2018Q1/Performance_2001Q3.txt
 Rows Loaded
-------------
    27978295
(1 row)

2019-04-22 01:09:25: Uploading file /opt/data/fnm/2018Q1/Performance_2001Q4.txt
 Rows Loaded
-------------
    37313715
(1 row)

2019-04-22 01:10:06: Uploading file /opt/data/fnm/2018Q1/Performance_2002Q1.txt
 Rows Loaded
-------------
    41732928
(1 row)

2019-04-22 01:10:52: Uploading file /opt/data/fnm/2018Q1/Performance_2002Q2.txt
 Rows Loaded
-------------
    26326784
(1 row)

2019-04-22 01:11:21: Uploading file /opt/data/fnm/2018Q1/Performance_2002Q3.txt
 Rows Loaded
-------------
    32663105
(1 row)

2019-04-22 01:11:57: Uploading file /opt/data/fnm/2018Q1/Performance_2002Q4.txt
 Rows Loaded
-------------
    71006154
(1 row)

2019-04-22 01:13:13: Uploading file /opt/data/fnm/2018Q1/Performance_2003Q1.txt
 Rows Loaded
-------------
    92355851
(1 row)

2019-04-22 01:14:51: Uploading file /opt/data/fnm/2018Q1/Performance_2003Q2.txt
 Rows Loaded
-------------
   122915993
(1 row)

2019-04-22 01:17:01: Uploading file /opt/data/fnm/2018Q1/Performance_2003Q3.txt
 Rows Loaded
-------------
   144363886
(1 row)

2019-04-22 01:19:32: Uploading file /opt/data/fnm/2018Q1/Performance_2003Q4.txt
 Rows Loaded
-------------
    64864563
(1 row)

2019-04-22 01:20:40: Uploading file /opt/data/fnm/2018Q1/Performance_2004Q1.txt
 Rows Loaded
-------------
    34098203
(1 row)

2019-04-22 01:21:17: Uploading file /opt/data/fnm/2018Q1/Performance_2004Q2.txt
 Rows Loaded
-------------
    48559537
(1 row)

2019-04-22 01:22:08: Uploading file /opt/data/fnm/2018Q1/Performance_2004Q3.txt
 Rows Loaded
-------------
    27093854
(1 row)

2019-04-22 01:22:38: Uploading file /opt/data/fnm/2018Q1/Performance_2004Q4.txt
 Rows Loaded
-------------
    26924397
(1 row)

2019-04-22 01:23:08: Uploading file /opt/data/fnm/2018Q1/Performance_2005Q1.txt
 Rows Loaded
-------------
    23091083
(1 row)

2019-04-22 01:23:33: Uploading file /opt/data/fnm/2018Q1/Performance_2005Q2.txt
 Rows Loaded
-------------
    25302666
(1 row)

2019-04-22 01:24:00: Uploading file /opt/data/fnm/2018Q1/Performance_2005Q3.txt
 Rows Loaded
-------------
    33285833
(1 row)

2019-04-22 01:24:35: Uploading file /opt/data/fnm/2018Q1/Performance_2005Q4.txt
 Rows Loaded
-------------
    27584264
(1 row)

2019-04-22 01:25:05: Uploading file /opt/data/fnm/2018Q1/Performance_2006Q1.txt
 Rows Loaded
-------------
    16949636
(1 row)

2019-04-22 01:25:24: Uploading file /opt/data/fnm/2018Q1/Performance_2006Q2.txt
 Rows Loaded
-------------
    18393084
(1 row)

2019-04-22 01:25:44: Uploading file /opt/data/fnm/2018Q1/Performance_2006Q3.txt
 Rows Loaded
-------------
    15398402
(1 row)

2019-04-22 01:26:01: Uploading file /opt/data/fnm/2018Q1/Performance_2006Q4.txt
 Rows Loaded
-------------
    16657317
(1 row)

2019-04-22 01:26:19: Uploading file /opt/data/fnm/2018Q1/Performance_2007Q1.txt
 Rows Loaded
-------------
    15413114
(1 row)

2019-04-22 01:26:36: Uploading file /opt/data/fnm/2018Q1/Performance_2007Q2.txt
 Rows Loaded
-------------
    17070057
(1 row)

2019-04-22 01:26:55: Uploading file /opt/data/fnm/2018Q1/Performance_2007Q3.txt
 Rows Loaded
-------------
    16758877
(1 row)

2019-04-22 01:27:13: Uploading file /opt/data/fnm/2018Q1/Performance_2007Q4.txt
 Rows Loaded
-------------
    20618621
(1 row)

2019-04-22 01:27:36: Uploading file /opt/data/fnm/2018Q1/Performance_2008Q1.txt
 Rows Loaded
-------------
    20408986
(1 row)

2019-04-22 01:27:58: Uploading file /opt/data/fnm/2018Q1/Performance_2008Q2.txt
 Rows Loaded
-------------
    23005421
(1 row)

2019-04-22 01:28:23: Uploading file /opt/data/fnm/2018Q1/Performance_2008Q3.txt
 Rows Loaded
-------------
    15429480
(1 row)

2019-04-22 01:28:40: Uploading file /opt/data/fnm/2018Q1/Performance_2008Q4.txt
 Rows Loaded
-------------
    13483697
(1 row)

2019-04-22 01:28:56: Uploading file /opt/data/fnm/2018Q1/Performance_2009Q1.txt
 Rows Loaded
-------------
    30970650
(1 row)

2019-04-22 01:29:29: Uploading file /opt/data/fnm/2018Q1/Performance_2009Q2.txt
 Rows Loaded
-------------
    41515043
(1 row)

2019-04-22 01:30:15: Uploading file /opt/data/fnm/2018Q1/Performance_2009Q3.txt
 Rows Loaded
-------------
    32325739
(1 row)

2019-04-22 01:30:50: Uploading file /opt/data/fnm/2018Q1/Performance_2009Q4.txt
 Rows Loaded
-------------
    21233015
(1 row)

2019-04-22 01:31:14: Uploading file /opt/data/fnm/2018Q1/Performance_2010Q1.txt
 Rows Loaded
-------------
    17250154
(1 row)

2019-04-22 01:31:33: Uploading file /opt/data/fnm/2018Q1/Performance_2010Q2.txt
 Rows Loaded
-------------
    16825868
(1 row)

2019-04-22 01:31:53: Uploading file /opt/data/fnm/2018Q1/Performance_2010Q3.txt
 Rows Loaded
-------------
    25155034
(1 row)

2019-04-22 01:32:21: Uploading file /opt/data/fnm/2018Q1/Performance_2010Q4.txt
 Rows Loaded
-------------
    37393514
(1 row)

2019-04-22 01:33:02: Uploading file /opt/data/fnm/2018Q1/Performance_2011Q1.txt
 Rows Loaded
-------------
    27593433
(1 row)

2019-04-22 01:33:33: Uploading file /opt/data/fnm/2018Q1/Performance_2011Q2.txt
 Rows Loaded
-------------
    13724480
(1 row)

2019-04-22 01:33:49: Uploading file /opt/data/fnm/2018Q1/Performance_2011Q3.txt
 Rows Loaded
-------------
    16020803
(1 row)

2019-04-22 01:34:07: Uploading file /opt/data/fnm/2018Q1/Performance_2011Q4.txt
 Rows Loaded
-------------
    29910975
(1 row)

2019-04-22 01:34:40: Uploading file /opt/data/fnm/2018Q1/Performance_2012Q1.txt
 Rows Loaded
-------------
    34019747
(1 row)

2019-04-22 01:35:18: Uploading file /opt/data/fnm/2018Q1/Performance_2012Q2.txt
 Rows Loaded
-------------
    29375041
(1 row)

2019-04-22 01:35:50: Uploading file /opt/data/fnm/2018Q1/Performance_2012Q3.txt
 Rows Loaded
-------------
    39597625
(1 row)

2019-04-22 01:36:34: Uploading file /opt/data/fnm/2018Q1/Performance_2012Q4.txt
 Rows Loaded
-------------
    40312980
(1 row)

2019-04-22 01:37:18: Uploading file /opt/data/fnm/2018Q1/Performance_2013Q1.txt
 Rows Loaded
-------------
    36875047
(1 row)

2019-04-22 01:37:58: Uploading file /opt/data/fnm/2018Q1/Performance_2013Q2.txt
 Rows Loaded
-------------
    34044630
(1 row)

2019-04-22 01:38:37: Uploading file /opt/data/fnm/2018Q1/Performance_2013Q3.txt
 Rows Loaded
-------------
    28952275
(1 row)

2019-04-22 01:39:09: Uploading file /opt/data/fnm/2018Q1/Performance_2013Q4.txt
 Rows Loaded
-------------
    17261878
(1 row)

2019-04-22 01:39:29: Uploading file /opt/data/fnm/2018Q1/Performance_2014Q1.txt
 Rows Loaded
-------------
    10558111
(1 row)

2019-04-22 01:39:42: Uploading file /opt/data/fnm/2018Q1/Performance_2014Q2.txt
 Rows Loaded
-------------
    11879050
(1 row)

2019-04-22 01:39:56: Uploading file /opt/data/fnm/2018Q1/Performance_2014Q3.txt
 Rows Loaded
-------------
    13981208
(1 row)

2019-04-22 01:40:13: Uploading file /opt/data/fnm/2018Q1/Performance_2014Q4.txt
 Rows Loaded
-------------
    13592153
(1 row)

2019-04-22 01:40:29: Uploading file /opt/data/fnm/2018Q1/Performance_2015Q1.txt
 Rows Loaded
-------------
    14367950
(1 row)

2019-04-22 01:40:46: Uploading file /opt/data/fnm/2018Q1/Performance_2015Q2.txt
 Rows Loaded
-------------
    15708534
(1 row)

2019-04-22 01:41:04: Uploading file /opt/data/fnm/2018Q1/Performance_2015Q3.txt
 Rows Loaded
-------------
    14415902
(1 row)

2019-04-22 01:41:22: Uploading file /opt/data/fnm/2018Q1/Performance_2015Q4.txt
 Rows Loaded
-------------
    11193923
(1 row)

2019-04-22 01:41:35: Uploading file /opt/data/fnm/2018Q1/Performance_2016Q1.txt
 Rows Loaded
-------------
     9614531
(1 row)

2019-04-22 01:41:48: Uploading file /opt/data/fnm/2018Q1/Performance_2016Q2.txt
 Rows Loaded
-------------
    11513200
(1 row)

2019-04-22 01:42:03: Uploading file /opt/data/fnm/2018Q1/Performance_2016Q3.txt
 Rows Loaded
-------------
    12029782
(1 row)

2019-04-22 01:42:19: Uploading file /opt/data/fnm/2018Q1/Performance_2016Q4.txt
 Rows Loaded
-------------
    11309957
(1 row)

2019-04-22 01:42:35: Uploading file /opt/data/fnm/2018Q1/Performance_2017Q1.txt
 Rows Loaded
-------------
     6594650
(1 row)

2019-04-22 01:42:44: Uploading file /opt/data/fnm/2018Q1/Performance_2017Q2.txt
 Rows Loaded
-------------
     4932157
(1 row)

2019-04-22 01:42:52: Finished in 2133 seconds.
```

3. Verify that data is uploaded

```shell
$ docker-compose exec vertica /opt/vertica/bin/vsql -U dbadmin -c "select count(*) from fnm_prf"
   count
------------
 1932446693
(1 row)
```

and matched the number of rows in CSV files

```shell
$ wc -l data-csv/fnm/2018Q1/Performance_*.txt
     9102336 data-csv/fnm/2018Q1/Performance_2000Q1.txt
     8215385 data-csv/fnm/2018Q1/Performance_2000Q2.txt
     8734051 data-csv/fnm/2018Q1/Performance_2000Q3.txt
    10171514 data-csv/fnm/2018Q1/Performance_2000Q4.txt
    15555760 data-csv/fnm/2018Q1/Performance_2001Q1.txt
    31570805 data-csv/fnm/2018Q1/Performance_2001Q2.txt
    27978295 data-csv/fnm/2018Q1/Performance_2001Q3.txt
    37313715 data-csv/fnm/2018Q1/Performance_2001Q4.txt
    41732928 data-csv/fnm/2018Q1/Performance_2002Q1.txt
    26326784 data-csv/fnm/2018Q1/Performance_2002Q2.txt
    32663105 data-csv/fnm/2018Q1/Performance_2002Q3.txt
    71006154 data-csv/fnm/2018Q1/Performance_2002Q4.txt
    92355851 data-csv/fnm/2018Q1/Performance_2003Q1.txt
   122915993 data-csv/fnm/2018Q1/Performance_2003Q2.txt
   144363886 data-csv/fnm/2018Q1/Performance_2003Q3.txt
    64864563 data-csv/fnm/2018Q1/Performance_2003Q4.txt
    34098203 data-csv/fnm/2018Q1/Performance_2004Q1.txt
    48559537 data-csv/fnm/2018Q1/Performance_2004Q2.txt
    27093854 data-csv/fnm/2018Q1/Performance_2004Q3.txt
    26924397 data-csv/fnm/2018Q1/Performance_2004Q4.txt
    23091083 data-csv/fnm/2018Q1/Performance_2005Q1.txt
    25302666 data-csv/fnm/2018Q1/Performance_2005Q2.txt
    33285833 data-csv/fnm/2018Q1/Performance_2005Q3.txt
    27584264 data-csv/fnm/2018Q1/Performance_2005Q4.txt
    16949636 data-csv/fnm/2018Q1/Performance_2006Q1.txt
    18393084 data-csv/fnm/2018Q1/Performance_2006Q2.txt
    15398402 data-csv/fnm/2018Q1/Performance_2006Q3.txt
    16657317 data-csv/fnm/2018Q1/Performance_2006Q4.txt
    15413114 data-csv/fnm/2018Q1/Performance_2007Q1.txt
    17070057 data-csv/fnm/2018Q1/Performance_2007Q2.txt
    16758877 data-csv/fnm/2018Q1/Performance_2007Q3.txt
    20618621 data-csv/fnm/2018Q1/Performance_2007Q4.txt
    20408986 data-csv/fnm/2018Q1/Performance_2008Q1.txt
    23005421 data-csv/fnm/2018Q1/Performance_2008Q2.txt
    15429480 data-csv/fnm/2018Q1/Performance_2008Q3.txt
    13483697 data-csv/fnm/2018Q1/Performance_2008Q4.txt
    30970650 data-csv/fnm/2018Q1/Performance_2009Q1.txt
    41515043 data-csv/fnm/2018Q1/Performance_2009Q2.txt
    32325739 data-csv/fnm/2018Q1/Performance_2009Q3.txt
    21233015 data-csv/fnm/2018Q1/Performance_2009Q4.txt
    17250154 data-csv/fnm/2018Q1/Performance_2010Q1.txt
    16825868 data-csv/fnm/2018Q1/Performance_2010Q2.txt
    25155034 data-csv/fnm/2018Q1/Performance_2010Q3.txt
    37393514 data-csv/fnm/2018Q1/Performance_2010Q4.txt
    27593433 data-csv/fnm/2018Q1/Performance_2011Q1.txt
    13724480 data-csv/fnm/2018Q1/Performance_2011Q2.txt
    16020803 data-csv/fnm/2018Q1/Performance_2011Q3.txt
    29910975 data-csv/fnm/2018Q1/Performance_2011Q4.txt
    34019747 data-csv/fnm/2018Q1/Performance_2012Q1.txt
    29375041 data-csv/fnm/2018Q1/Performance_2012Q2.txt
    39597625 data-csv/fnm/2018Q1/Performance_2012Q3.txt
    40312980 data-csv/fnm/2018Q1/Performance_2012Q4.txt
    36875047 data-csv/fnm/2018Q1/Performance_2013Q1.txt
    34044630 data-csv/fnm/2018Q1/Performance_2013Q2.txt
    28952275 data-csv/fnm/2018Q1/Performance_2013Q3.txt
    17261878 data-csv/fnm/2018Q1/Performance_2013Q4.txt
    10558111 data-csv/fnm/2018Q1/Performance_2014Q1.txt
    11879050 data-csv/fnm/2018Q1/Performance_2014Q2.txt
    13981208 data-csv/fnm/2018Q1/Performance_2014Q3.txt
    13592153 data-csv/fnm/2018Q1/Performance_2014Q4.txt
    14367950 data-csv/fnm/2018Q1/Performance_2015Q1.txt
    15708534 data-csv/fnm/2018Q1/Performance_2015Q2.txt
    14415902 data-csv/fnm/2018Q1/Performance_2015Q3.txt
    11193923 data-csv/fnm/2018Q1/Performance_2015Q4.txt
     9614531 data-csv/fnm/2018Q1/Performance_2016Q1.txt
    11513200 data-csv/fnm/2018Q1/Performance_2016Q2.txt
    12029782 data-csv/fnm/2018Q1/Performance_2016Q3.txt
    11309957 data-csv/fnm/2018Q1/Performance_2016Q4.txt
     6594650 data-csv/fnm/2018Q1/Performance_2017Q1.txt
     4932157 data-csv/fnm/2018Q1/Performance_2017Q2.txt
  1932446693 total
```

## Enable primary keys
```shell
$ docker-compose exec vertica /opt/vertica/bin/vsql -U dbadmin -f /home/dbadmin/gse-loan/sql/vertica/init/fnm_post.sql
ALTER TABLE
ALTER TABLE
```

