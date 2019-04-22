
alter table fnm_acq add constraint fnm_acq_pk primary key (loan_id) enabled;
alter table fnm_prf add constraint fnm_prf_pk primary key (loan_id, rpt_period) enabled;

