alter table fnm_input_acq
add constraint fnm_input_acq_pk
primary key (loan_id) enabled
;

alter table fnm_input_seq add 
constraint fnm_input_seq_pk 
primary key(loan_id, rpt_period) enabled
;

