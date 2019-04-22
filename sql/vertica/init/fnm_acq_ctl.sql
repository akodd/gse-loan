copy fnm_acq (
   loan_id,              
   channel,             
   seller_name,          
   oir,
   upb,
   term,
   odate_str filler varchar(7), 
   odate as to_date('01/' || odate_str, 'DD/MM/YYYY'),
   first_pmt_date_str filler varchar(7),
   first_pmt_date as to_date('01/' || first_pmt_date_str, 'DD/MM/YYYY'),
   oltv,
   ocltv,
   borrow_num,
   dti,
   score,
   first_time_flag,
   purpose,
   property_type,
   unit_num,
   occupancy_status,
   state,
   zip3,
   mi_pct,
   product_type,
   co_score,
   mi_type,
   recolation_mort_ind
) 
from 'ACQ_CURRENT' delimiter '|' null '' enforcelength trailing nullcols abort on error direct;

