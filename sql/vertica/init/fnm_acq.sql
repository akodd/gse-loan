create table fnm_acq (
   loan_id              char(20) not null,         -- LOAN ID
   channel              char(1),                   -- CHANNEL
   seller_name          varchar(80),               -- SELLER NAME
   oir                  numeric(14,10),            -- ORIGINAL INTEREST RATE
   upb                  numeric(11,2),             -- ORIGINAL UNPAID PRINCIPAL BALANCE (UPB)
   term                 numeric(3,0),              -- ORIGINAL LOAN TERM
   odate                date,                      -- ORIGINATION DATE
   first_pmt_date       date,                      -- FIRST PAYMENT DATE 
   oltv                 numeric(14,10),            -- ORIGINAL LOAN-TO-VALUE (LTV)
   ocltv                numeric(14,10),            -- ORIGINAL COMBINED LOAN-TO-VALUE (CLTV) 
   borrow_num           numeric(3,0),              -- NUMBER OF BORROWERS 
   dti                  numeric(14,10),            -- DEBT-TO-INCOME RATIO (DTI)
   score                numeric(3,0),              -- BORROWER CREDIT SCORE
   first_time_flag      char(1),                   -- FIRST-TIME HOME BUYER INDICATOR
   purpose              char(1),                   -- LOAN PURPOSE
   property_type        char(2),                   -- PROPERTY TYPE 
   unit_num             char(10),                  -- NUMBER OF UNITS
   occupancy_status     char(1),                   -- OCCUPANCY STATUS 
   state                varchar(20),               -- PROPERTY STATE
   zip3                 varchar(10),               -- ZIP (3-DIGIT) 
   mi_pct               numeric(14,10),            -- MORTGAGE INSURANCE PERCENTAGE
   product_type         varchar(20),               -- PRODUCT TYPE
   co_score             numeric(3,0),              -- CO-BORROWER CREDIT SCORE
   mi_type              varchar(1),                -- MORTGAGE INSURANCE TYPE
   recolation_mort_ind  char(1)                    -- RELOCATION MORTGAGE INDICATOR
)