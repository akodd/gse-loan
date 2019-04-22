create table fnm_prf (
   loan_id                    char(20) not null,      -- LOAN IDENTIFIER
   rpt_period                 date not null,          -- MONTHLY REPORTING PERIOD
   servicer_name              varchar(80),            -- SERVICER NAME
   int_rate                   numeric(14,10),         -- CURRENT INTEREST RATE
   current_upb                numeric(11,2),          -- CURRENT ACTUAL UNPAID PRINCIPAL BALANCE (UPB)
   age                        numeric(10,0),          -- LOAN AGE 
   months_to_maturity         numeric(3,0),           -- REMAINING MONTHS TO LEGAL MATURITY
   months_to_maturity_adj     numeric(3,0),           -- ADJUSTED REMAINING MONTHS TO MATURITY
   maturity_date              date,                   -- MATURITY DATE
   msa                        char(5),                -- METROPOLITAN STATISTICAL AREA (MSA)
   del                        char(5),                -- CURRENT LOAN DELINQUENCY STATUS
   modification_flag          char(1),                -- MODIFICATION FLAG
   zero_bal_code              char(2),                -- ZERO BALANCE CODE 
   zero_bal_date              date,                   -- ZERO BALANCE EFFECTIVE DATE
   lpi_date                   date,                   -- LAST PAID INSTALLMENT DATE
   foreclosure_date           date,                   -- FORECLOSURE DATE
   disposition_date           date,                   -- DISPOSITION DATE
   foreclosure_cost           numeric(18,12),         -- FORECLOSURE COSTS
   ppp_cost                   numeric(18,12),         -- PROPERTY PRESERVATION AND REPAIR COSTS
   recovery_cost              numeric(18,12),         -- ASSET RECOVERY COSTS
   holding_cost               numeric(18,12),         -- MISCELLANEOUS HOLDING EXPENSES AND CREDITS
   tax_cost                   numeric(18,12),         -- ASSOCIATED TAXES FOR HOLDING PROPERTY
   net_sales                  numeric(18,12),         -- NET SALE PROCEEDS
   credit_enh_procs           numeric(18,12),         -- CREDIT ENHANCEMENT PROCEEDS
   repurchase_procs           numeric(18,12),         -- REPURCHASE MAKE WHOLE PROCEEDS
   other_procs                numeric(18,12),         -- OTHER FORECLOSURE PROCEEDS
   non_int_upb                numeric(11,2),          -- NON INTEREST BEARING UPB
   prin_forg_upb              numeric(11,2),          -- PRINCIPAL FORGIVENESS UPB 
   make_whole_ind             char(1),                -- REPURCHASE MAKE WHOLE PROCEEDS FLAG
   forclosure_prin_write_off  numeric(11,2),          -- FORECLOSURE PRINCIPAL WRITE-OFF AMOUNT
   servicing_activity_ind     char(1)                 -- SERVICING ACTIVITY INDICATOR 
)