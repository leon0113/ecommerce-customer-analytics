# Insights and recommendations

All figures come from the notebooks in `notebooks/` and the workbook `reports/summary.xlsx`.
The data is **synthetic**: it was generated to behave like a Bangladeshi online retailer over
2023-2025 (see the README). The patterns below are real properties of this dataset, but they
are not claims about any real business. Correlations in observational data are not proof of
cause; where a recommendation depends on cause, it says how to test it.

**Definitions used throughout**

- **Net revenue:** quantity x unit price x (1 - order discount), delivered orders only, in BDT.
  Refunds are reported separately.
- **Churned:** no order for more than 140 days at 1 January 2026. 140 days is the
  90th percentile of the gap between a customer's consecutive orders.

## The business in one paragraph

Over three years the business made **17.9M BDT** of net revenue from **3,903 delivered orders**
and **1,208 buyers**, at an average order value of **4,592 BDT**. Revenue grew **26.8%** in 2024
and **28.4%** in 2025. All of that growth came from more buyers (438 → 614); order value stayed
flat. Meanwhile, customers do not stay: **45%** of buyers never place a second order, and only
**32%** of new buyers order again in the following quarter. The business is growing by
continually replacing the customers it loses.

## Key findings

### 1. Revenue is concentrated in a small group of customers

- The top 20% of buyers produce **66%** of net revenue.
- The RFM segment "Champions" is **15%** of buyers but **50%** of revenue. Together with
  "Loyal", it is **26%** of buyers and **64%** of revenue.
- Champions buy on discount the least: **20%** of their orders, against about 40% for every
  other segment.

### 2. The biggest leak is the second order

- Of the 1,039 buyers old enough to judge, **468 (45%)** bought once and never returned.
- Repeat buyers churn at **65%**, compared with 100% for one-order buyers by definition.
- The median gap between consecutive orders is **38 days**.

### 3. Churn goes with discount dependence, heavy returns and some acquisition channels

These comparisons use repeat buyers only, so one-order customers don't distort them.

| Driver | Lowest-churn group | Highest-churn group |
|---|---|---|
| Share of orders bought on discount | 0-25%: **54%** churned (n=281) | 50-100%: **86-87%** churned (n=124) |
| Share of order lines returned | up to 10%: **41%** (n=111) | over 20%: **76%** (n=68) |
| Acquisition channel | referral: **48%** (n=60) | affiliate: **79%** (n=63) |

Division and gender show no consistent difference.

The churn definition was checked against the generator's hidden record of when each customer
actually stopped buying. It agrees for **92%** of customers.

### 4. Acquisition channels bring customers of different quality

- In their first year, referral and paid-search customers come back in about **28%** of
  quarters, affiliate customers in **18%**.
- By the eighth quarter after their first order, **3%** of affiliate customers are still
  buying, against **10-12%** for referral and organic.
- Retention has not improved over time: next-quarter retention is **31-33%** for the 2023,
  2024 and 2025 cohorts alike.

### 5. Fashion and Footwear drive returns

- They account for **23%** of revenue but **51%** of all refunds.
- Their return rates are **18.6%** and **16.3%** of lines, against 1-6% for every other
  category.
- Electronics is the largest category at **25%** of revenue, with a 6.1% return rate.

### 6. Demand is predictable in time and place

- **Season:** November and December bring **23.7%** of annual revenue, against 16.7% if spread
  evenly. The Eid ul-Fitr window is the other recurring peak (April 2023-24, late March 2025).
- **Week:** Friday and Saturday, the Bangladeshi weekend, average **25%** more orders per day
  than other days.
- **Day:** **46%** of orders arrive between 18:00 and midnight.
- **Place:** Dhaka division brings **43%** of revenue, and Chattogram **18%**.
- **Payment:** cash on delivery is still **45%** of orders.

### 7. A few bad rows can distort totals badly

- Raw revenue overstated clean revenue by **19%** (21.3M vs 17.9M BDT). Almost all of the gap
  came from **16 order lines** whose prices were entered at 100x.
- After cleaning, **99.4%** of orders reconcile to the taka with the independent payment
  records.

## Recommendations

Ordered by expected impact relative to effort.

1. **Build a second-order programme.**
   - *Why:* 45% of buyers never return. Customers who do place a second order churn at 65%,
     and they include every high-value segment.
   - *Action:* contact new buyers within the 38-day median repeat gap, with a recommendation
     based on their first category rather than a blanket discount.
   - *First audience:* the **195 "New"** customers.
   - *Measure:* share of new buyers placing a second order within 90 days, against a
     holdout group.

2. **Work the at-risk list now.**
   - *Who:* **91** active customers have gone quiet for longer than a typical repeat gap.
     They hold **3.0M BDT (16.9%)** of past revenue, and most are Champions.
   - *Where:* the list is in `summary.xlsx` (sheet "At-risk customers"), ranked by value.
   - *Action:* personal outreach, such as a call or a priority-delivery offer, for the top 50.

3. **Re-weight acquisition spend towards channels that retain.**
   - *Why:* affiliate customers churn at 79% and almost none are buying two years on.
     Judging channels only on cost per first order overvalues them.
   - *Action:* compare channels on revenue per customer over 12 months instead.
   - *Test first:* referral looks best but rests on 111 customers, so grow it through a
     formal referral scheme and measure before committing large budgets.

4. **Test, don't assume, the discount effect.**
   - *What we know:* heavily discounted buyers churn far more. But discounts may attract
     customers who would have left anyway, rather than causing them to leave.
   - *Action:* hold back routine discounts from a random group of repeat buyers for a
     quarter and compare retention and margin.
   - *Meanwhile:* keep blanket discounts away from Champions. They buy at full price.

5. **Cut Fashion and Footwear returns.**
   - *Why:* half of all refunds come from under a quarter of revenue.
   - *Action:* size guides, fit notes and customer photos on product pages. Track the
     return rate per product to find the worst offenders.
   - *Watch:* buyers returning over 20% of lines also churn more, so this protects retention
     as well as margin.

6. **Plan around the calendar.**
   - *Stock and delivery capacity:* plan for November-December and the run-up to Eid.
   - *Campaigns:* time them for Friday-Saturday and for 18:00-23:00, when demand already
     peaks.

7. **Add data-entry checks.**
   - *Why:* a price-entry check (for example, flag any unit price more than 10x the list
     price) would have prevented a 19% overstatement of revenue.
   - *How:* the same rules used in `src/clean.py` can run at entry time.

## Limitations

- **The data is synthetic.** Patterns were built into the generator (see `config.yaml`) and
  then rediscovered. The analysis shows the method works; it doesn't measure a real market.
- **The sample is small.** Every table is under 10,000 rows. Segment and cohort sizes of
  40-140 customers mean differences of a few percentage points are noise. Results also vary
  by random seed: across 8 seeds, year-over-year growth varied by about ±6 points.
- **Churn is inferred, not observed.** The 140-day threshold is a reasonable rule, not a
  fact. Customers in the 169-person "too new" group are left out of churn rates.
- **Driver results are correlations.** No causal claim is made without an experiment.
- **Revenue measures sales, not profit.** The dataset has no product costs, so margin
  can't be analysed.
