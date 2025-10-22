# Price Discrepancy Investigation Runbook

## Issue Description
Customer reports scanning at wrong price - price shown differs from expected/configured price.

## Symptoms
- Ticket mentions "wrong price", "scanning", "discrepancy"
- Selling price vs actual price mismatch
- Often includes TPNB, GTIN, or location cluster information

## Diagnosis Steps

### 1. Extract Key Entities
From ticket description, identify:
- **TPNB** (Tesco Product Number Base) - 8-digit product code
- **GTIN** (Global Trade Item Number) - barcode
- **Location Cluster** - store group (e.g., "Large Stores England", cluster UUID)
- **Date/Time** - when price was scanned
- **Expected vs Actual** - price difference

### 2. Query Price Systems
Use these tools in order:

**a) Get Base Price:**
```
Tool: base_prices_get
Parameters:
  - tpnb: <from ticket>
  - locationClusterId: <from glossary mapping>
  - effectiveDateTime: <from ticket or current>
```

**b) Get Minimum Price:**
```
Tool: price_minimum_get
Parameters:
  - gtin: <from ticket>
  - locationClusterId: <cluster UUID>
```

**c) Check Active Promotions:**
```
Tool: competitor_promotional_prices_get
Parameters:
  - tpnbs: [<tpnb>]
  - locationClusterIds: [<cluster UUID>]
  - mechanic: "MULTIBUY" or "PRICE_CUT"
```

### 3. Check Audit Trail
Query Splunk for price change history:
```
index=price-lifecycle "price move" tpnb=<tpnb> cluster=<cluster>
| stats values(new_price) by user, timestamp
```

### 4. Verify Policies
```
Tool: policies_view
Parameters:
  - clusters: [<cluster UUID>]
  - classifications: [<product classification>]
```

## Tools to Use
- **base_prices_get**: Get current base price for TPNB
- **price_minimum_get**: Get minimum allowable price
- **competitor_promotional_prices_get**: Check for active promotions
- **policies_view**: View pricing policies
- **splunk_search**: Audit trail for price changes

## Resolution Steps

### If Price is Correct:
1. Explain pricing logic (base + promo + policy)
2. Provide breakdown showing calculation
3. Mark as "not a bug" with explanation

### If Price is Incorrect:
1. Identify which system has wrong data (base/promo/min)
2. Find user who loaded incorrect price (via Splunk audit)
3. Create corrective price move
4. Monitor for propagation (15-30 mins)

### If Data Missing:
1. Check if product is active in cluster
2. Verify location cluster mapping
3. Escalate to data team if missing master data

## Escalation Criteria
- Price discrepancy > £5 on high-volume product
- Affects multiple stores (regional issue)
- Price change user unknown (audit gap)
- Customer complaint escalated by store manager

## Expected Resolution Time
- Simple query: 5-10 minutes
- Price correction: 30-60 minutes
- Data issue: 2-4 hours (requires data team)

