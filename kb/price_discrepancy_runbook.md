# Price Discrepancy Investigation

## Issue Description
Unexpected price moves or scanning at wrong price.

## Symptoms
- Customer reports: "Scanning at the wrong price"
- Price doesn't match expected value
- Price move happened without team approval

## Diagnosis Steps

### 1. Check Base Prices API
**Tool to use**: `base_prices_get`

Retrieve current price data:
- Verify TPNB price
- Check location cluster
- Confirm effective dates

### 2. Check Price Minimum API
**Tool to use**: `price_minimum_get`

Verify minimum pricing rules:
- Check if price violates minimum rules
- Verify basket segment classification

### 3. Check Competitor Prices (if relevant)
**Tool to use**: `competitor_prices_get`

Compare with competitor pricing:
- Verify competitive positioning
- Check for price matching rules

### 4. Check Splunk Audit Logs
**Tool to use**: `splunk_search`

Query: `index=price-audit "price change" TPNB={tpnb}`

Look for:
- Who loaded the price change
- When the change was made
- What system triggered the change
- Approval workflow status

## Resolution Steps

1. **Immediate**: Retrieve price history and user audit trail
2. **Verification**: Confirm if price change was authorized
3. **Correction**: If unauthorized, revert to previous price
4. **Communication**: Notify stakeholders of resolution

## Severity
**Medium** - Customer-facing issue but usually isolated

## Escalation
If unauthorized price change, escalate to Pricing Operations Manager
