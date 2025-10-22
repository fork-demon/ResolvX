# Basket Segments File Drop Failure

## Issue Description
Basket segments feed from LHS (file drop process) fails with timeout errors.

## Symptoms
- Splunk alerts showing `CreateBasketSegmentsProcessor` errors
- Error message: `file pick-up process failed`
- Timeout: `java.io.InterruptedIOException: timeout`

## Diagnosis Steps

### 1. Check Splunk Logs
**Tool to use**: `splunk_search`

Query: `index=price-advisory-service "CreateBasketSegmentsProcessor" "file pick-up process failed"`

Look for:
- Timeout patterns
- File path issues
- Network connectivity problems
- Azure Graph API failures

### 2. Check Price Advisory API
**Tool to use**: `base_prices_get`

Verify the Price Advisory API is responding:
- Check if basket segment data can be retrieved
- Verify API health status

### 3. Check NewRelic Metrics (if performance-related)
**Tool to use**: `newrelic_metrics`

Query: `SELECT average(duration) FROM Transaction WHERE appName = 'price-advisory-service' SINCE 1 hour ago`

Look for:
- API response times
- Memory usage spikes
- Thread pool exhaustion

## Resolution Steps

1. **Immediate**: Check Azure Graph API connectivity
2. **Short-term**: Retry the file pick-up process manually
3. **Long-term**: Implement retry logic with exponential backoff

## Severity
**High** - Impacts pricing data pipeline

## Escalation
If issue persists > 2 hours, escalate to Engineering Team
