# Basket Segments File Drop Failure Runbook

## Issue Description
Basket segments feed from LHS (file drop process) fails with timeout errors.

## Symptoms
- Splunk alerts showing `CreateBasketSegmentsProcessor` errors
- Error message: `file pick-up process failed`
- Timeout: `java.io.InterruptedIOException: timeout`

## Diagnosis Steps

### 1. Check Splunk Logs
Query: `index=price-advisory-service "CreateBasketSegmentsProcessor" "file pick-up process failed"`

Look for:
- Timeout patterns
- File location issues
- Network connectivity to SharePoint/OneDrive

### 2. Verify File Availability
- Check if file was uploaded to the expected SharePoint location
- Verify file permissions and naming conventions
- Confirm file size and format

### 3. Check New Relic Metrics
Query: `SELECT average(duration) FROM Transaction WHERE appName = 'price-advisory-service' FACET name SINCE 1 hour ago`

Look for:
- Increased latency in file operations
- Memory issues during processing

## Tools to Use
- **splunk_search**: Search for timeout errors and patterns
- **newrelic_metrics**: Check service performance and memory
- **basket_segment_get**: Verify current basket segment data

## Resolution Steps

### If Timeout is Transient:
1. Retry the file drop process
2. Monitor for 15 minutes
3. If successful, mark as resolved

### If File Missing or Corrupted:
1. Contact data team to re-upload file
2. Verify file format matches schema
3. Trigger manual reprocessing

### If Service Performance Issue:
1. Check memory usage via New Relic
2. Scale up service if needed
3. Clear cache and restart pods

## Escalation Criteria
- File missing for > 2 hours
- Timeout persists after 3 retries
- Memory usage > 90%
- Business impact: Basket segments not updated for pricing decisions

## Expected Resolution Time
- Normal: 15-30 minutes (retry)
- Complex: 1-2 hours (file reupload + validation)
- Critical: Immediate escalation if pricing decisions blocked

