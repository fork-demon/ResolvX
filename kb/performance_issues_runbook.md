# Performance Issues - Memory/CPU High

## Issue Description
Application experiencing high memory usage, CPU spikes, or slow response times.

## Symptoms
- NewRelic alert: "Memory Usage % is too high"
- Application responding slowly
- Timeouts in API calls
- Pod restarts due to OOM (Out of Memory)

## Diagnosis Steps

### 1. Check NewRelic Metrics
**Tool to use**: `newrelic_metrics`

Query: `SELECT average(cpuPercent), average(memoryUsagePercent) FROM SystemSample WHERE appName LIKE '%price%' SINCE 1 hour ago`

Look for:
- Memory usage trends
- CPU utilization patterns
- Garbage collection frequency
- Thread count

### 2. Check NewRelic Transaction Traces
**Tool to use**: `newrelic_metrics`

Query: `SELECT average(duration), percentile(duration, 95) FROM Transaction WHERE appName = 'price-cmd-api' SINCE 1 hour ago FACET name`

Identify:
- Slow endpoints
- Database query performance
- External API call latency

### 3. Check Splunk Application Logs
**Tool to use**: `splunk_search`

Query: `index=price* "OutOfMemoryError" OR "GC overhead limit" OR "timeout"`

Look for:
- Memory leak indicators
- Thread deadlocks
- Connection pool exhaustion

### 4. Check Price API Health
**Tool to use**: `base_prices_get`

Verify API is responding:
- Test a sample price retrieval
- Check response time

## Resolution Steps

1. **Immediate**: 
   - Check if pod restart resolves the issue
   - Scale up replicas if needed
2. **Short-term**: 
   - Identify memory leak source
   - Optimize slow queries
3. **Long-term**: 
   - Implement caching
   - Review heap size configuration
   - Add circuit breakers

## Severity
**High** - Can impact production availability

## Escalation
If multiple pods affected or issue persists > 30 mins, escalate to DevOps Team

