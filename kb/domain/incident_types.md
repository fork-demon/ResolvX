# Incident Type Definitions and Handling Procedures

## Pricing and Competitor Data Incidents

### Price Not Found (NOF Issue)
- **Severity**: Medium  
- **Escalation Level**: Medium
- **Primary Team**: Pricing Engineering  
- **SLA Response**: 1 hour
- **SLA Resolution**: 4 hours

**Description**: Price missing for a GTIN in store or online.

**Common Causes & Team Assignment**:
1. **Price missing in Quote service** → Assign to **Quote Team**
   - Price API has price but Quote doesn't
   - Quote service errors in Splunk logs
   
2. **Price needs republishing** → Assign to **Adaptor Team**
   - Price exists but in DRAFT state
   - Price lifecycle stuck in intermediate state
   - Adaptor logs show processing errors
   
3. **Product not active** → Assign to **Product Team**
   - Product status is inactive/discontinued
   - Product not in active catalog
   - Product data incomplete
   
4. **Policy/Configuration issue** → Add **Comment** (Your Team handles)
   - Price policy misconfigured
   - Cluster/location mapping issue
   - Pricing rules need adjustment

**Procedures**:
1. Check Price API using `base_prices_get` tool
2. Search Splunk logs using `splunk_search` for Quote/Adaptor errors
3. Validate product status
4. **Decision Logic**:
   - IF price in API but not in Quote → **Assign to Quote Team**
   - IF price in DRAFT or needs republish → **Assign to Adaptor Team**
   - IF product inactive → **Assign to Product Team**
   - IF policy/config issue → **Add Comment** + keep with your team
   - IF unclear → **Escalate to Human**

**Tools**: base_prices_get, splunk_search, poll_queue

**Escalation Criteria**:
- Price missing for high-volume product
- Customer-facing impact
- Multiple GTINs affected
- Unable to determine root cause after tool execution

**Templates**: price_validation_report, team_assignment_notification

---

### Incorrect Price
- **Severity**: Medium  
- **Escalation Level**: Medium
- **Primary Team**: Pricing Engineering  
- **SLA Response**: 1 hour
- **SLA Resolution**: 4 hours

**Description**: Price is outdated or incorrect.

**Common Causes & Team Assignment**:
1. **Price stuck in DRAFT state** → Assign to **Adaptor Team**
   - Price not published from draft
   - Workflow stuck in intermediate state
   - Adaptor processing errors
   
2. **Quote service has wrong price** → Assign to **Quote Team**
   - Price API is correct but Quote is outdated
   - Sync issues between Price API and Quote
   - Quote service not updating
   
3. **Competitor pricing misaligned** → Add **Comment** (Your Team handles)
   - Price rules need adjustment
   - Competitive positioning decision needed
   - Policy review required
   
4. **Product worksheet/lifecycle issue** → Assign to **Product Team**
   - Product in worksheet state
   - Lifecycle not progressed correctly
   - Product data needs update

**Procedures**:
1. Get current price using `base_prices_get` tool
2. Check competitor prices using `competitor_prices_get` tool
3. Search Splunk logs for errors using `splunk_search`
4. **Decision Logic**:
   - IF price in DRAFT state → **Assign to Adaptor Team**
   - IF Price API correct but Quote wrong → **Assign to Quote Team**
   - IF competitive/policy issue → **Add Comment** + keep with your team
   - IF product lifecycle issue → **Assign to Product Team**
   - IF unclear → **Escalate to Human**

**Tools**: base_prices_get, competitor_prices_get, splunk_search

**Escalation Criteria**:
- Price discrepancy affects revenue
- Regulatory compliance issue
- Customer complaints received
- Complex multi-team coordination needed

**Templates**: price_correction_report, team_assignment_notification

---

### Competitor Promotional File Processing Failed
- **Severity**: Medium  
- **Escalation Level**: Medium
- **Handling Team**: Competitive Intelligence  
- **SLA Response**: 2 hours
- **SLA Resolution**: Next scheduled run

**Description**: Failure during competitor promotional CSV file ingestion from SharePoint (business user uploads).

**Common Causes**:
- Microsoft token/authentication issue for SharePoint access
- CSV file not processed by ingestion pipeline (check Splunk logs)
- CSV file not moved to archive folder in SharePoint after processing
- CSV file format or validation errors
- Business user uploaded CSV to wrong SharePoint folder

**Procedures**:
1. Use `splunk_search` to verify if CSV file was successfully processed in a subsequent run
2. Use `sharepoint_list_files` to check if CSV is in archive folder or still in upload folder
3. Check Microsoft token expiration in logs
4. Validate CSV file format if accessible
5. If not processed after 2+ attempts:
   - Competitive Intelligence squad refreshes token
   - Validates CSV format
   - Contacts business user if CSV format invalid
   - Manually reprocesses or escalates
6. Document processing status and retry attempts
7. Monitor next scheduled run

**Tools**: splunk_search, sharepoint_list_files, sharepoint_search_documents

**Escalation Criteria**:
- CSV file not processed after 2 retry attempts
- Critical promotional data missing (time-sensitive)
- CSV format validation fails repeatedly
- Microsoft token refresh fails

**Templates**: file_processing_report, business_user_notification

---

### Basket Segment File Processing Failed
- **Severity**: Medium  
- **Escalation Level**: Medium
- **Handling Team**: Competitive Intelligence  
- **SLA Response**: 2 hours
- **SLA Resolution**: Next scheduled run

**Description**: Failure during basket segment CSV file ingestion from SharePoint (business user uploads).

**Common Causes**:
- Microsoft token/authentication issue for SharePoint access
- CSV file not processed by ingestion pipeline (check Splunk logs)
- CSV file not moved to archive folder in SharePoint after processing
- CSV file format or validation errors
- Business user uploaded CSV to wrong SharePoint folder
- CSV contains invalid TPNB or location cluster data

**Procedures**:
1. Use `splunk_search` to verify if CSV file was successfully processed in a subsequent run
2. Use `sharepoint_list_files` to check if CSV is in archive folder or still in upload folder
3. Check Microsoft token expiration in logs
4. Validate CSV file format and data if accessible
5. Use `basket_segment_get` to verify if data is available in system (if TPNB known)
6. If not processed after 2+ attempts:
   - Competitive Intelligence squad refreshes token
   - Validates CSV format and data
   - Contacts business user if CSV invalid
   - Manually reprocesses or escalates
7. Document processing status and retry attempts
8. Monitor next scheduled run

**Tools**: splunk_search, basket_segment_get, sharepoint_list_files, sharepoint_search_documents

**Escalation Criteria**:
- CSV file not processed after 2 retry attempts
- Business reporting deadline approaching
- Multiple consecutive CSV upload failures
- CSV format validation fails repeatedly

**Templates**: file_processing_report, business_user_notification

---

## Additional Context

These incident types are designed for pricing and competitive intelligence operations. The LLM will:
1. **Match incident descriptions** to these types based on keywords (price, GTIN, competitor, basket segment, file processing)
2. **Recommend appropriate tools** based on the Tools section (base_prices_get, competitor_prices_get, splunk_search, sharepoint tools)
3. **Guide supervisor decisions** based on Procedures and Escalation Criteria
4. **Determine severity** and escalation needs automatically

The supervisor agent will use this information to:
- Decide whether to add comments or escalate
- Select the appropriate handling team
- Apply correct SLA expectations
- Choose relevant templates for communication
