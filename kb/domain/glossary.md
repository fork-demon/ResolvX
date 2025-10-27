# Business Domain Glossary

## Entity Definitions

### GTIN (Global Trade Item Number)
- **Type**: Product Identifier
- **Description**: International identifier for trade items; always 14 digits when zero-padded
- **Patterns**: `\b\d{12,14}\b`
- **Examples**: `05000123456789`
- **Synonyms**: barcode, ean, upc
- **Zero Pad Length**: 14

### TPNB (Tesco Product Number - Base)
- **Type**: Product Identifier
- **Description**: Internal product identifier; always 9 digits when zero-padded
- **Patterns**: `\b\d{6,10}\b`
- **Examples**: `1234567`
- **Synonyms**: product id, sku
- **Zero Pad Length**: 9

### LocationCluster
- **Type**: Location Group
- **Description**: Group of stores/locations treated as a unit for operations
- **Examples**: 
  - Large Store England
  - Express Scotland
  - ROI National
- **Synonyms**: region, area, store cluster

## Location Cluster Mapping

| Cluster Name | UUID |
|-------------|------|
| Large Store England | aec7c8db-013d-414d-bbdd-9226ed96d6fc |
| Large Store Scotland | 05e7673f-b28d-491a-82b4-08a836112e2a |
| Large Store Wales | 967badfc-9889-4b4b-b7e9-3e30e55316e7 |
| Large Store Northern Ireland | 36787780-018d-4555-80f0-d1d66ff5fa5b |
| Express England | f6458c43-25c0-4c24-935f-4f284041d573 |
| Express Scotland | f9a5d04f-5980-4a0a-9ee4-5af726962f9e |
| Express Wales | fdf0c7e6-15b4-49ee-ace2-1086bbd1d79c |
| Express Northern Ireland | 4cca015b-8fcb-4d36-a33b-92fc3fdf5402 |
| Express Isle of Man | e87ba962-61fb-45b0-aa93-c28696c89808 |
| Whoosh England | 5b95799f-773a-43bf-ba73-40179a29d56a |
| Whoosh Scotland | be3bc0c9-4252-4bb7-bf6d-cca84102ac3d |
| Whoosh Wales | 41da8745-3204-493d-951b-5ec1dbebb80a |
| Whoosh Northern Ireland | 9f60aa92-0b9c-4512-8f07-86a44454eb2a |
| Premium/Metro | 041fece7-aa93-42c6-9e76-72a43f599498 |
| ROI Express | 48973981-9008-48ed-8571-b2d708f135d0 |
| ROI National | 0adead7a-5d32-4efe-99d9-8aa9746cda3c |

## Tool Selection Guide

### Price Lookup Tools

#### base_prices_get
- **Purpose**: Get base price for TPNB at location cluster
- **Required Parameters**: 
  - `tpnb` (Tesco Product Number)
  - `locationClusterId` (Location Cluster UUID)

#### price_minimum_get
- **Purpose**: Get minimum price for GTIN and cluster
- **Required Parameters**: 
  - `gtin` (Global Trade Item Number)
  - `locationClusterId` (Location Cluster UUID)

#### price_minimum_calculate
- **Purpose**: Calculate minimum price for GTIN across clusters
- **Required Parameters**: 
  - `gtin` (Global Trade Item Number)
  - `locationClusterIds` (Array of Location Cluster UUIDs)
  - `qtyContents.totalQuantity` (Total quantity)
  - `qtyContents.quantityUom` (Unit of measure)
  - `taxTypeCode` (Tax type)
  - `supplierABV` (Supplier ABV)

### Basket Analysis Tools

#### basket_segment_get
- **Purpose**: Get basket segment for TPNB and cluster
- **Required Parameters**: 
  - `tpnb` (Tesco Product Number)
  - `locationClusterId` (Location Cluster UUID)
  - `subClass` (Product subclass)

### Competitor Analysis Tools

#### competitor_prices_get
- **Purpose**: Get competitor prices for TPNB across clusters
- **Required Parameters**: 
  - `tpnb` (Tesco Product Number)
  - `locationClusterIds` (Array of Location Cluster UUIDs)

#### competitor_promotional_prices_get
- **Purpose**: Get competitor promotional prices for TPNBs
- **Required Parameters**: 
  - `tpnbs` (Array of Tesco Product Numbers)
  - `locationClusterIds` (Array of Location Cluster UUIDs)

#### promo_effectiveness_get
- **Purpose**: Get promotion effectiveness for TPNBs
- **Required Parameters**: 
  - `tpnbs` (Array of Tesco Product Numbers)
  - `locationClusterIds` (Array of Location Cluster UUIDs)

### Policy Tools

#### policies_view
- **Purpose**: View pricing policies for clusters and classifications
- **Optional Parameters**: 
  - `tpnb` (Tesco Product Number)
  - `locationCluster` (Location Cluster name)
  - `policyType` (Type of policy)

## Entity Extraction Hints

1. **Prefer explicit labels near numbers** (e.g., "GTIN: 0500…")
2. **Disambiguate numeric identifiers** by expected length and context words
3. **Map natural language location names** to cluster UUID via mapping table

