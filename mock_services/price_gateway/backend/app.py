from fastapi import FastAPI, Header, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone


app = FastAPI(title="Price Lifecycle Mock API", version="0.1.0")


class QtyContents(BaseModel):
    totalQuantity: float
    quantityUom: str


class CalculateMinPriceRequest(BaseModel):
    gtin: str
    locationClusterIds: List[str]
    qtyContents: QtyContents
    taxTypeCode: Optional[str] = None
    supplierABV: Optional[str] = None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@app.post("/pricelifecycle/v1/minimumPrices/calculate")
def calculate_min_price(
    payload: CalculateMinPriceRequest,
    authorization: Optional[str] = Header(default=None),
    teamnumber: Optional[str] = Header(default=None),
    traceid: Optional[str] = Header(default=None),
):
    # Simple deterministic mock calculation for demo purposes
    base = 1.0
    if payload.supplierABV:
        try:
            base += float(payload.supplierABV) * 0.05
        except Exception:
            base += 0.0
    base += (payload.qtyContents.totalQuantity or 0) * 0.002
    if payload.taxTypeCode:
        base *= 1.05

    clusters: List[Dict[str, Any]] = []
    for cid in payload.locationClusterIds:
        clusters.append({
            "locationClusterId": cid,
            "minPrice": round(base, 2),
            "currency": "GBP",
            "effectiveDateTime": now_iso(),
        })

    return {
        "gtin": payload.gtin,
        "clusterPrices": clusters,
        "traceId": traceid,
        "team": teamnumber,
        "calculatedAt": now_iso(),
    }


@app.get("/pricelifecycle/v1/minimumPrices")
def get_minimum_price(
    locationClusterId: str = Query(...),
    gtin: str = Query(...),
    effectiveDateTime: Optional[str] = Query(default=None),
    authorization: Optional[str] = Header(default=None),
    teamnumber: Optional[str] = Header(default=None),
    traceid: Optional[str] = Header(default=None),
):
    price = 2.49 if gtin[-1] in {"1", "3", "5", "7", "9"} else 1.99
    return {
        "gtin": gtin,
        "locationClusterId": locationClusterId,
        "minPrice": price,
        "currency": "GBP",
        "effectiveDateTime": effectiveDateTime or now_iso(),
        "traceId": traceid,
        "team": teamnumber,
    }


@app.get("/pricelifecycle/v1/basketSegments")
def get_basket_segment(
    tpnb: str = Query(...),
    locationClusterId: str = Query(...),
    subClass: str = Query(...),
    authorization: Optional[str] = Header(default=None),
    teamnumber: Optional[str] = Header(default=None),
    traceid: Optional[str] = Header(default=None),
):
    # Mock segmentation based on simple hash of inputs
    key = f"{tpnb}:{subClass}"
    segment = ["Value", "Core", "Premium"][hash(key) % 3]
    return {
        "tpnb": tpnb,
        "locationClusterId": locationClusterId,
        "subClass": subClass,
        "segment": segment,
        "computedAt": now_iso(),
        "traceId": traceid,
        "team": teamnumber,
    }


@app.get("/pricelifecycle/v1/competitorPrices")
def get_competitor_prices(
    locationClusterId: List[str] = Query(...),
    tpnb: str = Query(...),
    authorization: Optional[str] = Header(default=None),
    teamnumber: Optional[str] = Header(default=None),
    traceid: Optional[str] = Header(default=None),
):
    data = []
    for cid in locationClusterId:
        data.append({
            "locationClusterId": cid,
            "tpnb": tpnb,
            "competitors": [
                {"name": "CompA", "price": 2.35, "currency": "GBP"},
                {"name": "CompB", "price": 2.29, "currency": "GBP"},
            ],
            "asOf": now_iso(),
        })
    return {"results": data, "traceId": traceid, "team": teamnumber}


@app.get("/pricelifecycle/v1/advisory/promotions/competitor-promotional-prices")
def get_competitor_promotional_prices(
    locationClusterIds: List[str] = Query(...),
    mechanic: str = Query(...),
    tpnbs: List[str] = Query(...),
    authorization: Optional[str] = Header(default=None),
    teamnumber: Optional[str] = Header(default=None),
    traceid: Optional[str] = Header(default=None),
):
    items = []
    for cid in locationClusterIds:
        for t in tpnbs:
            items.append({
                "locationClusterId": cid,
                "tpnb": t,
                "mechanic": mechanic,
                "promotionalPrice": 1.79,
                "currency": "GBP",
                "asOf": now_iso(),
            })
    return {"results": items, "traceId": traceid, "team": teamnumber}


@app.get("/pricelifecycle/v1/advisory/promotions/effectiveness")
def get_promo_effectiveness(
    locationClusterIds: List[str] = Query(...),
    mechanic: str = Query(...),
    tpnbs: List[str] = Query(...),
    authorization: Optional[str] = Header(default=None),
    teamnumber: Optional[str] = Header(default=None),
    traceid: Optional[str] = Header(default=None),
):
    rows = []
    for cid in locationClusterIds:
        for t in tpnbs:
            rows.append({
                "locationClusterId": cid,
                "tpnb": t,
                "mechanic": mechanic,
                "lift": 0.12,
                "confidence": 0.82,
                "asOf": now_iso(),
            })
    return {"results": rows, "traceId": traceid, "team": teamnumber}


class PoliciesViewRequest(BaseModel):
    applicableEntities: List[str] = Field(default_factory=list)
    classifications: List[str] = Field(default_factory=list)
    clusters: List[str] = Field(default_factory=list)


@app.post("/pricelifecycle/v1/policies/view")
def policies_view(
    payload: PoliciesViewRequest,
    authorization: Optional[str] = Header(default=None),
    teamnumber: Optional[str] = Header(default=None),
    traceid: Optional[str] = Header(default=None),
):
    policies = []
    for cid in payload.clusters:
        for cls in payload.classifications or ["PRICE_CHANGE_POLICY"]:
            policies.append({
                "clusterId": cid,
                "classification": cls,
                "entities": payload.applicableEntities or ["priceChange", "initialPrice"],
                "policyId": f"POL-{cid[:6]}-{cls[:6]}",
                "version": 1,
                "effectiveDateTime": now_iso(),
            })
    return {"policies": policies, "traceId": traceid, "team": teamnumber}


@app.get("/pricelifecycle/v5/basePrices")
def get_base_prices(
    locationClusterId: str = Query(...),
    tpnb: str = Query(...),
    teamnumber: Optional[str] = Query(default=None),
    effectiveDateTime: Optional[str] = Query(default=None),
    authorization: Optional[str] = Header(default=None),
    traceid: Optional[str] = Header(default=None),
):
    # Default effectiveDateTime to current time if not provided
    effective = effectiveDateTime or now_iso()
    base_price = 2.15 if int(tpnb[-1]) % 2 == 0 else 2.45
    return {
        "tpnb": tpnb,
        "locationClusterId": locationClusterId,
        "basePrice": base_price,
        "currency": "GBP",
        "effectiveDateTime": effective,
        "traceId": traceid,
        "team": teamnumber,
    }


# Uvicorn entrypoint (internal backend for price gateway)
def _main():
    import uvicorn
    # Internal backend API - only accessible from price gateway
    uvicorn.run(app, host="127.0.0.1", port=8090)


if __name__ == "__main__":
    _main()


