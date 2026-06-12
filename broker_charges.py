from config import BROKER_CONFIGS

def calculate_charges(trade_type: str, quantity: float, price: float, broker: str) -> dict:
    """
    Calculate precise transaction charges for Indian stock market delivery trades.
    Includes Brokerage, STT, Exchange Transaction Charges, SEBI Turnover fees, GST, and Stamp Duty.
    """
    trade_value = quantity * price
    cfg = BROKER_CONFIGS.get(broker)
    if not cfg:
        # Default to Zerodha if broker is invalid
        cfg = BROKER_CONFIGS["Zerodha"]

    # 1. Brokerage
    brokerage = 0.0
    if cfg["brokerage_percent"] > 0:
        brokerage = (cfg["brokerage_percent"] / 100.0) * trade_value
        if cfg["brokerage_max"] > 0:
            brokerage = min(brokerage, cfg["brokerage_max"])
    if cfg.get("brokerage_flat", 0.0) > 0:
        brokerage += cfg["brokerage_flat"]

    # 2. STT (Securities Transaction Tax) - 0.1% on buy & sell for Equity Delivery
    stt = (cfg["stt_percent"] / 100.0) * trade_value

    # 3. Exchange Transaction Charges - NSE Delivery: 0.00322%
    exch_txn = (cfg["exchange_txn_percent"] / 100.0) * trade_value

    # 4. SEBI Turnover Fees - 0.0001% (Rs 10/crore)
    sebi = (cfg["sebi_turnover_percent"] / 100.0) * trade_value

    # 5. Stamp Duty - 0.015% on buy only for Delivery
    stamp = 0.0
    if trade_type.lower() == "buy":
        stamp = (cfg["stamp_duty_percent"] / 100.0) * trade_value

    # 6. GST - 18% on (Brokerage + Exchange Txn Charges + SEBI Turnover Fee)
    gst = (cfg["gst_percent"] / 100.0) * (brokerage + exch_txn + sebi)

    total_charges = brokerage + stt + exch_txn + sebi + stamp + gst

    return {
        "trade_value": round(trade_value, 2),
        "brokerage": round(brokerage, 2),
        "stt": round(stt, 2),
        "exchange_txn": round(exch_txn, 2),
        "sebi": round(sebi, 2),
        "stamp_duty": round(stamp, 2),
        "gst": round(gst, 2),
        "total_charges": round(total_charges, 2)
    }

def find_break_even_price(quantity: float, buy_price: float, broker: str) -> float:
    """
    Find the exact sell price per share required to break even after factoring in
    both buy-side and sell-side transaction fees. Uses binary search for high accuracy.
    """
    if quantity <= 0:
        return buy_price
        
    buy_res = calculate_charges("buy", quantity, buy_price, broker)
    target_net_outflow = (quantity * buy_price) + buy_res["total_charges"]

    # Binary search boundaries for break-even sell price
    low = buy_price
    high = buy_price * 1.25  # Assume fees don't exceed 25% of trade value (usually <1%)
    
    # In case of custom brokers with high percentages, scale high boundary
    cfg = BROKER_CONFIGS.get(broker, BROKER_CONFIGS["Zerodha"])
    if cfg["brokerage_percent"] > 0:
        high = buy_price * (1.0 + (cfg["brokerage_percent"] * 4) / 100.0)

    iterations = 0
    while (high - low) > 0.001 and iterations < 50:
        mid = (low + high) / 2
        sell_res = calculate_charges("sell", quantity, mid, broker)
        net_inflow = (quantity * mid) - sell_res["total_charges"]
        
        if net_inflow < target_net_outflow:
            low = mid
        else:
            high = mid
        iterations += 1

    return round((low + high) / 2, 2)

def calculate_portfolio_metrics(quantity: float, buy_price: float, current_price: float, broker: str) -> dict:
    """
    Compute total buy cost, current value, gross P&L, transaction fees, and net P&L.
    """
    buy_charges = calculate_charges("buy", quantity, buy_price, broker)
    sell_charges = calculate_charges("sell", quantity, current_price, broker)

    total_buy_cost_gross = quantity * buy_price
    total_buy_cost_net = total_buy_cost_gross + buy_charges["total_charges"]

    current_value_gross = quantity * current_price
    current_value_net = current_value_gross - sell_charges["total_charges"]

    net_pnl = current_value_net - total_buy_cost_net
    net_pnl_percent = (net_pnl / total_buy_cost_net) * 100 if total_buy_cost_net > 0 else 0.0

    break_even = find_break_even_price(quantity, buy_price, broker)

    return {
        "buy_cost_gross": round(total_buy_cost_gross, 2),
        "buy_cost_net": round(total_buy_cost_net, 2),
        "buy_charges": round(buy_charges["total_charges"], 2),
        "current_value_gross": round(current_value_gross, 2),
        "current_value_net": round(current_value_net, 2),
        "sell_charges": round(sell_charges["total_charges"], 2),
        "total_charges": round(buy_charges["total_charges"] + sell_charges["total_charges"], 2),
        "net_pnl": round(net_pnl, 2),
        "net_pnl_percent": round(net_pnl_percent, 2),
        "break_even_price": break_even
    }

if __name__ == "__main__":
    # Quick verification
    print("Testing Zerodha calculations for 100 shares @ Rs 300:")
    charges = calculate_charges("buy", 100, 300, "Zerodha")
    print(charges)
    be = find_break_even_price(100, 300, "Zerodha")
    print(f"Break-even price: Rs {be}")
    metrics = calculate_portfolio_metrics(100, 300, be, "Zerodha")
    print(f"Net P&L at break-even: Rs {metrics['net_pnl']}")
