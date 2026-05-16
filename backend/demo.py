import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from signal_rules import generate_all_signals


def load_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV缺少必要列: {col}")
    return df


def generate_sample_data(stock_code='000001.SZ', start_date='2020-01-01', end_date='2026-05-01'):
    np.random.seed(hash(stock_code) % 2**32)
    dates = pd.bdate_range(start=start_date, end=end_date)
    n = len(dates)

    base_price = 13.0
    trend = np.linspace(0, 0.3, n)
    cycle1 = np.sin(np.linspace(0, 8 * np.pi, n)) * 0.08
    cycle2 = np.sin(np.linspace(0, 20 * np.pi, n)) * 0.03
    noise = np.random.normal(0, 0.015, n)
    returns = 0.0005 + trend / n + cycle1 + cycle2 + noise
    returns = np.clip(returns, -0.095, 0.095)

    prices = base_price * np.cumprod(1 + returns)
    closes = prices
    opens = closes * (1 + np.random.uniform(-0.012, 0.012, n))
    highs = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0.008, 0.006, n)))
    lows = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0.008, 0.006, n)))
    base_vol = 8000000
    vol_trend = np.linspace(0, 0.4, n)
    volumes = (base_vol * (1 + vol_trend) * (1 + np.random.uniform(-0.3, 0.5, n))).astype(int).astype(float)

    df = pd.DataFrame({
        'date': dates.strftime('%Y-%m-%d'),
        'open': np.round(opens, 2),
        'high': np.round(highs, 2),
        'low': np.round(lows, 2),
        'close': np.round(closes, 2),
        'volume': volumes,
    })
    return df


def main():
    print("=" * 70)
    print("  股票交易信号生成系统 - Demo演示")
    print("  Stock Trading Signal Generator")
    print("=" * 70)

    stock_code = '000001.SZ'
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    csv_path = os.path.join(data_dir, f'{stock_code}.csv')

    use_real_data = False
    if os.path.exists(csv_path):
        try:
            df = load_from_csv(csv_path)
            use_real_data = True
            print(f"\n[数据源] 本地CSV文件: {csv_path}")
        except Exception as e:
            print(f"\n[警告] CSV加载失败 ({e})，使用模拟数据")
    else:
        print(f"\n[数据源] 未找到本地数据文件，使用模拟数据")

    if not use_real_data:
        df = generate_sample_data(stock_code=stock_code)
        print(f"  股票代码: {stock_code}")
        print(f"  模拟日期范围: 2020-01-01 ~ 2026-05-01")

    print(f"\n[数据概览]")
    print(f"  总交易日数: {len(df)}")
    print(f"  起始日期:   {df['date'].iloc[0]}")
    print(f"  截止日期:   {df['date'].iloc[-1]}")
    print(f"  价格范围:   {df['low'].min():.2f} ~ {df['high'].max():.2f}")

    print(f"\n{'='*70}")
    print(f"[信号生成中...] 正在运行全部规则...")
    print(f"{'='*70}\n")

    result_df, all_rules = generate_all_signals(df)

    total_buy = result_df['buy_signal'].sum()
    total_sell = result_df['sell_signal'].sum()

    print(f"[信号统计汇总]")
    print(f"  {'规则名称':<22} {'买入信号':>10} {'卖出信号':>10} {'占比':>10}")
    print(f"  {'-'*54}")
    for name, b, s in all_rules:
        b_cnt = int(b.sum())
        s_cnt = int(s.sum())
        pct = f"{(b_cnt + s_cnt) / len(df) * 100:.2f}%"
        print(f"  {name:<22} {b_cnt:>10} {s_cnt:>10} {pct:>10}")
    print(f"  {'-'*54}")
    combined_pct = f"{(total_buy + total_sell) / len(df) * 100:.2f}%"
    print(f"  {'合并后（去冲突）':<22} {total_buy:>10} {total_sell:>10} {combined_pct:>10}")

    buy_dates = result_df[result_df['buy_signal']]['date'].tolist()
    sell_dates = result_df[result_df['sell_signal']]['date'].tolist()

    print(f"\n[最近信号详情 - 最近10个交易日]")
    print(f"  {'日期':<14} {'收盘价':>10} {'信号':>8} {'涨跌':>10}")
    recent = result_df.tail(10)
    for _, row in recent.iterrows():
        sig = "----"
        if row['buy_signal']:
            sig = "【买入】"
        elif row['sell_signal']:
            sig = "【卖出】"
        chg = ""
        if len(recent) > 1:
            prev_close = recent.iloc[list(recent.index).index(row.name) - 1]['close'] if list(recent.index).index(row.name) > 0 else row['close']
            if prev_close != row['close']:
                chg_val = (row['close'] - prev_close) / prev_close * 100
                chg = f"{chg_val:+.2f}%"
        print(f"  {row['date']:<14} {row['close']:>10.2f} {sig:>8} {chg:>10}")

    output_path = os.path.join(data_dir, 'signal_output.csv')
    os.makedirs(data_dir, exist_ok=True)
    output_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'buy_signal', 'sell_signal']
    result_df[output_cols].to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n[输出] 结果已保存至: {output_path}")
    print(f"       包含列: {', '.join(output_cols)}")

    print(f"\n{'='*70}")
    print(f"  Demo运行完成!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
