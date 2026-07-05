# Reuters NAP 多品种交易看板

这是一个独立的 Streamlit + Plotly 交易研究终端，用于解析 Reuters Excel 拉取的 `Nap.xlsx` 日度时间序列，并提供行情地图、序列详情、季节性、关系分析、组合价差、远期曲线、风险、运费套利和价格词典。

## 启动

```powershell
pip install -r requirements.txt
streamlit run src/nap_dashboard.py
```

本机快捷方式会调用 `start_nap_dashboard.ps1`，脚本会优先使用项目内 `.venv`。这样可以避免全局 Anaconda 依赖版本变化影响看板；如果 `.venv` 不存在，脚本才会回退到系统里的 Streamlit。

默认数据源：

```text
C:\Users\74100\Nutstore\1\油气-djx-\NAP-丙烯-坚果云\Nap_calendar_month_live_formula.xlsx
```

看板左侧支持直接拖入新的 `Nap.xlsx` 或 `Nap_calendar_month_live_formula.xlsx`，也可以输入本机完整路径。页面会按“路径 + 文件大小 + 修改时间”生成独立缓存；同名 Excel 更新后会自动进入新缓存，不会误用旧 parquet 或 pickle。点击 `重新解析当前 Excel` 可以强制刷新。

侧边栏提供两种查看模式：

- `连续月 C1-C12`：沿用 Reuters 原始连续月序列，适合看 M1-M2、远期曲线和传统近远月结构。
- `自然月 1-12月`：使用 C1-C12 按交易日月份反算出的自然月序列，适合比较 1月、2月等固定自然月。自然月模式下可以勾选任意月份组合，比如只看 3-5月旺季或 9-12月窗口。

`Nap` sheet 新增的 `Naphtha CIF NWE Outright Swap Monthly Continuation 1-12` 会自动归入 `Naphtha / NWE CIF石脑油 / 西北欧`；`Naphtha CIF NWE Swap Assessments Crack Spread Month Continuation 1-12` 会自动归入 `Cracks / NWE石脑油裂解 / 西北欧`。

## 数据刷新

解析器位于 `src/nap_adapter.py`。它会识别 `_xll.RDP.HistoricalPricing` 和 `_xll.RHistory` 输出区域，把横向 Reuters 宽表转换为标准长表：

```text
date, series_id, display_name, value, sheet, sector, product, region,
contract_month, term_type, calendar_month, unit_native, unit_normalized,
ric, is_derived, source
```

其中 `term_type=continuous` 表示原始 C1-C12 连续月；`term_type=calendar` 表示看板按公式逻辑从连续月反算出的 1-12 月自然月序列。自然月序列不依赖 Excel 公式缓存，即使 Excel 公式页没有保存计算结果，看板也会用原始 C1-C12 数据重新计算。

默认缓存基础路径：

```text
data/processed/nap_timeseries.parquet
```

缓存会按 workbook 签名写入独立 parquet；缓存命中时只读已处理好的长表，Excel 文件更新或点击页面左侧 `重新解析当前 Excel` 时才会重新扫描 workbook。若当前环境无法写 parquet，会自动写入同目录 `.pkl` fallback。手动刷新：

```powershell
python -m src.nap_adapter --refresh
```

## Catalog 与解释

首次解析会生成：

```text
config/nap_series_catalog.yaml
```

可以人工编辑 `display_name`、`sector`、`product`、`region`、`contract_month`、`unit_native`、`unit_normalized`、`ric` 等字段。

价格说明位于：

```text
config/nap_explanations.yaml
```

`序列详情` 和 `价格词典` 会读取这里的中文说明。可以在 `series` 下补充单条序列说明，也可以在 `defaults` 下维护板块通用解释。

## 价差公式

公式库位于：

```text
config/nap_formula_registry.yaml
```

每条公式由若干带权重的腿组成。腿可以直接指定 `series_id`，也可以用 `product`、`region`、`contract_month`、`ric`、`contains` 等条件选择序列。

```yaml
- name: RBOB - WTI
  legs:
    - weight: 1
      selector: {product: RBOB, contract_month: M1}
    - weight: -1
      selector: {product: WTI, contract_month: M1}
```

`关系实验室` 可以调用公式库，也可以手动选择多条序列构建临时价差。

## 组合价差与月差

`组合价差` 页面包含三组分析：

- `汽油-石脑油逐月组合`：计算新加坡92RON汽油纸货 M1-M12 减 MOPJ M1-M12，以及欧洲 EBOB 汽油纸货 M1-M12 减 NWE CIF 石脑油 M1-M12。
- `MOPJ 驱动对比`：把 MOPJ 升贴水、MOPJ 裂解和 Dated Brent/Brent 代理序列放到同一张图里，默认用 250D z-score 便于比较节奏。
- `月差分析`：按板块、品种、地区选择 M1-Mn 曲线，展示 M1-M2、M1-M3、M1-M6、M2-M3 等月差。

单位统一规则：

| 代码/品种 | Reuters 原始单位 | 看板计算单位 | 换算 | 主要用途 |
| --- | --- | --- | --- | --- |
| RBOB、Heating Oil / HO | USD/gal | USD/bbl | `value * 42` | RBOB-WTI、HO-WTI 裂解/价差 |
| Singapore 92 汽油纸货 | USD/bbl | USD/bbl | 不换算 | Singapore 92 - MOPJ |
| MOPJ / Japan CFR Naphtha | USD/mt | USD/bbl | `value / 8.90` | Singapore 92 - MOPJ |
| Naphtha CIF NWE outright | USD/mt | USD/bbl | `value / 8.90` | EBOB - NWE 石脑油 |
| EBOB NWE 汽油纸货 | USD/mt | USD/bbl | `value / 8.33` | EBOB - NWE 石脑油 |
| LSGO / Low Sulphur Gasoil | USD/mt | USD/bbl | `value / 7.45` | LSGO-Brent |

原始 `value` 永远保留 Reuters 拉取值；跨品种价差、裂解和周报图默认使用 `value_normalized`。涉及桶吨换算的价差会在页面说明里标出公式，例如：

```text
Singapore 92 - MOPJ/8.90
EBOB/8.33 - NWE Naphtha/8.90
```

## 验证

```powershell
pytest -q
```

当前公式 workbook 可识别 938 条序列，其中连续月 578 条、自然月 360 条，并检查 `(date, series_id)` 无重复。
