# Reuters NAP 多品种交易看板

这是一个独立的 Streamlit + Plotly 交易研究终端，用于解析 Reuters Excel 拉取的 `Nap.xlsx` 日度时间序列，并提供行情地图、序列详情、季节性、关系分析、组合价差、远期曲线、风险、运费套利和价格词典。

## 启动

```powershell
pip install -r requirements.txt
streamlit run src/nap_dashboard.py
```

默认数据源：

```text
C:\Users\74100\Nutstore\1\油气-djx-\NAP-丙烯-坚果云\Nap.xlsx
```

看板左侧支持直接拖入新的 `Nap.xlsx`，也可以输入本机完整路径。页面会按“路径 + 文件大小 + 修改时间”生成独立缓存；同名 Excel 更新后会自动进入新缓存，不会误用旧 parquet 或 pickle。点击 `重新解析当前 Excel` 可以强制刷新。

`Nap` sheet 新增的 `Naphtha CIF NWE Outright Swap Monthly Continuation 1-12` 会自动归入 `Naphtha / NWE CIF石脑油 / 西北欧`；`Naphtha CIF NWE Swap Assessments Crack Spread Month Continuation 1-12` 会自动归入 `Cracks / NWE石脑油裂解 / 西北欧`。

## 数据刷新

解析器位于 `src/nap_adapter.py`。它会识别 `_xll.RDP.HistoricalPricing` 和 `_xll.RHistory` 输出区域，把横向 Reuters 宽表转换为标准长表：

```text
date, series_id, display_name, value, sheet, sector, product, region,
contract_month, unit_native, unit_normalized, ric, is_derived, source
```

默认缓存基础路径：

```text
data/processed/nap_timeseries.parquet
```

如果当前环境无法写 parquet，会自动写入同目录 `.pkl` fallback。手动刷新：

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

石脑油从 `USD/mt` 换算成 `USD/bbl` 时使用 `1 metric tonne = 8.90 barrels`，因此公式为：

```text
USD/bbl = USD/mt / 8.9
```

这个换算用于 MOPJ 和 NWE CIF 石脑油对汽油纸货的逐月组合价差。原始 `value` 仍按 Reuters 拉取值保留，组合图使用换算后的腿计算。

## 验证

```powershell
pytest -q
```

当前本地 workbook 可识别 545 条序列，其中 Crude 160 条、Crk 120 条、Nap 64 条，并检查 `(date, series_id)` 无重复。
