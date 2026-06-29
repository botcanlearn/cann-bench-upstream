# AiInfraMoeInitRoutingV3 算子 API 描述

## 1. 算子简介

AiInfraMoeInitRoutingV3算子

**主要应用场景**：
- MoE 模型门控路由初始化
- 大规模稀疏专家模型的前向路由
- Top-K 专家选择策略的 token 分发

**算子特征**：
- 难度等级：L3（MoE）
- 4 输入，4 输出，9 个属性参数
- 可选属性：active_num, expert_capacity, expert_num, drop_pad_mode, expert_tokens_num_type, expert_tokens_num_flag, quant_mode, active_expert_range, row_idx_type

## 2. 算子定义

### 数学公式

- 1.对输入expertIdx做排序，得出排序后的结果sortedExpertIdx和对应的序号sortedRowIdx：

    $$
    sortedExpertIdx, sortedRowIdx=keyValueSort(expertIdx,rowIdx)
    $$

  2.以sortedRowIdx做位置映射得出expandedRowIdxOut：
    - rowIdxType等于1时, 输出scatter索引

      $$
      expandedRowIdxOut[i]=sortedRowIdx[i]
      $$

    - rowIdxType等于0时, 输出gather索引

      $$
      expandedRowIdxOut[sortedRowIdx[i]]=i
      $$
      
  3.对sortedExpertIdx的每个专家统计直方图结果，得出expertTokensCountOrCumsumOutOptional：

    $$
    expertTokensCountOrCumsumOutOptional[i]=Histogram(sortedExpertIdx)
    $$

  4.如果quantMode不等于-1, 计算quant结果：
     - 静态quant

     $$
     quantResult=round((x∗scaleOptional)+offsetOptional)
     $$
     
    - 动态quant：
        - 若不输入scale：

            $$
            dynamicQuantScaleOutOptional = row\_max(abs(x)) / 127
            $$

            $$
            quantResult = round(x / dynamicQuantScaleOutOptional)
            $$

        - 若输入scale:

            $$
            dynamicQuantScaleOutOptional = row\_max(abs(x * scaleOptional)) / 127
            $$

            $$
            quantResult = round(x / dynamicQuantScaleOutOptional)
            $$
  
  5.若活跃的expert范围为全专家范围时，按照Scatter索引搬运token；反之按照Gather索引搬运token。在dropPadMode为1时将每个专家需要处理的Token个数对齐为expertCapacity个，超过expertCapacity个的Token会被Drop，不足的会用0填充。得出expandedXOut：
    - 非量化场景
      - 按照Scatter索引搬运

      $$
      expandedXOut[i]=x[scatterRowIdx[i] // K]
      $$

      - 按照Gather索引搬运

      $$
      expandedXOut[gatherRowIdx[i]]=x[i // K]
      $$

    - 量化场景
      - 按照Scatter索引搬运

      $$
      expandedXOut[i]=quantResult[scatterRowIdx[i] // K]
      $$

      - 按照Gather索引搬运

      $$
      expandedXOut[gatherRowIdx[i]]=quantResult[i // K]
      $$

  6.expandedRowIdxOut的有效元素数量availableIdxNum，计算方式为expertIdx中activeExpertRangeOptional范围内的元素的个数

    $$
    availableIdxNum = |\{x\in expertIdx| expert\_start \le x<expert\_end \ \}|
    $$

### 特殊情况

| 输入 | 输出 |
|------|------|
| 各维度为 1 的退化 shape | 正常计算，输出 shape 与输入一致 |
| 空张量（某维度为 0） | 未定义行为，需避免 |

## 3. 接口规范

### 算子原型

```python
cann_bench.ai_infra_moe_init_routing_v3(Tensor x, Tensor expert_idx, Tensor? scale=None, Tensor? offset=None, int64 active_num=-1, int64 expert_capacity=-1, int64 expert_num=-1, int64 drop_pad_mode=0, int64 expert_tokens_num_type=0, bool expert_tokens_num_flag=false, int64 quant_mode=-1, list[int] active_expert_range=[0, 256], int64 row_idx_type=0) -> (Tensor expanded_x, Tensor expanded_row_idx, Tensor expert_tokens_count_or_cumsum, Tensor expanded_scale)
```

### 输入输出参数说明

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 158px">
  <col style="width: 120px">
  <col style="width: 333px">
  <col style="width: 400px">
  <col style="width: 212px">
  <col style="width: 100px">
  <col style="width: 107px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x（aclTensor）</td>
      <td>输入</td>
      <td>MOE的输入，即token特征输入</td>
      <td>shape为(NUM_ROWS, H)，quantMode=6时支持输入类型为HIFLOAT8</td>
      <td>FLOAT16、BFLOAT16、FLOAT32、INT8、HIFLOAT8</td>
      <td>ND</td>
      <td>2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertIdx（aclTensor）</td>
      <td>输入</td>
      <td>每一行特征对应的K个处理专家，里面元素专家id不能超过专家数</td>
      <td>shape为(NUM_ROWS, K)</td>
      <td>INT32</td>
      <td>ND</td>
      <td>2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleOptional（aclTensor）</td>
      <td>输入</td>
      <td>表示用于计算量化结果的参数</td>
      <td><ul>
        <li>如果不输入表示计算时不使用scale;</li>
        <li>非量化场景下为可选输入，如果输入则要求为1D的Tensor，shape为(NUM_ROWS,);</li>
        <li>静态量化场景必须输入，输入要求为1D的Tensor，shape为[1, ]；</li>
        <li>动态量化场景下为可选输入，如果输入则要求为2D的Tensor，shape为(expertEnd-expertStart, H)；</li>
        <li>MXFP8量化场景下（quantMode为2、3）不输入。</li>
        <li>HIF8直转和HIF8 PERTOKEN量化场景下（quantMode为6、8）不输入。</li>
        <li>HIF8 PERTENSOR量化场景下（quantMode为7）,输入要求为1D的Tensor，shape为[1, ]。</li>
        </ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>offsetOptional（aclTensor）</td>
      <td>输入</td>
      <td>表示用于计算quant结果的偏移值</td>
      <td><ul>
        <li>在非量化场景下不输入;</li><li>静态量化场景必须输入，输入要求为1D的Tensor，shape为[1, ]；</li>
        <li>动态量化、MXFP8量化、HIF8量化场景下不输入。</li>
      </ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activeNum（int64_t）</td>
      <td>输入</td>
      <td>表示总的最大处理row数，输出expandedXOut只有这么多行是有效的</td>
      <td>入参校验需大于等于0，0表示Dropless场景，大于0时表示Active场景，约束所有专家共同处理tokens总量。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertCapacity（int64_t）</td>
      <td>输入</td>
      <td>表示每个专家能够处理的tokens数</td>
      <td>入参校验大于0小于NUM_ROWS。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertNum（int64_t）</td>
      <td>输入</td>
      <td>表示专家数</td>
      <td>expertTokensNumType为key_value模式时，取值范围为[0, 5120]，其它模式取值范围[0, 10240]</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dropPadMode（int64_t）</td>
      <td>输入</td>
      <td>表示是否为DropPad场景</td>
      <td>取值为0和1
        <br>0：表示Dropless场景，该场景下不校验expertCapacity；
        <br>1：表示DropPad场景；
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertTokensNumType（int64_t）</td>
      <td>输入</td>
      <td>表示直方图的不同模式</td>
      <td>取值为0、1和2
        <br>0：表示 comsum 模式；
        <br>1：表示 count 模式；
        <br>2：表示 key_value 模式；
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertTokensNumFlag（bool）</td>
      <td>输入</td>
      <td>表示是否输出 expertTokensCountOrCumsumOut </td>
      <td>取值为false和true</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantMode（int64_t）</td>
      <td>输入</td>
      <td>表示不同量化场景</td>
      <td>取值为0、1、-1、2、3、6、7、8（不同产品支持情况有差异，见表后描述）
        <br>0：表示静态 quant 场景;
        <br>1：表示动态 quant 场景;
        <br>-1：表示不量化场景;
        <br>2：表示MXFP8量化场景，expandedXOut量化到FLOAT8_E5M2;
        <br>3：表示MXFP8量化场景，expandedXOut量化到FLOAT8_E4M3FN;
        <br>6：表示HIF8直转量化场景，expandedXOut量化到HIFLOAT8;
        <br>7：表示HIF8 PERTENSOR量化场景，expandedXOut按照pertensor模式量化到HIFLOAT8;
        <br>8：表示HIF8 PERTOKEN量化场景，expandedXOut按照pertoken模式量化到HIFLOAT8;
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activeExpertRangeOptional（aclIntArray）</td>
      <td>输入</td>
      <td>表示活跃的expert范围</td>
      <td>长度为2，数组内的值为[expertStart, expertEnd]，左闭右开，要求值大于等于0，并且expertEnd不大于expertNum；Drop/Pad场景下，expertStart等于0, expertEnd等于expertNum </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>rowIdxType（int64_t）</td>
      <td>输入</td>
      <td>表示expandedRowIdxOut使用的索引类型</td>
      <td>取值为0、1
        <br>0：表示gather类型的索引
        <br>1：表示scatter类型的索引</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expandedXOut（aclTensor）</td>
      <td>输出</td>
      <td>根据expertIdx进行扩展过的特征</td>
      <td><ul>
        <li>Dropless场景shape为[NUM_ROWS * K, H]。</li>
        <li>Active场景shape为[min(activeNum, NUM_ROWS * K), H]。</li>
        <li>Drop/Pad场景下要求是一个3D的Tensor，shape为[expertNum, expertCapacity, H]。</li>
        <li>非量化场景下数据类型同x，量化场景quantMode为0、1时数据类型支持INT8，quantMode为2、3时数据类型分别支持FLOAT8_E5M2、FLOAT8_E4M3FN，quantMode为6、7、8时数据类型支持HIFLOAT8。</li>
      </ul></td>
      <td>FLOAT16、BFLOAT16、FLOAT32、INT8、FLOAT8_E5M2、FLOAT8_E4M3FN、HIFLOAT8</td>
      <td>ND</td>
      <td>2</td>
      <td>-</td>
    </tr>
  </tbody></table>


### 输出
  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 158px">
  <col style="width: 120px">
  <col style="width: 333px">
  <col style="width: 400px">
  <col style="width: 212px">
  <col style="width: 100px">
  <col style="width: 107px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>expandedRowIdxOut（aclTensor）</td>
      <td>输出</td>
      <td>expandedXOut和x的索引映射关系</td>
      <td>输出shape为(NUM_ROWS*K, )， 前availableIdxNum个元素为有效数据，其余无效数据由rowIdxType决定：
        <ul><li>当rowIdxType为0时，无效数据由-1填充</li>
        <li>当rowIdxType为1时，无效数据未初始化</li></ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertTokensCountOrCumsumOut（aclTensor）</td>
      <td>输出</td>
      <td>输出每个专家处理的token数量的统计结果或累加值</td>
      <td><ul>
        <li>在expertTokensNumType为0时，表示activeExpertRangeOptional范围内expert在排序后处理token总数的前缀和。</li>
        <li>在expertTokensNumType为1时，表示activeExpertRangeOptional范围内expert对应的处理token的总数。</li>
        <li>在expertTokensNumType为2时，表示activeExpertRangeOptional范围内token总数为非0的expert，以及对应expert处理token的总数。</li>
      </ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1-2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expandedScaleOut（aclTensor）</td>
      <td>输出</td>
      <td>输出不同量化过程中scaleOptional的中间值。</td>
      <td> 输出shape为expandedXOut的shape去掉最后一维之后所有维度的乘积。
        <ul style="list-style-type: circle;">
        <li>非量化场景下，当scaleOptional输入时，前availableIdxNum个元素为有效数据。</li>
        <li>动态量化场景下，当scaleOptional输入时，前availableIdxNum个元素为有效数据。</li>
        <li>静态量化场景下不输出。</li>
        <li>MXFP8量化场景下，输出FLOAT8_E8M0类型，Shape为[NUM_ROWS*K, M]，其中M=CeilAlign(CeilDiv(H,32),2)，NUM_ROWS*K的前availableIdxNum行为有效数据。</li>
        <li>HIF直转8量化场景下，输出FLOAT32类型，Shape为[NUM_ROWS*K, ]，当scaleOptional输入时，前availableIdxNum个元素为有效数据。</li>
        <li>HIF8 PERTENSOR量化场景下，expandedScaleOut不输出。</li>
        <li>HIF8 PERTOKEN量化场景下，输出FLOAT32类型，Shape为[NUM_ROWS*K, 1]。</li></ul>
      </td>
      <td>FLOAT32、FLOAT8_E8M0</td>
      <td>ND</td>
      <td>1-2</td>
      <td>-</td>
    </tr>
  </tbody></table>

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| int8/float16/bfloat16/float32 | int8/float16/bfloat16/float32 |
| int8/float16/bfloat16/float32 | int32 |
| int8/float16/bfloat16/float32 | int64 |
| int8/float16/bfloat16/float32 | float32 |
| int32 | int8/float16/bfloat16/float32 |
| int32 | int32 |
| int32 | int64 |
| int32 | float32 |
| float32 | int8/float16/bfloat16/float32 |
| float32 | int32 |
| float32 | int64 |
| float32 | float32 |

### 规则与约束

- 无额外约束

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `ndim`（输入维度数） | 1 ~ 2 | cases 实测范围 |
| `dim_0`（第0维大小） | 0 ~ 1048576 | cases 实测范围 |
| `dim_1`（第1维大小） | 1 ~ 5974 | cases 实测范围 |
| `dtype` | bfloat16, float16, float32, int32, int8 | cases 实测覆盖 |
| `active_num` | -1 ~ 8388608 | cases 实测范围 |
| `expert_capacity` | 0 ~ 32 | cases 实测范围 |
| `expert_num` | 132 ~ 498 | cases 实测范围 |
| `drop_pad_mode` | 0 ~ 1 | cases 实测范围 |
| `expert_tokens_num_type` | 0 ~ 2 | cases 实测范围 |
| `expert_tokens_num_flag` | False ~ True | cases 实测范围 |
| `quant_mode` | -1 ~ 1 | cases 实测范围 |
| `row_idx_type` | 0 ~ 1 | cases 实测范围 |
| `active_expert_range` | [0, 124], [0, 150], [0, 251], [0, 256], [0, 266], [0, 27], [0, 498], [100, 103] | cases 实测值 |

## 4. 精度要求

采用[生态算子精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)进行验证。

**误差指标**：

1. 平均相对误差（MERE）：采样点中相对误差平均值

   $$
   \text{MERE} = \text{avg}(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+\text{1e-7}})
   $$

2. 最大相对误差（MARE）：采样点中相对误差最大值

   $$
   \text{MARE} = \max(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+\text{1e-7}})
   $$

**通过标准**：

| 数据类型 | FLOAT16 | BFLOAT16 | FLOAT32 | 
|----------|---------|----------|---------|
| **通过阈值(Threshold)** | 2^-10 | 2^-7 | 2^-13 |

当平均相对误差 MERE < Threshold，最大相对误差 MARE < 10 * Threshold 时判定为通过。

## 5. 标准 Golden 代码

```python
import torch
from typing import Optional, Tuple, List

def ai_infra_moe_init_routing_v3(
    x: torch.Tensor,
    expert_idx: torch.Tensor,
    scale: [torch.Tensor] = None,
    offset: Optional[torch.Tensor] = None,
    active_num: int = -1,
    expert_capacity: int = -1,
    expert_num: int = -1,
    drop_pad_mode: int = 0,
    expert_tokens_num_type: int = 0,
    expert_tokens_num_flag: bool = False,
    quant_mode: int = -1,
    active_expert_range: Optional[List[int]] = None,
    row_idx_type: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_dtype = x.dtype
    if x_dtype != torch.int8:
        x = x.to(torch.float64)
    else:
        x = x
    expert_idx = expert_idx
    if scale is not None:
        scale = scale
    if offset is not None:
        offset = offset
    expert_start = active_expert_range[0]
    expert_end = active_expert_range[1]
    num_rows = x.shape[0]
    h = x.shape[1]
    k = expert_idx.shape[1]

    expert_idx_in = expert_idx.clone().reshape(-1)
    actual_expert_total_num = int(torch.sum(
        (expert_idx >= expert_start) & (expert_idx < expert_end)).item())

    # print("cpu:",actual_expert_total_num)
    max_int32 = torch.iinfo(torch.int32).max
    expert_idx_in[expert_idx_in < expert_start] = max_int32
    sorted_expert_indices = torch.argsort(expert_idx_in, dim=-1, stable=True)
    sorted_expert_idx = expert_idx_in[sorted_expert_indices]

    if row_idx_type == 1:
        expanded_row_idx = sorted_expert_indices.clone()
    else:
        # gather
        expanded_row_idx = torch.ones(num_rows * k, dtype=torch.int32, device=x.device) * -1
        tmp_indices = torch.arange(actual_expert_total_num, dtype=torch.int32, device=x.device)
        expanded_row_idx[sorted_expert_indices[:actual_expert_total_num]] = tmp_indices

    # 计算直方图
    if not expert_tokens_num_flag:
        expert_tokens_count = None
    else:
        if drop_pad_mode == 0:
            if expert_tokens_num_type == 1:
                expert_tokens_count = torch.bincount(
                    sorted_expert_idx[:actual_expert_total_num] - expert_start,
                    minlength=(expert_end - expert_start))
            elif expert_tokens_num_type == 0:
                expert_tokens_count = torch.bincount(
                    sorted_expert_idx[:actual_expert_total_num] - expert_start,
                    minlength=(expert_end - expert_start))
                expert_tokens_count = torch.cumsum(expert_tokens_count, dim=0)
            elif expert_tokens_num_type == 2:
                # key-value
                unique_experts, counts = torch.unique(
                    sorted_expert_idx[:actual_expert_total_num], return_counts=True)
                expert_tokens_count = torch.stack([unique_experts.to(torch.int64), counts.to(torch.int64)], dim=1)
                pad_len = expert_num - expert_tokens_count.shape[0]
                if pad_len > 0:
                    pad_tensor = torch.zeros((pad_len, 2), dtype=torch.int64, device=x.device)
                    expert_tokens_count = torch.cat([expert_tokens_count, pad_tensor], dim=0)
        else:
            expert_tokens_count = torch.bincount(
                sorted_expert_idx[:actual_expert_total_num] - expert_start,
                minlength=(expert_end - expert_start))
        expert_tokens_count = expert_tokens_count.to(torch.int64)

    vaild_num = 0
    if drop_pad_mode == 0:
        if active_num <= 0:
            vaild_num = actual_expert_total_num
        else:
            vaild_num = min(active_num, actual_expert_total_num)
        expanded_scale = None
        expanded_x = x[sorted_expert_indices[:vaild_num] // k, :]
        if scale is not None and quant_mode == -1:
            expanded_scale = scale[sorted_expert_indices[:vaild_num] // k]
    else:
        # droppad=1时计算逻辑
        adapter_capacity(sorted_expert_indices, sorted_expert_idx, expert_capacity)

        sort_row_tmp = torch.full((expert_num * expert_capacity,), -1, dtype=torch.int64, device=x.device)
        offset_tmp = 0
        lastExpertId = 0
        for i in range(sorted_expert_indices.shape[0]):
            val = sorted_expert_indices[i].item()
            if val != -1:
                cur_expert = sorted_expert_idx[i].item()
                if lastExpertId != cur_expert:
                    offset_tmp = 0
                    lastExpertId = cur_expert
                sort_row_tmp[cur_expert * expert_capacity + offset_tmp] = val
                offset_tmp = offset_tmp + 1

        # expand_row_idx
        expanded_row_idx = torch.full(sorted_expert_indices.shape, -1, dtype=torch.int32, device=x.device)
        for i in range(sort_row_tmp.shape[0]):
            val = sort_row_tmp[i].item()
            if val != -1:
                expanded_row_idx[val] = i

        # expanded_x
        expanded_x_mask = torch.ones((expert_num * expert_capacity, h), dtype=torch.bool, device=x.device)
        expanded_x = torch.zeros((expert_num * expert_capacity, h), dtype=x.dtype, device=x.device)
        for i in range(sort_row_tmp.shape[0]):
            val = sort_row_tmp[i].item()
            if val != -1:
                expanded_x[i] = x[val // k]
                expanded_x_mask[i] = False

    # 非量化
    if quant_mode == -1:
        expanded_x = expanded_x
        expanded_row_idx = expanded_row_idx
        if scale is None or drop_pad_mode == 1:
            expanded_scale = None

    # 静态量化
    if quant_mode == 0:
        expanded_scale = None
        expanded_x_fp32 = expanded_x.to(torch.float32)
        scale_val = scale.to(torch.float32)
        offset_val = offset.to(torch.float32)
        scale_rst = expanded_x_fp32 * scale_val[0]
        add_offset = scale_rst + offset_val[0]
        round_data = torch.round(add_offset)
        round_data = torch.clamp(round_data, -128, 127)
        expanded_x = round_data.to(torch.int8)

    # 动态量化
    if quant_mode == 1:
        x_final = expanded_x.to(torch.float32)
        if scale is None:
            x_abs = torch.abs(x_final)
            x_max = torch.max(x_abs, dim=-1, keepdim=True)[0]
            expanded_scale = x_max / 127.0
            expanded_x = torch.round(x_final / expanded_scale).to(torch.int8)
        else:
            scale = scale.to(torch.float32)
            if scale.shape[0] == 1:
                x_final = x_final * scale
            else:
                if drop_pad_mode == 0:
                    x_final = x_final * scale[sorted_expert_idx[:vaild_num] - expert_start]
                else:
                    for i in range(sort_row_tmp.shape[0]):
                        val = sort_row_tmp[i].item()
                        if val != -1:
                            x_final[i] = x_final[i] * scale[i // expert_capacity]

            x_abs = torch.abs(x_final)
            x_max = torch.max(x_abs, dim=-1, keepdim=True)[0]
            expanded_scale = x_max / 127.0
            expanded_x = torch.round(x_final / expanded_scale).to(torch.int8)

    if drop_pad_mode == 1:
        expanded_x = expanded_x.masked_fill(expanded_x_mask, 0)
        expanded_x = expanded_x.reshape(expert_num, expert_capacity, h)

    if row_idx_type == 1:
        expanded_row_idx = expanded_row_idx[:vaild_num]

    if drop_pad_mode == 0:
        if expanded_scale is not None:
            expanded_scale = expanded_scale.flatten()[:vaild_num]
        if active_num <= 0:
            active_num = num_rows * k
        else:
            active_num = min(active_num, num_rows * k)
        expanded_x = expanded_x[:vaild_num]
        # 将张量转到 CPU 进行 padding，避免 NPU OOM
        original_device = expanded_x.device
        expanded_x, expanded_row_idx, expanded_scale = post_process_golden_output(
            expanded_x.cpu(),
            expanded_row_idx.cpu(),
            expanded_scale.cpu() if expanded_scale is not None else None,
            h, active_num, num_rows * k)
        expanded_x = expanded_x.to(original_device)
        expanded_row_idx = expanded_row_idx.to(original_device)
        if expanded_scale is not None:
            expanded_scale = expanded_scale.to(original_device)

    if expert_tokens_count is None:
        expert_tokens_count = torch.tensor([], dtype=torch.int64)
    else:
        expert_tokens_count = expert_tokens_count.to(torch.int64)
    if expanded_scale is None:
        # expanded_scale = torch.ones()
        expanded_scale = torch.tensor([], dtype=torch.float32)
    else:
        expanded_scale = expanded_scale.to(torch.float32).reshape(-1)

    expanded_row_idx = expanded_row_idx.to(torch.int32)
    if quant_mode == -1:
        expanded_x = expanded_x.to(x_dtype)
    return expanded_x, expanded_row_idx, expert_tokens_count, expanded_scale  

def post_process_golden_output(expanded_x, expanded_row_idx, expanded_scale, h, active_num, totalLength):
    pad_x = torch.ones((active_num - expanded_x.shape[0], h), dtype=expanded_x.dtype, device=expanded_x.device)
    expanded_x = torch.cat([expanded_x, pad_x], dim=0)
    pad_idx = torch.full((totalLength - expanded_row_idx.shape[0],), -1, dtype=expanded_row_idx.dtype, device=expanded_row_idx.device)
    expanded_row_idx = torch.cat([expanded_row_idx, pad_idx], dim=0)
    if expanded_scale is not None:
        pad_scale = torch.ones((active_num - expanded_scale.shape[0],), dtype=expanded_scale.dtype, device=expanded_scale.device)
        expanded_scale = torch.cat([expanded_scale, pad_scale], dim=0)
    return expanded_x, expanded_row_idx, expanded_scale


def adapter_capacity(sorted_row_idx, sorted_expert_idx, capacity):
    count = 0
    last = sorted_expert_idx[0]
    for i, val in enumerate(sorted_expert_idx):
        if last != val:
            count = 1
            last = val
        else:
            count += 1
            if count > capacity:
                sorted_expert_idx[i] = -1
                sorted_row_idx[i] = -1
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

x = torch.randn(507, 4608, dtype=torch.bfloat16, device="npu")
expert_idx = torch.randint(0, 256, (507, 8), dtype=torch.int32, device="npu")
scale = torch.randn(507, dtype=torch.float32, device="npu")
offset = torch.randn(1, dtype=torch.float32, device="npu")
expanded_x, expanded_row_idx, expert_tokens_count_or_cumsum, expanded_scale = cann_bench.ai_infra_moe_init_routing_v3(x, expert_idx, scale, offset, active_num=0, expert_capacity=32, expert_num=256, drop_pad_mode=0, expert_tokens_num_type=1, expert_tokens_num_flag=True, quant_mode=-1, active_expert_range=[0, 256], row_idx_type=0)
```
