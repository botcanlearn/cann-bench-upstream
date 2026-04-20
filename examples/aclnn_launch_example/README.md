# ACLNN Launch Example

基于 registry-invoke 模式的 ACLNN 自定义算子示例工程，支持算子自注册机制和Python API接口。

## 目录结构

```
aclnn_launch_example/
├── cann_bench/         # Python包
├── cmake/              # 公共CMake配置(不感知算子)
│   ├── func.cmake      # 注册宏定义
│   └── ...
├── csrc/
│   ├── extension.cpp   # Python扩展入口
│   └── ops/            # 算子目录
│       ├── CMakeLists.txt  # 自动发现算子
│       ├── add/            # Add算子
│       │   ├── CMakeLists.txt  # 算子自注册
│       │   ├── op_host/
│       │   │   ├── add_def.cpp      # 算子定义
│       │   │   ├── add_infershape.cpp # Shape推导
│       │   │   └── add_tiling.cpp   # Tiling实现
│       │   ├── op_kernel/
│       │   │   ├── add.cpp          # Kernel入口
│       │   │   ├── add.h            # Kernel头文件
│       │   │   ├── add_tiling_data.h # Tiling数据结构
│       │   │   └── add_tiling_key.h  # Tiling Key定义
│       │   ├── op_api/
│       │   │   ├── aclnn_add.cpp    # L2 API
│       │   │   └ add.cpp            # L0 API
│       │   └── op_plugin/
│       │       └ add_plugin.cpp     # Python bindings
│       └── sqrt/           # Sqrt算子(结构相同)
├── dist/               # 输出目录
│   ├── cann_bench_linux_aarch64.run   # run包
│   └── cann_bench-1.0.0-*.whl         # wheel包
├── scripts/            # 打包脚本
│   ├── build_run.sh
│   └── build_wheel.sh
├── tests/
├── build.sh            # 统一构建入口
├── build_and_test.sh   # 端到端构建测试脚本
└and setup.py
```

## 构建方法

### 端到端构建测试

```bash
bash build_and_test.sh
```

此脚本会自动完成：
1. 构建 run package 和 wheel package
2. 安装 run package 到 CANN opp 目录
3. 安装 wheel package
4. 设置 ASCEND_CUSTOM_OPP_PATH 环境变量
5. 运行测试验证

### 单独构建

```bash
bash build.sh --soc=ascend910b
```

输出到 `dist/` 目录:
- `cann_bench_linux_aarch64.run` - ACLNN算子安装包（架构根据环境自动获取）
- `cann_bench-1.0.0-cp38-abi3-linux_aarch64.whl` - Python wheel包

---

## 新增算子详细步骤

### 第一步：创建算子目录结构

```bash
# 以新增 Mul 算子为例
cd csrc/ops
mkdir -p mul/op_host
mkdir -p mul/op_kernel
mkdir -p mul/op_api
mkdir -p mul/op_plugin
```

**目录说明：**
| 目录 | 用途 | 编译器 |
|------|------|--------|
| `op_host/` | 算子定义、Shape推导、Tiling | bisheng |
| `op_kernel/` | Kernel源文件和头文件 | bisheng |
| `op_api/` | ACLNN L2/L0 API | bisheng |
| `op_plugin/` | Python bindings (torch.library) | g++ |

### 第二步：编写算子源文件

#### 2.1 op_host目录

**op_host/mul_def.cpp** - 算子定义：
```cpp
#include "register/op_def_registry.h"

namespace ops {
class Mul : public OpDef {
public:
    explicit Mul(const char* name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicShapeSupportFlag(true)
            .ExtendCfgInfo("opFile.value", "mul");
        this->AICore().AddConfig("ascend910b", aicoreConfig);
    }
};
OP_ADD(Mul);
}
```

**op_host/mul_tiling.cpp** - Tiling实现：
```cpp
#include "register/op_def_registry.h"
#include "../op_kernel/mul_tiling_data.h"
#include "../op_kernel/mul_tiling_key.h"

namespace optiling {
IMPL_OP_OPTILING(Mul).Tiling(MulTilingFunc).TilingParse<MulCompileInfo>(TilingParseForMul);
}
```

#### 2.2 op_kernel目录

**mul.cpp** - Kernel入口：
```cpp
#include "mul.h"

template <typename D_T_X, int BUFFER_MODE>
__global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tilingData, tiling);
    NsMul::Mul<D_T_X, BUFFER_MODE> op;
    op.Init(x, y, z, &tilingData);
    op.Process();
}
```

**mul_tiling_data.h** - Tiling数据结构：
```cpp
struct MulTilingData {
    int64_t totalNum;
    int64_t blockFactor;
    int64_t ubFactor;
};
```

**mul_tiling_key.h** - Tiling Key定义：
```cpp
#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(Mul,
    ASCENDC_TPL_DATATYPE_DECL(D_T_X, C_DT_FLOAT, C_DT_FLOAT16, ASCENDC_TPL_INPUT(0)),
    ASCENDC_TPL_UINT_DECL(BUFFER_MODE, 8, ASCENDC_TPL_UI_LIST, 0, 1)
);
// ASCENDC_TPL_SEL definitions...
```

**mul.h** - Kernel头文件：
```cpp
#include "kernel_operator.h"
#include "mul_tiling_data.h"
#include "mul_tiling_key.h"

namespace NsMul {
template <typename T, int BUFFER_MODE>
class Mul {
    // Kernel实现
};
}
```

#### 2.3 op_api目录

**op_api/aclnn_mul.cpp** - L2 API：
```cpp
#include "aclnn_mul.h"
// aclnnMulGetWorkspaceSize, aclnnMul实现
```

**op_api/mul.cpp** - L0 API：
```cpp
#include "mul.h"
namespace l0op {
const aclTensor* Mul(const aclTensor* x1, const aclTensor* x2, aclOpExecutor* executor);
}
```

#### 2.4 op_plugin目录

**op_plugin/mul_plugin.cpp** - Python bindings：
```cpp
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "../op_api/aclnn_mul.h"

namespace cann_bench {

static torch::Tensor mul_meta(const torch::Tensor& x, const torch::Tensor& y) {
    return torch::empty_like(x);
}

static torch::Tensor mul_npu(const torch::Tensor& x, const torch::Tensor& y) {
    auto z = mul_meta(x, y);
    at_npu::native::OpCommand cmd;
    cmd.Name("Mul")
       .Input(x)
       .Input(y)
       .Output(z)
       .Run();
    return z;
}

TORCH_LIBRARY_FRAGMENT(cann_bench, m) {
    m.def("mul(Tensor x, Tensor y) -> Tensor");
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m) {
    m.impl("mul", mul_meta);
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m) {
    m.impl("mul", mul_npu);
}

}
```

### 第三步：编写算子CMakeLists.txt

**csrc/ops/mul/CMakeLists.txt**：
```cmake
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# Mul算子自注册

# Host源文件
set(MUL_HOST_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host/mul_def.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host/mul_infershape.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host/mul_tiling.cpp
)

# API源文件
set(MUL_API_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_api/aclnn_mul.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/op_api/mul.cpp
)

# 注册ACLNN算子
register_aclnn_op(
    Mul                      # 算子类型名
    "${MUL_HOST_SRCS}"       # Host源文件列表
    "${MUL_API_SRCS}"        # API源文件列表
    mul/op_kernel            # Kernel目录(相对于csrc/ops/)
    mul.cpp                  # Kernel入口文件名
    op_kernel                # Tiling include目录(相对于算子目录)
    op_api                   # API include目录
)

# 注册Python插件(可选)
set(MUL_PLUGIN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/mul_plugin.cpp
)
register_aclnn_plugin("${MUL_PLUGIN_SRCS}" op_api)
```

### 第四步：重新构建

```bash
bash build_and_test.sh
```

**无需修改任何公共CMakeLists.txt文件！**

---

## 算子自注册机制原理

### 注册宏定义 (cmake/func.cmake)

```cmake
# 注册ACLNN算子到全局列表
macro(register_aclnn_op OP_TYPE HOST_SRCS API_SRCS KERNEL_DIR KERNEL_FILE TILING_INCLUDE_DIR API_INCLUDE_DIR)
    # 将源文件添加到全局变量 ALL_HOST_OPS_SRCS, ALL_API_OPS_SRCS
    # 将kernel信息添加到 ALL_KERNEL_OPS_INFO
    # 将include目录添加到全局列表
endmacro()

# 注册Python插件
macro(register_aclnn_plugin PLUGIN_SRCS PLUGIN_INCLUDE_DIR)
    # 将插件源文件添加到 ALL_PLUGIN_SRCS
endmacro()
```

### 自动发现算子 (csrc/ops/CMakeLists.txt)

```cmake
# 遍历所有子目录
file(GLOB SUB_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(SUB_DIR ${SUB_DIRS})
    if(IS_DIRECTORY ${SUB_DIR})
        get_filename_component(DIR_NAME ${SUB_DIR} NAME)
        add_subdirectory(${SUB_DIR})  # 调用算子的CMakeLists.txt
    endif()
endforeach()
```

### 公共CMakeLists.txt不感知算子

顶层 `CMakeLists.txt` 仅使用全局变量：
```cmake
# 使用注册的源文件构建
npu_op_code_gen(SRC ${ALL_HOST_OPS_SRCS} ...)
npu_op_library(cust_optiling TILING ${ALL_HOST_OPS_SRCS})
npu_op_library(cust_opapi ACLNN ${ALL_API_OPS_SRCS})
foreach(KERNEL_INFO ${ALL_KERNEL_OPS_INFO})
    npu_op_kernel_sources(all_kernels ...)  # 为每个kernel生成ini
endforeach()
```

---

## 安装使用

### run包安装（build_and_test.sh已自动安装）
```bash
./dist/cann_bench_*.run --install-path=${ASCEND_HOME_PATH}/opp --quiet
source ${ASCEND_HOME_PATH}/opp/vendors/custom_ops/bin/set_env.bash
```

### wheel包安装（build_and_test.sh已自动安装）
```bash
pip install --force-reinstall --no-deps dist/cann_bench-*.whl
```

### Python API
```python
import cann_bench

# 设置环境变量（如果未通过set_env.bash设置）
import os
os.environ['ASCEND_CUSTOM_OPP_PATH'] = '/home/xxx/Ascend/cann/opp/vendors/custom_ops'

# 调用算子
z = cann_bench.add(x, y)          # Add算子
r = cann_bench.sqrt(x)            # Sqrt算子
m = cann_bench.mul(x, y)          # 新增Mul算子

# 或通过torch.ops调用
z = torch.ops.cann_bench.add(x, y)
```

---

## 测试说明

测试用例使用 CPU 计算期望值，避免 NPU 内置算子受 `ASCEND_CUSTOM_OPP_PATH` 环境变量影响：

```python
# 正确方式：使用CPU计算expected
z = cann_bench.add(x, y)
expected = x.cpu() + y.cpu()
assert torch.allclose(z.cpu(), expected)
```