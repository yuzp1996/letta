# 自定义Anthropic Base URL解决方案

## 🚨 问题描述

你配置了Anthropic API密钥，但对应的base URL需要指向 `https://api.sydney-ai.com/v1` 而不是默认的 `https://api.anthropic.com/v1`。

## ✅ 已完成的代码修改

我已经修改了以下文件来支持自定义Anthropic base URL：

### 1. `letta/settings.py`
- 添加了 `anthropic_base_url` 配置项
- 默认值为 `https://api.anthropic.com/v1`
- 可通过环境变量 `ANTHROPIC_BASE_URL` 覆盖

### 2. `letta/llm_api/anthropic_client.py`
- 修改了 `_get_anthropic_client()` 方法
- 修改了 `_get_anthropic_client_async()` 方法
- 修改了 `count_tokens()` 方法
- 所有Anthropic客户端初始化都使用自定义base URL

### 3. `letta/schemas/providers/anthropic.py`
- 修改了 `list_llm_models_async()` 方法
- 确保模型列表获取也使用自定义base URL

### 4. `.env` 文件
- 修复了格式问题（移除了引号和注释导致的解析错误）
- 添加了 `ANTHROPIC_BASE_URL=https://api.sydney-ai.com/v1`

## 📋 错误分析

### 错误现象
- ✅ 服务器启动成功
- ✅ ADE界面能连接
- ✅ 能获取智能体基本信息
- ❌ 计算上下文窗口时失败

### 错误原因
系统尝试使用Anthropic Claude模型计算token数量，但缺少`ANTHROPIC_API_KEY`环境变量。

## 🛠️ 立即解决方案

### 方案1：设置OpenAI API密钥（推荐）

```bash
# 停止当前服务器（Ctrl+C）
# 然后设置OpenAI API密钥
export OPENAI_API_KEY="your_openai_api_key"

# 重新启动服务器
letta server --port=8283
```

### 方案2：设置Anthropic API密钥

```bash
# 停止当前服务器（Ctrl+C）
# 设置Anthropic API密钥
export ANTHROPIC_API_KEY="your_anthropic_api_key"

# 重新启动服务器
letta server --port=8283
```

### 方案3：同时设置两个API密钥（最佳）

```bash
# 停止当前服务器（Ctrl+C）
export OPENAI_API_KEY="your_openai_api_key"
export ANTHROPIC_API_KEY="your_anthropic_api_key"

# 重新启动服务器
letta server --port=8283
```

## 🔧 检查智能体配置

你的智能体ID是：`agent-79f42777-79b4-4d96-a385-01792c326df6`

可能这个智能体配置使用了Anthropic模型，所以需要Anthropic API密钥。

### 检查智能体模型配置

```bash
# 服务器运行后，检查智能体配置
curl http://localhost:8283/v1/agents/agent-79f42777-79b4-4d96-a385-01792c326df6
```

### 修改智能体模型（如果需要）

如果智能体使用的是Anthropic模型，你可以：

1. **通过ADE界面修改**：
   - 访问 https://app.letta.com
   - 选择智能体
   - 修改模型配置为OpenAI

2. **通过API修改**：
```bash
curl -X PATCH http://localhost:8283/v1/agents/agent-79f42777-79b4-4d96-a385-01792c326df6 \
  -H "Content-Type: application/json" \
  -d '{
    "llm_config": {
      "model": "openai/gpt-4o-mini",
      "model_endpoint_type": "openai",
      "model_endpoint": "https://api.openai.com/v1"
    }
  }'
```

## 🔍 .env文件问题

你的启动日志显示.env文件解析警告：
```
python-dotenv could not parse statement starting at line 2
python-dotenv could not parse statement starting at line 3
...
```

### 检查.env文件

```bash
# 查看.env文件内容
cat .env
```

### 修复.env文件格式

确保.env文件格式正确：
```env
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
OLLAMA_BASE_URL=http://localhost:11434
```

**注意**：
- 不要使用引号（除非值中包含空格）
- 不要在行末添加分号
- 避免空行和注释在变量行中

## 🚀 推荐操作步骤

1. **停止当前服务器**：在终端按 `Ctrl+C`

2. **设置环境变量**：
```bash
export OPENAI_API_KEY="your_openai_api_key"
```

3. **重新启动服务器**：
```bash
letta server --port=8283
```

4. **测试ADE界面**：
   - 访问 https://app.letta.com
   - 连接到本地服务器
   - 尝试查看智能体信息

## 🔍 调试技巧

### 启用详细日志
```bash
export LETTA_LOG_LEVEL=DEBUG
letta server --port=8283
```

### 检查API密钥
```bash
echo "OpenAI: ${OPENAI_API_KEY:0:10}..."
echo "Anthropic: ${ANTHROPIC_API_KEY:0:10}..."
```

### 测试API连接
```python
# test_api.py
import os
import openai

try:
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    models = client.models.list()
    print("✅ OpenAI API连接成功")
except Exception as e:
    print(f"❌ OpenAI API连接失败: {e}")
```

## 📞 如果问题持续

1. **检查智能体模型配置**
2. **确保API密钥有效**
3. **尝试创建新的智能体**
4. **查看完整的服务器日志**

## 🆕 新问题：count_tokens API不支持

### 问题描述
```
anthropic.NotFoundError: Error code: 404 - {'error': {'message': 'Invalid URL (POST /v1/messages/count_tokens)', 'type': 'invalid_request_error', 'code': ''}}
```

### 原因分析
sydney-ai.com服务器不支持Anthropic的`count_tokens` API端点。这是因为：
1. sydney-ai.com可能是OpenAI兼容的API服务
2. 不完全支持Anthropic的所有专有API端点
3. `/v1/messages/count_tokens`是Anthropic特有的功能

### 解决方案
我已经修改了`anthropic_client.py`中的`count_tokens`方法：
- 添加了错误处理和优雅降级
- 当API不支持时，使用粗略的token估算
- 确保系统能继续正常工作

### 修改内容
- ✅ 添加了`NotFoundError`异常处理
- ✅ 实现了`_estimate_tokens`备用方法
- ✅ 使用字符数/4的粗略估算
- ✅ 添加了详细的警告日志

现在重启服务器后，即使sydney-ai.com不支持count_tokens API，系统也能正常工作。

---

*这些修改确保了与非标准Anthropic API服务的兼容性*
