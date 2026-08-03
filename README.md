# 多课程 RAG 知识库

面向课程课件的检索增强问答（RAG）应用。基于 Streamlit 提供中文界面，支持多课程管理、课件向量化入库，以及基于阿里云 DashScope（Embedding + Qwen）的课程问答。

## 功能概览

- **课程管理**：创建、查看、维护多门课程
- **课件入库**：上传或导入课件，解析后分块并写入 FAISS 向量库
- **课程问答**：按课程检索相关片段，调用大模型生成带依据的回答
- **Parent Retrieval**：可选扩展为同页 / 同幻灯片全文再回答

支持格式：`.pptx`、`.docx`、`.pdf`、`.md`、`.txt`

## 技术栈

| 组件 | 说明 |
|------|------|
| UI | Streamlit |
| 向量库 | FAISS |
| Embedding | DashScope `qwen3.7-text-embedding`（可配置） |
| LLM | DashScope `qwen3.7-max`（可配置） |

## 目录结构

```
dominican-agent/
├── app.py                 # Streamlit 入口与导航
├── run_app.sh             # 启动脚本
├── requirements.txt
├── .env.example           # 环境变量模板
├── views/                 # 页面：首页 / 课程管理 / 课件入库 / 课程问答
├── src/                   # 解析、分块、向量化、检索、问答等核心逻辑
└── data/courses/          # 课程数据与向量索引（默认不入库）
```

## 环境要求

- Python 3.10+（建议）
- 阿里云 [DashScope](https://dashscope.console.aliyun.com/) API Key

## 快速开始

### 1. 克隆仓库

```bash
git clone <你的仓库地址>
cd dominican-agent
```

### 2. 创建虚拟环境并安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入 API Key：

```env
DASHSCOPE_API_KEY=your_api_key_here

# 可选覆盖
# EMBEDDING_MODEL=qwen3.7-text-embedding
# LLM_MODEL=qwen3.7-max
# EMBEDDING_BATCH_SIZE=10
```

### 4. 启动应用

```bash
./run_app.sh
# 或
streamlit run app.py
```

浏览器打开终端提示的本地地址（一般为 `http://localhost:8501`）。

## 使用流程

1. **课程管理**：新建课程  
2. **课件入库**：上传课件或使用「样例一键导入」，完成向量化  
3. **课程问答**：选择课程，输入问题，生成答案  

## 数据说明

- 课程元数据与向量索引保存在 `data/courses/`
- 该目录下各课程子目录默认被 `.gitignore` 忽略，不会提交到 Git
- 请勿将 `.env` 提交到仓库

## 许可证

按项目需要自行补充。
