# CMedHalu 核心代码

从原实验代码中提取的最小公开实现，仅发布源码，不包含数据集、临床指南原文、模型权重、个人目录、历史预测和结果表。原始压缩包保持不变。

## 保留内容

| 文件 | 来源与作用 |
| --- | --- |
| construction.py | generate_data.py 中的实体库整理、实体过滤、同类型实体替换、答案错配与 SFT 格式转换 |
| prompts.py | 原 evaluation_hallucination/prompt.py，保留实体与逻辑任务提示词及参考知识版本 |
| infer.py | 新增的通用本地模型入口，替代绑定服务器目录的旧推理脚本 |
| score.py | 原 get_score.py 的四项指标计算，预测列表改用 ast.literal_eval；中文 BLEU 直接调用 SacreBLEU |
| entity_metrics.py | 原实体抽取精确与非精确匹配指标函数 |

## 使用

建议 Python 3.10+，GPU 推理环境另行安装匹配的 PyTorch。依赖未锁定，不代表原论文软件版本。

```bash
pip install -r requirements.txt
python construction.py entities --input data/kg.txt --output data/entity_map.json
python construction.py filter --input data/with_entities.jsonl --output data/filtered.jsonl
python construction.py construct --input data/filtered.jsonl --entity-map data/entity_map.json --output data/constructed.jsonl
python construction.py sft --input data/train.jsonl --output data/train_sft.jsonl
python infer.py --model Qwen/Qwen3-8B --input data/test_entity.jsonl --output results/entity.jsonl --task entity
python infer.py --model Qwen/Qwen3-8B --input data/test_logic.jsonl --output results/logic.jsonl --task logic
python score.py --input results/combined.jsonl --entity-negative-count 750
```

先自行创建 data 和 results 目录。输入数据及医学知识库需合法取得，本次未发布。将两个任务的预测 JSONL 按实体子集、逻辑子集顺序合并为 combined.jsonl；负例顺序必须与该顺序一致。750 是原测试集实体负例数量，其他数据划分必须调整。

构建输入至少需要 question、question_type、file_name、answer、entity_list_in_answer（元素为“实体@@类型”）；实体过滤还需要原始抽取字段，参见函数实现。推理输入需要 question、maybe_wrong_answer；使用 --with-chunk 时需提供 chunk。评分还需要 hallucination、hallucination_type、hallucination_entity_list、hallucination_logic、entity_list_in_answer、answer、prediction 等原实验标注字段。

## 复现边界与已知限制

- 这是原代码的核心子集，不是经过完整验证的论文复现包。没有重新运行模型或复算论文表格。
- 原构建函数只约束 question_type 相同、file_name 不同，未实现修订稿描述的疾病相关性约束；保留原有 40% 逻辑、40% 实体的构建比例。论文最终平衡子集的构建与按知识片段划分流程不在此次公开代码中。
- 原构建代码可能抽到相同替换实体，要求非空实体列表；未添加新的算法以免改变原始实验含义。file_name 必须由调用方提供不含隐私和本机路径的来源标识。
- 原 SFT 转换使用 right_entity_list，而原评分 label4 使用 entity_list_in_answer；两者的纠正目标范围不同，保留此差异供作者核对。
- 原评分要求严格四行输出；逻辑定位使用中文或英文引号，早期构建代码却使用书名号。该格式差异保留，评分前需核对最终标注。
- 新推理入口采用贪心解码，未声称与原各模型采样设置相同；不复用历史预测，不包含 API 客户端或密钥。思考型模型的额外输出可能不符合原四行解析规则。
- 不含训练执行器与训练配置；SFT 输出为 instruction/input/output 格式，可接入用户自己的训练工具。

清理：移除原服务器路径、用户名相关注释、旧实验入口和缓存；所有输入输出及模型位置由命令行传入。保留提示词中的医学示例，它们是任务定义的一部分，不应作为临床建议使用。
