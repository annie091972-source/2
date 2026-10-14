# ExViT：Houston / Trento 全部GT逐类5%实验

这是基于你上传的模型和训练代码整理的替换版本。数据、权重和历史日志不包含在压缩包内。
建议解压到新目录，将自己的dataset目录复制或链接进来；不要覆盖旧实验结果。

## 实验协议

- seeds：21、22、80、443、445。
- Houston：先合并TRLabel与TSLabel；检查无重叠后作为全部GT。
- Trento：使用GT_Trento。
- 每类训练数floor(N×0.05)，其余有标签中心像素测试；背景0不参与训练和精度计算。
- 每个seed重新划分；保存坐标及GT。新版本使用局部default_rng，不逐类重设全局随机数。
- 当前实际数据：Houston训练745/测试14284；Trento训练1508/测试28706。
- 训练测试中心不重叠，但随机划分下13×13邻域可能重叠。本协议不是空间隔离实验。
- 与论文官方划分不同，不应直接将精度差异解释为模型复现错误。

## 安装

Python建议3.9或更新版本。安装与你CUDA/GPU匹配的PyTorch，再执行：

```bash
pip install -r requirements.txt
```

GPU采集使用nvidia-ml-py（import pynvml）。CUDA设备选择是PyTorch逻辑编号；若CUDA_VISIBLE_DEVICES=2，程序里的--gpu应为0。
GPU负载和设备显存通过实际CUDA设备UUID对应NVML，避免选错显卡。
CUDA设备或NVML监控初始化失败会报错，避免日志静默缺失；CPU测试使用--gpu -1，GPU字段为null。

## 数据结构（注意文件名大小写）

```text
dataset/houston2013/
  HSI.mat      -> HSI (349,1905,144)
  LiDAR.mat    -> LiDAR (349,1905)
  TRLabel.mat  -> TRLabel
  TSLabel.mat  -> TSLabel
dataset/Trento/
  HSI_Trento.mat   -> HSI_Trento (166,600,63)
  Lidar_Trento.mat -> Lidar_Trento (166,600)
  GT_Trento.mat    -> GT_Trento (166,600)
```

分卷解压时各卷放同一目录，从.001开始解压。Houston包含001–017，Trento包含001–002。
解压得到的Houston_2013目录可以重命名为houston2013，也可以通过--root指定原路径。

## 先检查再训练

```bash
python run_houston_5percent.py --out log/preflight --preflight
python run_trento_5percent.py --out log/preflight --preflight
```

预检查检查文件keys、维度、有限值、类别、划分和GPU/NVML初始化，并保存划分。
预检查输出与正式实验分开；输出目录已存在时程序拒绝覆盖，需要换--out。

正式训练（默认每个数据集跑全部5个seed，500epochs）：

```bash
python run_houston_5percent.py --out log/allGT_5percent
python run_trento_5percent.py --out log/allGT_5percent
```

也可使用统一入口：

```bash
python run_checked.py --dataset Houston --root dataset/houston2013 --out log/allGT_5percent
python run_checked.py --dataset Trento --root dataset/Trento --out log/allGT_5percent
```

默认select=last：选最后一轮模型；定期测试仅记录结果。若明确沿用原代码按测试OA选择模型，在命令末尾加：

```text
--select test_peak
```

test_peak会利用测试集选择模型，日志明确记录该选择方式。没有另设验证集，也没有从5%训练标签中偷偷扣除验证样本。
只测试流程可用--seeds 21 --epochs 1，正式实验请恢复默认。

## 保留与论文的差异

模型保留作者的3层DW/PW/BatchNorm/GELU、16/32/64通道、共享位置嵌入、4层独立编码加2层联合编码、全部token加权融合。
修复squeeze(-1)，支持batch size=1。dim64、heads4、dim_head16、mlp_dim32、patch13。
保留你上传的训练参数：lr3e-4；Houston weight_decay7.5e-4/dropout0.15/翻转增强；Trento weight_decay1e-4/dropout0.2/不增强。
HS使用全图逐波段min-max（常量波段安全处理）；LiDAR保持原值。全图归一化使用无标签测试区域的统计信息，属于同场景处理协议。
论文/作者Houston示例是lr5e-4、weight_decay0、dropout0.1、无翻转增强，可通过--lr 0.0005 --weight-decay 0 --dropout 0.1 --no-augment显式切换。
StepLR每20epochs乘0.9；不做ImageNet预训练；按需读取patch降低CPU内存。

## 每个seed的输出

```text
selected.pt              所选模型state_dict
last.pt                  最后一轮state_dict
result.mat               GT、预测矩阵、测试mask、训练测试坐标、精度、混淆矩阵
epochs.csv               每轮训练loss/OA/lr/耗时，定期测试OA/AA/Kappa/耗时
gpu_usage.csv            每秒采样的时间/阶段/设备使用率/设备显存
metrics.json             精度、各类准确率、耗时和GPU摘要
gt_map.png
test_pred_map.png
test_error_map.png
labeled_pred_map.png     所有有标签位置的预测，含训练位置；精度仅在测试点计算
```

数据集级别：config.json、split_SEED.mat、summary.json。summary统计跨seed均值及样本标准差ddof=1，逐seed完成后更新。
OA/AA/class_acc单位为百分数，Kappa原值。MAT中坐标和预测向量为0-based；图像标签为1..K，背景0。
pt只存模型权重，config.json保存构造参数；没有断点续训优化器状态。

时间指标：train_time_s累计训练epoch耗时；periodic_test_time_s累计定期评估耗时；selected_single_test_time_s所选模型单次测试耗时；map_inference_time_s制图预测耗时；wall_time_s包括训练到保存制图完成，不包括初始读取数据和构建模型。
CUDA计时前后同步。GPU使用率为设备整体负载，可能包含其他进程；短阶段可能没有采样，字段为null而不是虚构0。peak_torch_allocated_MiB是PyTorch进程分配峰值，与设备显存指标不同。

## 全影像分类图与训练后绘图

训练命令加--full-map会额外预测所有像素（包括无标签区域），生成full_pred_map.png及对应MAT字段。
默认分类图只预测有标签位置，背景黑色。test_pred_map.png仅测试点；训练位置保持黑色。
PNG按原像素分辨率保存，所有seed固定类别颜色；配色是本项目自定义配色，不声称与论文图完全一致。

训练后无需GPU、无需重新训练，可直接重新绘图：

```bash
python draw_houston.py --root log/allGT_5percent/Houston
python draw_trento.py --root log/allGT_5percent/Trento
python run_checked.py --draw log/allGT_5percent/Houston/seed_21/result.mat
```

只绘图需安装CPU或CUDA版PyTorch，因为模块会导入模型，但不会运行GPU训练。

## 提取汇总和各类精度

```bash
python extract_results.py --root log/allGT_5percent
```

生成根目录results.csv，以及每个seed的class_accuracy.csv和confusion_matrix_row_normalized.csv。
extract_perclass_acc.py、extract_cm_norm.py是同一提取工具的兼容入口，参数相同。

gen_trento_split.py可独立生成5个seed的5%划分；训练会通过同一split函数确定性地生成相同划分，无需先运行它。

## 验证范围

已检查两套实际数据全部5个seed划分、真实13×13 patch前向及batch=1；小规模CPU训练、权重重载、MAT/PNG输出、多seed汇总、test_peak、full-map和离线绘图已通过。
未执行完整500epochs×10次训练；当前验证环境是CPU，NVML/CUDA采样尚需在你的GPU上运行preflight验证。

原论文引用：Jing Yao et al., Extended Vision Transformer (ExViT) for Land Use and Land Cover Classification: A Multimodal Deep Learning Framework, IEEE TGRS, 2023, DOI:10.1109/TGRS.2023.3284671。
