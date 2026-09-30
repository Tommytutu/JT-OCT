"""Build the comparison from saved measurements, excluding the adaptive F1 pilot."""
import json
from pathlib import Path
from statistics import median

ROOT=Path(__file__).resolve().parent
REPORT=ROOT/'reports'


def main():
    main_runs=json.loads((REPORT/'native_accuracy_20260930/runs.json').read_text(encoding='utf-8'))
    runs=json.loads((REPORT/'native_accuracy_final_20260930/runs.json').read_text(encoding='utf-8'))
    historic=json.loads((REPORT/'historical_fico_reference.json').read_text(encoding='utf-8'))
    rows=[];profiles=[]
    for depth in (4,5):
        for penalty in (0.,.01):
            selected={m:[r for r in runs if r['model']==m and r['depth']==depth and r['penalty']==penalty]
                      for m in ('native','cc_f1','f1')}
            for m in ('native','cc_f1','f1'):
                if len(selected[m])!=2:raise AssertionError(f'Missing repetitions: {depth}, {penalty}, {m}')
            times={m:median(r['process_seconds'] for r in records) for m,records in selected.items()}
            native_call=median(r['call_seconds'] for r in selected['native'])
            old=next(r for r in historic if int(r['case'].split('_D')[1].split('_')[0])==depth and float(r['case'].split('_L')[1])==penalty)
            rows.append(dict(depth=depth,penalty=penalty,native_process=times['native'],native_call=native_call,
                             native_range=[min(r['process_seconds'] for r in selected['native']),max(r['process_seconds'] for r in selected['native'])],
                             historical_call=old['call'],historical_speedup=old['call']/native_call,
                             f1_process=times['f1'],cc_f1_process=times['cc_f1'],f1_speedup=times['f1']/times['native'],
                             accuracy_objective=selected['native'][0]['UB']))
            if depth==5:
                profiles.append(dict(penalty=penalty,old=old,new={k:median(r[k] for r in selected['native']) for k in
                    ('d3_calls','d3_seconds','gpu_compute_and_join_seconds','metadata_seconds','pack_seconds','recovery_seconds','rmp_seconds','rmp_solves')}))
    (REPORT/'native_accuracy_summary.json').write_text(json.dumps(dict(comparisons=rows,profiles=profiles),indent=2),encoding='utf-8')
    lines=['# FICO：JT-CG 原生准确率定价加速验证','',
        '本次实现保留 JT-CG 的收缩模型、完整祖先分隔集、限制主问题和最优性证明，移植 General JT 的原生线性 D3 求值服务。目标仍是误分类率 + λK。General JT/F1 原目录的算法文件和动态库未修改，核对记录见 f1_unchanged_verification.json；变更仅作用于 JT-OCT 内的独立副本。', '',
        '## 同环境完整进程比较','',
        '全部使用 FICO 10,459 × 159 的同一文件、同一 Python 环境、RTX 4080 SUPER、8 个 CPU 工作线程，独立进程顺序执行。下表为两次运行的中位数，包含启动、读数据、求解、写结果和退出。通用 F1 使用历史实验的 budget 策略，CC-F1 使用 native 引擎。所有四组准确率结果均 OPT，LB=UB；F1/CC-F1 均在 1e-7 容差内证明最优。','',
        '| 深度 | λ | 加速 JT-CG / 秒 | 通用 F1 / 秒 | CC-F1 / 秒 | 相对通用 F1 加速 |','|---|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| D{r['depth']} | {r['penalty']:g} | {r['native_process']:.3f} | {r['f1_process']:.3f} | {r['cc_f1_process']:.3f} | {r['f1_speedup']:.2f}× |")
    lines+=['','只有两次重复，因此这些是描述性结果，不是显著性检验。与 CC-F1 的比较须逐项看，不能把“超过通用 F1”写成“全面超过 CC-F1”。准确率和 F1 的目标函数不同，不能比较二者目标值大小。', '',
        '## 与用户给出的原版结果比较','',
        '此表两边统一使用求解调用时间。原版来自 2026-09-19 已保存诊断，与新版本不是同一时段；本次同时重跑的原版记录另存于原始 CSV。原版首次 D4、λ=0 明显含首次初始化/编译开销，不用于夸大加速比。','',
        '| 深度 | λ | 原版调用 / 秒 | 加速版调用 / 秒 | 加速 | 相同最优值 |','|---|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| D{r['depth']} | {r['penalty']:g} | {r['historical_call']:.3f} | {r['native_call']:.3f} | {r['historical_speedup']:.2f}× | {r['accuracy_objective']:.12f} |")
    lines+=['','## 瓶颈定位','',
        '原版 D5 的 D3 条件子树服务占总求解时间约 92%–94%；其中 GPU 成本计算约占整个求解的 82%–85%。主问题只占约 3%–5%。因此只调 Gurobi 或换 Python 启动顺序无法消除主要开销。D5 相比 D4 多了一层上部前缀，候选 D3 行集合由约 318 个增至约 46,359 个，原版实际调用约 3.8–4.4 万次。', '',
        'λ=0.01 尤其明显：最优树只有两次切分、3,109 个误分类，D4 和 D5 的最优值都为 0.317255951812。初始化很早就找到这棵树；随后绝大部分时间是在证明不存在更好的深树。因此，得到好解很快和证明最优很慢可以同时发生。', '',
        '| λ | 阶段 | 原版 / 秒 | 新版 / 秒 |','|---:|---|---:|---:|']
    for r in profiles:
        for label,oldkey,newkey in [('D3 服务总计','d3_seconds','d3_seconds'),('GPU 计算与归约','gpu_compute_and_join_seconds','gpu_compute_and_join_seconds'),('准备与原生控制开销','metadata_seconds','metadata_seconds'),('限制主问题','rmp_seconds','rmp_seconds')]:
            lines.append(f"| {r['penalty']:g} | {label} | {r['old'][oldkey]:.3f} | {r['new'][newkey]:.3f} |")
    lines+=['','| λ | 原版 D3 请求数 | 新版 D3 请求数 | 原版主问题求解次数 | 新版主问题求解次数 |','|---:|---:|---:|---:|---:|']
    for r in profiles:
        lines.append(f"| {r['penalty']:g} | {r['old']['d3_calls']} | {r['new']['d3_calls']:g} | {r['old']['rmp_solves']} | {r['new']['rmp_solves']:g} |")
    lines+=['','GPU 计时说明：新内核去掉一次末尾冗余同步，执行等待转入 join。表中合并 kernel-launch 与 join 时间；不能用单独的 launch 时间冒充 GPU 求值总时间。后台预处理与求解重叠，不另行相加。metadata 列含 native 内的准备、等待和校验等未归入 GPU 阶段的时间。', '',
        '## 为什么非线性 F1 能更快','',
        'F1 = 2TP/(P+N+TP−TN)。固定阈值 θ 时，F1 ≥ θ 等价于 (2−θ)TP + θTN ≥ θ(P+N)。固定分裂预算后，也可以围绕带权计数线性子问题产生支撑界。因此 General JT 没有把整个问题交给通用非线性黑箱；它重用少量线性支撑求解和大量与权重无关的几何数据。目标表达式是非线性的，不代表这套实现的每次定价更贵。', '',
        '本次采用的策略：', '',
        '1. C++ 常驻求值器，准确率对应 a=b=1；保留整数计数、紧凑转置位集、等价/互补特征代表及对称特征对求值。',
        '2. 全部待求 D3 行集合启动一次并行后台准备，重用几何缓存；不再每 64 个请求创建一次完整准备批次。',
        '3. 高维二分类 D5 使用 256 个请求一批、最多八批后重新求主问题。旧对偶继续通过原有修复界认证，所有未解决签名仍在完整域中。',
        '4. 省去未使用的条件根成本下载和冗余同步；合并 Python 树恢复与独立校验，避免每个节点重复构造全部特征列表。',
        '5. 把不依赖 SciPy/Gurobi Python 模型的公共函数移入轻量模块，减少 D4 短任务的启动成本；函数 AST 与原版逐一核对相同。', '',
        '相似子问题下界转移也做过试验：约增加 14 秒，未有效减少查询，没有启用。准确率专用编译展开也未测得稳定收益，没有保留。D2 新模块被一并移植并做穷举校验，但本次公开的 D4/D5 加速使用 D3 尾部，不能把全部收益归因于 D2。', '',
        '## 正确性与适用范围','',
        '独立穷举对照覆盖原始/路由子问题、D2/D3、重复与互补特征、常量列、冲突样本、非标准二元标签、均匀权重缩放、叶节点样本量约束；另检查 D4/D5 完整 CG 的 LB/UB、超时可行解、空缓存和旧 CPU 路径。每棵 native 返回的树均重新路由核验。', '',
        '本次性能结论只覆盖 FICO 的上述四个配置。默认入口仍可用 legacy；native 模式限定二分类、均匀权重、共享特征/代价、STOP 和 D4/D5 的 D3 尾部。min_leaf=0 与 F1 的 1 在这里最优值等价：非负代价下可去除通往空分支的切分。原生缓存预算最多 3 GiB，可能比原版多用内存。', '',
        '## 可复核文件','',
        '- `native_accuracy_20260930/results.csv`：原版、native、CC-F1 及默认 adaptive F1 的探索性记录；最终 F1 对照不使用 adaptive 行。该探索批次在原版 D5 λ=0.01 的冗余复跑期间终止，没有把未完成项列为结果。',
        '- `native_accuracy_final_20260930/results.csv`：统一复测 native、CC-F1 和匹配历史 budget 策略的通用 F1，各两次。',
        '- 两个目录的 `runs.json`、`manifest.json` 和 `raw/`：实际命令、文件哈希、输出、日志及可行树。',
        '- `historical_fico_reference.json`：用户给出原版结果的源文件和阶段计时。',
        '- `../native/general_jt/provenance.json`：借鉴源码与动态库的 SHA256。',
        '- `../README_NATIVE_ACCURACY.md`：运行、编译和验证方法。','']
    (REPORT/'NATIVE_ACCURACY_FINDINGS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(rows,ensure_ascii=True,indent=2))


if __name__=='__main__':main()
