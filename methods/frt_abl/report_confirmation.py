"""Render the frozen confirmation results; optionally update until jobs finish."""
import argparse
import json
from pathlib import Path
import time

from .confirmation_campaign import ROOT, read


def percentage(value):
    return "—" if value is None else f"{100 * value:.2f}%"


def interval(row):
    ci = row.get("ci95_t")
    return "—" if ci is None else f"[{100*ci[0]:.2f}, {100*ci[1]:.2f}]"


def render():
    state = read(ROOT / "status.json")
    rows = state.get("completed", [])
    done = sum(row["status"] == "COMPLETED" for row in rows)
    failed = sum(row["status"] != "COMPLETED" for row in rows)
    lines = ["# 三类实验：协议修正与确认结果", "",
             f"状态：**{state['status']}**；预算/参数训练完成 {done}/18，失败 {failed}。", "",
             "种子口径：参数与预算确认 seeds 0–2（n=3）；泛化确认 seeds 0–3（n=4）。"
             "seed4 用于先前筛选/探索，全部排除出确认区间。", "",
             "训练和预测数值如实保留。三种子区间只描述训练随机性的估计，不保证改善。", ""]
    summary_path = ROOT / "training_summary.json"
    if summary_path.exists():
        summary = read(summary_path)
        lines += ["## 参数候选确认（validation）", "",
                  "| 设置 | n | 准确率均值 | seed 标准差 | 均值 95% t 区间（%） |", "",
                  "|---|---:|---:|---:|---:|"]
        # Remove the blank line between a table header and delimiter.
        lines.pop(-2)
        for key in ("sensitivity/baseline", "sensitivity/graph_sigma=0.25",
                    "sensitivity/full_boundary_weight=3x"):
            row = summary["groups"].get(key)
            if row:
                lines.append(f"| {key.split('/',1)[1]} | {row['n']} | {percentage(row['mean'])} | "
                             f"{percentage(row['std'])} | {interval(row)} |")
        lines += ["", "候选均只改变一个参数，两个候选不组合。旧 runs-v10 基线训练代码哈希不同，"
                  "本次使用新训练的同环境 baseline 配对。", "",
                  "| 候选相对 baseline | 配对 n | 均值差（百分点） | 差值 95% t 区间（百分点） |",
                  "|---|---:|---:|---:|"]
        for arm, row in summary["sensitivity_paired"].items():
            delta = "—" if row["mean"] is None else f"{100*row['mean']:+.2f}"
            lines.append(f"| {arm} | {row['n']} | {delta} | {interval(row)} |")
        lines += ["", "两个候选比较没有作多重比较校正，不能将上述区间解释为已校正的显著性检验。"
                  "不支持改善时保留完整结果并放弃参数提升主张；不追加确认集筛选。", "",
                  "## 固定 weak 池的预算确认（validation）", "",
                  "旧 v1 预算曲线存在 weak 池、角色迁移与更新步数混杂，停止其扩种子。"
                  "v2 保持 weak 2,097 条、validation 1,000 条及其 identities/hash，"
                  "各档预热 20 步、每轮 66 步、总计 3,320 步。"
                  "低预算 unused reservoir 不进入 weak。标注角色内部嵌套，"
                  "anchor 复用频次和校准样本量仍随预算改变。", "",
                  "| 名义预算 | n | validation 均值 | seed 标准差 | 均值 95% t 区间（%） |",
                  "|---|---:|---:|---:|---:|"]
        for arm in ("1pct", "3pct", "10pct"):
            row = summary["groups"].get(f"budget/{arm}")
            if row:
                lines.append(f"| {arm} | {row['n']} | {percentage(row['mean'])} | "
                             f"{percentage(row['std'])} | {interval(row)} |")
        lines += ["", "1% gate 样本少于 50，继续使用原始 fail-closed 规则；不降低阈值。"
                  "预算实验不读取 formal test。旧 v1 与新 v2 不混合汇总。", ""]
    general_path = ROOT / "generalization_summary.json"
    lines += ["## 长度与组合泛化", "",
              "重要口径修正：native abduction 使用真实算式结果，因此 native task accuracy "
              "主要反映约束搜索覆盖，不能用作无答案预测泛化成绩。主指标使用 argmax（不读取真实结果），"
              "native symbol/sequence 只报告为已知结果条件下的符号恢复能力。", "",
              "探索 seed4：长度7 argmax 准确率 62.17%；组合划分长度7 为 55.17%，"
              "长度5未见窗口子组为 69.09%（110 条）。这些值不是多种子均值。", ""]
    if general_path.exists():
        general = read(general_path)
        lines += ["| 确认指标 | n | 均值 | seed 标准差 | 95% t 区间（%） |",
                  "|---|---:|---:|---:|---:|"]
        keys = ("length/length7/argmax_task_accuracy", "length/length7/argmax_symbol_accuracy",
                "length/length7/native_sequence_accuracy", "combo/5|True/argmax_task_accuracy",
                "combo/5|False/argmax_task_accuracy", "combo/7|True/argmax_task_accuracy",
                "combo/7|False/argmax_task_accuracy", "combo/length7/argmax_task_accuracy")
        for key in keys:
            row = general["summary"][key]
            lines.append(f"| {key} | {row['n']} | {percentage(row['mean'])} | "
                         f"{percentage(row['std'])} | {interval(row)} |")
        lines += ["", "组合主 OOD 组：长度5且包含保留窗口，共 110 条；长度7单独列出，"
                  "避免将组合与长度偏移混成一个结论。", ""]
    else:
        lines += ["确认组训练/评估尚未全部完成，暂不填入多种子测试均值。", ""]
    lines += ["扩种子决定参考过 seed4 test；虽然 seed4 被排除出区间，仍应披露测试集已探索。"
              "同一固定测试集跨训练 seed 重复评价，区间只描述训练种子的波动，不能把样本数当作 seed 数。"
              "本次没有同划分的外部基线或 anchor-only 对照，因此绝对泛化成绩不能独立证明 FRT 模块优势。", "",
              "## 产物", "", "- 冻结矩阵与训练源哈希：`plan.json`。",
              "- 当前状态：`status.json`；逐任务日志：`logs/`。",
              "- validation 汇总与逐 seed 路径：`training_summary.json`。",
              "- 泛化 checkpoint 冻结：`generalization_checkpoint_freeze.json`。",
              "- 泛化分组确认结果：`generalization_summary.json`。", ""]
    (ROOT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    return state["status"], done, failed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    previous = None
    while True:
        current = render()
        if current != previous:
            print(json.dumps(dict(status=current[0], completed=current[1], failed=current[2],
                                  report=str(ROOT / "RESULTS.md"))), flush=True)
            previous = current
        if not args.watch or current[0] in ("COMPLETED", "FAILED"):
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
