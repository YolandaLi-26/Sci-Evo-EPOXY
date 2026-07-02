#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG评估系统 —— 领域专业知识推理数据集性能测试
==============================================
对比 DeepSeek-R1 在「无知识库（基线）」与「含知识库（RAG）」
两种条件下的回答质量，定量 + 定性双维度评估。

依赖：openai  jieba  scikit-learn  numpy  matplotlib
运行：python rag_evaluator.py
"""

import os, json, re, time, csv, random, datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import matplotlib
matplotlib.use("Agg")          # 非交互式后端，适合笔记本无显示器运行
import matplotlib.pyplot as plt


# ╔══════════════════════════════════════════════════════════════╗
# ║  ① 配置区  —— 修改此处                                        ║
# ╚══════════════════════════════════════════════════════════════╝
API_KEY    = os.getenv("DEEPSEEK_API_KEY", "")  # ← 填入你的Key
BASE_URL   = "https://api.deepseek.com"
MAIN_MODEL = "deepseek-reasoner"   # R1：主问答模型
EVAL_MODEL = "deepseek-chat"       # V3：LLM评判（更快更便宜）

DATA_DIR   = ""    # ← 改成你的JSON文件目录
OUT_DIR    = ""  # ← 改成你的输出目录（自动创建）
TOP_K      = 3                     # 每次检索返回文档数
CALL_WAIT  = 3                     # API调用间隔秒数（防限速）
MAX_RETRY  = 3                     # API失败最大重试次数


# ╔══════════════════════════════════════════════════════════════╗
# ║  ② 工具函数                                                    ║
# ╚══════════════════════════════════════════════════════════════╝
def nums_in(text: str) -> set:
    """提取文本中所有数字（字符串形式）"""
    return set(re.findall(r"\d+\.?\d*", text))

def kws_in(text: str, n: int = 8) -> List[str]:
    """用 jieba 分词，取前 n 个有意义的中文词"""
    words = [w for w in jieba.cut(text) if len(w) > 1 and re.search(r"[\u4e00-\u9fff]", w)]
    seen, out = set(), []
    for w in words:
        if w not in seen:
            seen.add(w); out.append(w)
    return out[:n]

def safe_mean(lst) -> Optional[float]:
    """对列表取均值，忽略 None"""
    vals = [v for v in lst if v is not None]
    return float(np.mean(vals)) if vals else None

def doc2text(doc: Dict) -> str:
    return json.dumps(doc, ensure_ascii=False)


# ╔══════════════════════════════════════════════════════════════╗
# ║  ③ 数据加载                                                    ║
# ╚══════════════════════════════════════════════════════════════╝
def load_data(data_dir: str, max_n: int = None) -> Tuple[List[Dict], List[str]]:
    p = Path(data_dir)
    if not p.exists():
        raise FileNotFoundError(f"数据目录不存在：{p.resolve()}\n请创建该目录并放入JSON文件。")

    files = sorted(p.glob("*.json"))
    if max_n:
        files = files[:max_n]
    if not files:
        raise FileNotFoundError(f"在 {p} 中未找到任何 .json 文件")

    docs, names = [], []
    for f in files:
        try:
            docs.append(json.loads(f.read_text("utf-8")))
            names.append(f.name)
        except Exception as e:
            print(f"  ⚠ 跳过 {f.name}：{e}")

    print(f"✅ 成功加载 {len(docs)} 个文档")
    return docs, names


# ╔══════════════════════════════════════════════════════════════╗
# ║  ④ TF-IDF 检索器（中文，基于 jieba）                            ║
# ╚══════════════════════════════════════════════════════════════╝
class Retriever:
    def __init__(self, docs: List[Dict], names: List[str]):
        self.docs  = docs
        self.names = names
        print("⏳ 构建 TF-IDF 检索索引...")
        tokenized = [" ".join(jieba.cut(doc2text(d))) for d in docs]
        self.vec = TfidfVectorizer(min_df=1)
        self.mat = self.vec.fit_transform(tokenized)
        print(f"✅ 索引就绪（{len(docs)} 篇文档）")

    def retrieve(self, query: str, k: int = TOP_K) -> List[Dict]:
        q_vec = self.vec.transform([" ".join(jieba.cut(query))])
        sims  = cosine_similarity(q_vec, self.mat)[0]
        idx   = np.argsort(sims)[::-1][:k]
        return [
            {"doc": self.docs[i], "score": float(sims[i]), "fn": self.names[i]}
            for i in idx if sims[i] > 1e-6
        ]


# ╔══════════════════════════════════════════════════════════════╗
# ║  ⑤ 问题生成器（从 JSON 模板结构自动提取问答对）                   ║
# ╚══════════════════════════════════════════════════════════════╝
TYPE_LABEL = {
    "materials":   "原材料清单",
    "strategy":    "实验策略",
    "synthesis":   "合成工艺参数",
    "performance": "最优性能数值",
    "application": "应用场景",
}

class QuestionGenerator:
    def run(self, docs: List[Dict], names: List[str]) -> List[Dict]:
        qs = []
        for doc, fn in zip(docs, names):
            qs += self._from_doc(doc, fn)
        return qs

    @staticmethod
    def _to_str(val) -> str:
        """将任意JSON值统一转为字符串：列表用顿号拼接，其余直接str()"""
        if val is None:
            return ""
        if isinstance(val, list):
            return "、".join(str(x) for x in val if x)
        if isinstance(val, dict):
            return "；".join(f"{k}={v}" for k, v in val.items())
        return str(val)

    def _from_doc(self, doc: Dict, fn: str) -> List[Dict]:
        qs   = []
        info = doc.get("01_初始信息", {})
        # 兼容 info 不是 dict 的情况
        if not isinstance(info, dict):
            info = {}

        goal = self._to_str(info.get("全局目标", "该研究课题")) or "该研究课题"
        tools = info.get("用户工具", {})
        if not isinstance(tools, dict):
            tools = {}

        # ── 类型①：原材料（关键词题）────────────────────────────────
        mat_raw = tools.get("原材料", "")
        mat = self._to_str(mat_raw)
        if mat:
            qs.append(self._q(
                f"{fn}_mat",
                f"研究课题'{goal}'中使用了哪些主要原材料？请逐一列出。",
                mat, "materials", fn,
                [m.strip() for m in re.split(r"[、，,；;]", mat) if m.strip()][:6],
                []
            ))

        # ── 类型②：整体实验策略（综合理解题）──────────────────────────
        strat = self._to_str(info.get("整体策略", ""))
        if strat:
            qs.append(self._q(
                f"{fn}_strat",
                f"'{goal}'研究的核心实验策略和技术路线是什么？",
                strat, "strategy", fn,
                kws_in(strat), self._extract_nums(strat)
            ))

        # ── 类型③：合成工艺参数（数值题）──────────────────────────────
        reasoning_chain = doc.get("02_推理链", [])
        if not isinstance(reasoning_chain, list):
            reasoning_chain = []

        for step in reasoning_chain[:2]:
            if not isinstance(step, dict):
                continue

            sn = step.get("推理序号", 1)
            goal_of_step = self._to_str(step.get("当前目标", ""))

            # ★ 修复：其它参数可能是 list 而非 dict，做类型防御
            inputs = step.get("输入", {})
            if not isinstance(inputs, dict):
                inputs = {}

            raw_params = inputs.get("其它参数", {})
            if not isinstance(raw_params, dict):
                raw_params = {}

            params = {
                k: self._to_str(v)
                for k, v in raw_params.items()
                if v and "未明确" not in str(v)
            }
            if params:
                ref = "；".join(f"{k}={v}" for k, v in params.items())
                qs.append(self._q(
                    f"{fn}_s{sn}",
                    f"'{goal}'第{sn}步（{goal_of_step}）的关键工艺参数（温度/时间/配比等）是什么？",
                    ref, "synthesis", fn,
                    list(params.keys()), self._extract_nums(ref)
                ))

        # ── 类型④：最优性能数值（最重要！RAG优势最明显）──────────────
        target_confirm = doc.get("04_目标确认", {})
        if not isinstance(target_confirm, dict):
            target_confirm = {}

        perf_scale = target_confirm.get("效能刻度", {})
        if not isinstance(perf_scale, dict):
            perf_scale = {}

        for k, v in perf_scale.items():
            # ★ 修复：v 可能不是 dict
            if not isinstance(v, dict):
                continue
            perf = self._to_str(v.get("性能或应用", ""))
            num  = self._to_str(v.get("数值和单位", ""))
            note = self._to_str(v.get("解读", ""))
            if perf and num and "未明确" not in num:
                ref = f"{num}。{note}".strip("。")
                qs.append(self._q(
                    f"{fn}_{k}",
                    f"'{goal}'研究中，{perf}的最优值是多少？这说明了什么？",
                    ref, "performance", fn,
                    [perf] + kws_in(note), self._extract_nums(ref)
                ))

        # ── 类型⑤：应用场景（综合应用题）──────────────────────────────
        tasks = doc.get("03_任务域", [])
        if not isinstance(tasks, list):
            tasks = []

        if tasks:
            task0 = tasks[0]
            if isinstance(task0, dict):
                sc    = self._to_str(task0.get("落地场景", ""))
                logic = self._to_str(task0.get("驱动逻辑", ""))
                if sc:
                    ref = f"{sc}。{logic}".strip("。")
                    qs.append(self._q(
                        f"{fn}_app",
                        f"'{goal}'研究成果的主要应用场景及其实际价值是什么？",
                        ref, "application", fn,
                        kws_in(ref), []
                    ))

        return qs

    def _q(self, qid, question, ref, qtype, fn, kws, nums) -> Dict:
        return {
            "id":   qid,
            "q":    question,
            "ref":  ref,
            "type": qtype,
            "fn":   fn,
            "kws":  [k for k in kws if k],
            "nums": [n.strip() for n in nums if re.search(r"\d", n)],
        }

    def _extract_nums(self, text: str) -> List[str]:
        return re.findall(r"\d+\.?\d*\s*(?:MPa|kJ/m²|kJ|℃|°C|K|min|h|%|g|ml|mol|mm|μm|nm|phr)?", text)


# ╔══════════════════════════════════════════════════════════════╗
# ║  ⑥ DeepSeek API 客户端                                        ║
# ╚══════════════════════════════════════════════════════════════╝
class LLM:
    def __init__(self):
        self.cli = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    def _call(self, msgs: List[Dict], model: str, temp: float = 0.1) -> str:
        for attempt in range(MAX_RETRY):
            try:
                r = self.cli.chat.completions.create(
                    model=model, messages=msgs,
                    temperature=temp, max_tokens=1500,
                )
                return r.choices[0].message.content or ""
            except Exception as e:
                wait = CALL_WAIT * (attempt + 1)
                print(f"    ⚠ 重试 {attempt+1}/{MAX_RETRY}（{e}），{wait}s 后...")
                time.sleep(wait)
        return "[API Error]"

    # 基线：不提供任何知识库上下文
    def baseline(self, q: str) -> str:
        return self._call([
            {"role": "system", "content":
             "你是材料科学和化学领域的专家助手。"
             "请仅用自身已有专业知识回答，若不确定具体数值请如实说明，不要猜测。"},
            {"role": "user", "content": q},
        ], MAIN_MODEL)

    # RAG：将检索到的JSON文档作为上下文注入
    def rag(self, q: str, ctxs: List[Dict]) -> str:
        ctx_text = "\n\n---\n".join(
            f"【参考资料{i+1}】来源文件: {c['fn']}  相关度: {c['score']:.3f}\n{doc2text(c['doc'])}"
            for i, c in enumerate(ctxs)
        )
        return self._call([
            {"role": "system", "content":
             "你是材料科学和化学领域的专家助手。"
             "请优先依据提供的知识库资料作答，并明确引用数据来源（文件名+具体数值）。"
             "若知识库中有直接答案，务必给出精确数值。"},
            {"role": "user", "content":
             f"知识库资料：\n{ctx_text}\n\n"
             f"{'─'*40}\n"
             f"问题：{q}\n\n"
             f"请从知识库中提取具体数值、原材料名称和工艺条件回答上述问题。"},
        ], MAIN_MODEL)

    # LLM-as-Judge：让 V3 评判 R1 的两个答案
    def judge(self, q: str, ans_a: str, ans_b: str, ref: str) -> Dict:
        prompt = f"""你是材料科学领域的严格评审专家，请评估两个AI回答的质量。

【问题】
{q}

【标准参考答案】
{ref}

【答案A】（无外部知识库，仅凭模型记忆）
{ans_a[:800]}

【答案B】（提供了外部知识库，RAG增强）
{ans_b[:800]}

请为每个答案在以下三个维度打分（1-10分）：
- 准确性：具体数值、材料名称、工艺参数是否与参考答案一致？
- 完整性：是否涵盖了参考答案中的主要信息点？
- 无幻觉：是否包含虚构/错误信息？（10=完全无幻觉，1=严重捏造）
- 总分 = 三项均值

【严格要求】只输出以下JSON，不含任何其他文字：
{{"A":{{"准确":X,"完整":X,"无幻觉":X,"总分":X,"评":"一句评语"}},"B":{{"准确":X,"完整":X,"无幻觉":X,"总分":X,"评":"一句评语"}},"对比":"RAG相比基线的核心差异"}}"""

        resp = self._call([{"role": "user", "content": prompt}], EVAL_MODEL, temp=0)
        try:
            m = re.search(r"\{[\s\S]*\}", resp)
            return json.loads(m.group()) if m else {}
        except Exception:
            return {}


# ╔══════════════════════════════════════════════════════════════╗
# ║  ⑦ 定量评估指标                                                ║
# ╚══════════════════════════════════════════════════════════════╝
def num_accuracy(answer: str, ref_nums: List[str]) -> Optional[float]:
    """
    数值准确率：答案中命中参考数值的比例
    ref_nums 来自 JSON 效能刻度中的具体数字
    """
    ref_set = nums_in(" ".join(ref_nums))
    if not ref_set:
        return None
    ans_set = nums_in(answer)
    return len(ref_set & ans_set) / len(ref_set)

def kw_recall(answer: str, keywords: List[str]) -> Optional[float]:
    """
    关键词召回率：答案中提到参考关键词的比例
    keywords 用 jieba 从标准答案中提取
    """
    if not keywords:
        return None
    return sum(1 for k in keywords if k in answer) / len(keywords)


# ╔══════════════════════════════════════════════════════════════╗
# ║  ⑧ 报告生成（JSON + CSV + 可视化图表）                          ║
# ╚══════════════════════════════════════════════════════════════╝
class Reporter:
    def __init__(self, out_dir: str):
        self.d = Path(out_dir)
        self.d.mkdir(exist_ok=True)

    def save_all(self, results: List[Dict], ts: str) -> Dict[str, Path]:
        paths = {}
        paths["json"]  = self._save_json(results, ts)
        paths["csv"]   = self._save_csv(results, ts)
        try:
            paths["chart"] = self._save_chart(results, ts)
        except Exception as e:
            print(f"  ⚠ 图表生成跳过：{e}")
        return paths

    def _save_json(self, results: List[Dict], ts: str) -> Path:
        p = self.d / f"detail_{ts}.json"
        p.write_text(json.dumps(results, ensure_ascii=False, indent=2), "utf-8")
        return p

    def _save_csv(self, results: List[Dict], ts: str) -> Path:
        p = self.d / f"summary_{ts}.csv"
        cols = [
            "问题ID", "类型", "问题（前50字）", "来源文件",
            "基线_数值准确率", "RAG_数值准确率", "数值提升",
            "基线_关键词召回", "RAG_关键词召回", "召回提升",
            "基线_LLM总分", "RAG_LLM总分", "LLM提升",
            "检索命中源文件", "检索到的文件", "LLM对比评语",
        ]
        def f(v): return f"{v:.3f}" if v is not None else "-"
        def diff(a, b): return f"{b-a:+.2f}" if a is not None and b is not None else "-"

        rows = []
        for r in results:
            j  = r.get("judge", {})
            a  = j.get("A", {}); b = j.get("B", {})
            rows.append({
                "问题ID": r["id"],
                "类型": TYPE_LABEL.get(r["type"], r["type"]),
                "问题（前50字）": r["q"][:50],
                "来源文件": r["fn"],
                "基线_数值准确率": f(r.get("bn")),
                "RAG_数值准确率":  f(r.get("rn")),
                "数值提升":        diff(r.get("bn"), r.get("rn")),
                "基线_关键词召回": f(r.get("bk")),
                "RAG_关键词召回":  f(r.get("rk")),
                "召回提升":        diff(r.get("bk"), r.get("rk")),
                "基线_LLM总分":    a.get("总分", "-"),
                "RAG_LLM总分":     b.get("总分", "-"),
                "LLM提升":         diff(a.get("总分"), b.get("总分")),
                "检索命中源文件":  "✅" if r.get("hit") else "❌",
                "检索到的文件":    " | ".join(c["fn"] for c in r.get("ctxs", [])),
                "LLM对比评语":     j.get("对比", "")[:60],
            })

        with open(p, "w", newline="", encoding="utf-8-sig") as f_:
            w = csv.DictWriter(f_, fieldnames=cols)
            w.writeheader(); w.writerows(rows)
        return p

    def _save_chart(self, results: List[Dict], ts: str) -> Path:
        # 中文字体
        plt.rcParams.update({
            "font.sans-serif": ["SimHei", "PingFang SC", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
        })

        # ── 聚合数据 ────────────────────────────────────────
        def gavg(key, sub=None):
            if sub:
                vals = [r.get("judge", {}).get(sub, {}).get(key) for r in results]
            else:
                vals = [r.get(key) for r in results]
            return safe_mean(vals) or 0

        bn = gavg("bn"); rn = gavg("rn")
        bk = gavg("bk"); rk = gavg("rk")
        bl = gavg("总分", "A"); rl = gavg("总分", "B")

        types = list(dict.fromkeys(r["type"] for r in results))  # 保持顺序

        def by_type(sub, key):
            return [safe_mean([r.get("judge",{}).get(sub,{}).get(key)
                               for r in results if r["type"] == t]) or 0
                    for t in types]

        type_bl = by_type("A", "总分")
        type_rl = by_type("B", "总分")
        type_labels = [TYPE_LABEL.get(t, t) for t in types]

        # ── 绘图 ────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        fig.suptitle("RAG 评估报告 —— 数据集性能对比", fontsize=14, fontweight="bold", y=1.01)

        RED, GRN = "#E74C3C", "#27AE60"
        w = 0.32

        # 左图：三大指标总体对比
        ax = axes[0]
        cats = ["数值准确率", "关键词召回率", "LLM评分(/10)"]
        bv   = [bn, bk, bl / 10]
        rv   = [rn, rk, rl / 10]
        x    = np.arange(3)
        b1 = ax.bar(x - w/2, bv, w, label="基线（无知识库）", color=RED,  alpha=0.85, edgecolor="white", linewidth=0.8)
        b2 = ax.bar(x + w/2, rv, w, label="RAG（含知识库）",  color=GRN, alpha=0.85, edgecolor="white", linewidth=0.8)
        ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=10)
        ax.set_ylim(0, 1.3); ax.set_ylabel("得分", fontsize=11)
        ax.set_title("整体性能对比", fontsize=12); ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        for bar, col in [(b1, RED), (b2, GRN)]:
            for b in bar:
                h = b.get_height()
                ax.text(b.get_x() + b.get_width()/2, h + 0.012, f"{h:.3f}",
                        ha="center", fontsize=8.5, color="#333")

        # 右图：按问题类型的 LLM 评分
        ax2 = axes[1]
        x2 = np.arange(len(types))
        ax2.bar(x2 - w/2, type_bl, w, label="基线", color=RED,  alpha=0.85, edgecolor="white")
        ax2.bar(x2 + w/2, type_rl, w, label="RAG",  color=GRN, alpha=0.85, edgecolor="white")
        ax2.set_xticks(x2); ax2.set_xticklabels(type_labels, fontsize=9, rotation=15, ha="right")
        ax2.set_ylim(0, 12); ax2.set_ylabel("LLM 评判总分（/10）", fontsize=11)
        ax2.set_title("各问题类型得分对比", fontsize=12); ax2.legend(fontsize=9)
        ax2.grid(axis="y", alpha=0.3, linestyle="--")
        for bl_v, rl_v, xi in zip(type_bl, type_rl, x2):
            ax2.text(xi - w/2, bl_v + 0.1, f"{bl_v:.1f}", ha="center", fontsize=8)
            ax2.text(xi + w/2, rl_v + 0.1, f"{rl_v:.1f}", ha="center", fontsize=8)

        plt.tight_layout()
        p = self.d / f"chart_{ts}.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        return p

    def print_summary(self, results: List[Dict]):
        """终端打印汇总报告"""
        def gavg(key, sub=None):
            if sub:
                vals = [r.get("judge", {}).get(sub, {}).get(key) for r in results]
            else:
                vals = [r.get(key) for r in results]
            return safe_mean(vals)

        bn = gavg("bn"); rn = gavg("rn")
        bk = gavg("bk"); rk = gavg("rk")
        bl = gavg("总分", "A"); rl = gavg("总分", "B")

        def row(name, base, rag, fmt=".3f"):
            if base is None or rag is None:
                return
            diff = rag - base
            sym  = "↑" if diff > 0 else "↓"
            bar  = "█" * int(abs(diff) / 0.05) if fmt == ".3f" else "█" * int(abs(diff) / 0.5)
            print(f"  {name:<22} 基线={base:{fmt}}  RAG={rag:{fmt}}  {sym}{abs(diff):{fmt}}  {bar}")

        sep = "═" * 58
        print(f"\n{sep}")
        print(f"  📊 RAG 评估汇总报告  （共 {len(results)} 道测试题）")
        print(sep)
        print(f"  数据集规模: {len(set(r['fn'] for r in results))} 个 JSON 文档")
        print(f"\n  {'指标':<22}  {'基线':^10}  {'RAG':^10}  变化")
        print(f"  {'─'*54}")
        row("数值准确率",        bn, rn)
        row("关键词召回率",      bk, rk)
        row("LLM 评判总分(/10)", bl, rl, ".2f")

        # 命中率
        hit_rate = sum(1 for r in results if r.get("hit")) / len(results)
        print(f"\n  检索命中源文件率: {hit_rate:.1%}（共 {len(results)} 题）")

        # 按类型
        types = list(dict.fromkeys(r["type"] for r in results))
        print(f"\n  按问题类型（LLM 总分）：")
        print(f"  {'─'*54}")
        for t in types:
            tr  = [r for r in results if r["type"] == t]
            ta  = safe_mean([r.get("judge", {}).get("A", {}).get("总分") for r in tr])
            tb  = safe_mean([r.get("judge", {}).get("B", {}).get("总分") for r in tr])
            lbl = TYPE_LABEL.get(t, t)
            if ta and tb:
                diff = tb - ta
                sym  = "↑" if diff > 0 else "↓"
                print(f"  {lbl:<12}  基线={ta:.2f} → RAG={tb:.2f}  {sym}{abs(diff):.2f}  (n={len(tr)})")

        print(f"\n{sep}\n")


# ╔══════════════════════════════════════════════════════════════╗
# ║  ⑨ 主流程                                                      ║
# ╚══════════════════════════════════════════════════════════════╝
def run(max_docs: int = None, max_qs: int = None, seed: int = 42):
    """
    Parameters
    ----------
    max_docs : int | None
        限制加载的文档数（调试时用小数字，完整评测设 None）
    max_qs : int | None
        限制测试问题数（调试时用小数字，完整评测设 None）
    seed : int
        随机种子（保证采样可复现）
    """
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    Path(OUT_DIR).mkdir(exist_ok=True)
    progress_file = Path(OUT_DIR) / f"progress_{ts}.json"

    print("\n" + "╔" + "═"*54 + "╗")
    print(f"  🚀 RAG 评估系统 启动  {ts}")
    print(f"  主问答模型: {MAIN_MODEL}")
    print(f"  评判模型:   {EVAL_MODEL}")
    print("╚" + "═"*54 + "╝\n")

    # Step 1 ── 加载数据
    print("─── Step 1 / 5: 加载数据集 ───")
    docs, names = load_data(DATA_DIR, max_docs)

    # Step 2 ── 构建检索器
    print("\n─── Step 2 / 5: 构建检索系统 ───")
    retriever = Retriever(docs, names)

    # Step 3 ── 生成问题
    print("\n─── Step 3 / 5: 生成测试问题 ───")
    all_qs = QuestionGenerator().run(docs, names)
    if max_qs and len(all_qs) > max_qs:
        random.seed(seed)
        random.shuffle(all_qs)
        questions = all_qs[:max_qs]
        print(f"  随机采样 {max_qs} / {len(all_qs)} 个问题（seed={seed}）")
    else:
        questions = all_qs

    print(f"  共 {len(questions)} 个问题，类型分布：")
    type_cnt = {}
    for q in questions:
        type_cnt[q["type"]] = type_cnt.get(q["type"], 0) + 1
    for t, c in type_cnt.items():
        print(f"    {TYPE_LABEL.get(t, t):<12}: {c} 个")

    # Step 4 ── 初始化工具
    llm      = LLM()
    reporter = Reporter(OUT_DIR)
    results  = []

    # Step 5 ── 逐题评测
    print(f"\n─── Step 4 / 5: 开始评测 ({len(questions)} 题) ───\n")
    total_cost_est = len(questions) * 3  # baseline + rag + judge
    print(f"  预计 API 调用次数约 {total_cost_est} 次，请耐心等待...\n")

    for i, q in enumerate(questions, 1):
        print(f"┌── [{i:02d}/{len(questions):02d}] {TYPE_LABEL.get(q['type'], q['type'])}")
        print(f"│   问题：{q['q'][:60]}{'…' if len(q['q'])>60 else ''}")

        # 4a. 基线
        print("│   📤 [基线] 调用 R1（无知识库）...")
        t0 = time.time()
        base_ans = llm.baseline(q["q"])
        print(f"│   ✓ 完成（{time.time()-t0:.0f}s），答案 {len(base_ans)} 字")
        time.sleep(CALL_WAIT)

        # 4b. 检索上下文
        ctxs = retriever.retrieve(q["q"], TOP_K)
        hit  = q["fn"] in [c["fn"] for c in ctxs]
        print(f"│   🔍 检索：{[c['fn'] for c in ctxs]}")
        print(f"│      命中源文件：{'✅ 是' if hit else '❌ 否（需调整检索策略）'}")

        # 4c. RAG
        print("│   📤 [RAG] 调用 R1（含知识库）...")
        t0 = time.time()
        rag_ans = llm.rag(q["q"], ctxs)
        print(f"│   ✓ 完成（{time.time()-t0:.0f}s），答案 {len(rag_ans)} 字")
        time.sleep(CALL_WAIT)

        # 4d. 量化指标
        bn = num_accuracy(base_ans, q["nums"])
        rn = num_accuracy(rag_ans,  q["nums"])
        bk = kw_recall(base_ans, q["kws"])
        rk = kw_recall(rag_ans,  q["kws"])

        num_str = f"数值准确率 基线={f'{bn:.2f}' if bn is not None else 'N/A'} → RAG={f'{rn:.2f}' if rn is not None else 'N/A'}"
        kw_str  = f"关键词召回 基线={f'{bk:.2f}' if bk is not None else 'N/A'} → RAG={f'{rk:.2f}' if rk is not None else 'N/A'}"
        print(f"│   📈 {num_str}")
        print(f"│   📈 {kw_str}")

        # 4e. LLM 评判
        print("│   ⚖️  LLM 评判（deepseek-chat）...")
        judge = llm.judge(q["q"], base_ans, rag_ans, q["ref"])
        a_s = judge.get("A", {}).get("总分", "?")
        b_s = judge.get("B", {}).get("总分", "?")
        cmt = judge.get("对比", "")[:50]
        print(f"│   ✓ 基线={a_s}/10  RAG={b_s}/10  「{cmt}」")
        print("└" + "─"*54)

        # 汇总这道题的结果
        result = {
            **q,
            "base_answer": base_ans,
            "rag_answer":  rag_ans,
            "ctxs": [{"fn": c["fn"], "score": round(c["score"], 4)} for c in ctxs],
            "hit": hit,
            "bn": bn, "rn": rn,
            "bk": bk, "rk": rk,
            "judge": judge,
        }
        results.append(result)

        # 实时保存进度（防止中途崩溃丢失数据）
        progress_file.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), "utf-8"
        )
        time.sleep(CALL_WAIT)

    # Step 5 ── 生成报告
    print(f"\n─── Step 5 / 5: 生成评估报告 ───")
    paths = reporter.save_all(results, ts)
    reporter.print_summary(results)

    print("📁 输出文件（./results/ 目录）：")
    labels = {"json": "详细记录 JSON", "csv": "汇总表格 CSV", "chart": "对比图表 PNG"}
    for k, p in paths.items():
        print(f"  [{labels.get(k, k)}]  {p.name}")
    print(f"  [进度备份]              {progress_file.name}")

    return results


# ╔══════════════════════════════════════════════════════════════╗
# ║  ⑩ 入口                                                        ║
# ╚══════════════════════════════════════════════════════════════╝
if __name__ == "__main__":
    # ┌─────────────────────────────────────────────────────────┐
    # │ 参数说明：                                               │
    # │  max_docs=5,  max_qs=10  →  快速验证（约 30~60 分钟）    │
    # │  max_docs=20, max_qs=60  →  中等规模（约 2~4 小时）      │
    # │  max_docs=None, max_qs=None  →  完整评测（约半天）       │
    # │                                                          │
    # │ API 费用估算（15题快速测试）：                             │
    # │  R1 × 30次 + V3 × 15次 ≈ 约 ¥2~5                       │
    # └─────────────────────────────────────────────────────────┘
    run(
        max_docs=None,    # 先用 5 个文档做快速验证
        max_qs=10,        # 先测 10 道题
    )
