#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG 推理链评测系统
==================
专项评测「加入领域知识库前后」模型在推理链/逻辑链维度的能力变化。

评测维度（仅针对推理链）：
  ① 步骤完整性   —— 是否正确复现了N步推理链的顺序与数量
  ② 决策逻辑性   —— 每步「当前状态→决策→判定」的因果链是否合理
  ③ 参数精确性   —— 工艺参数/数值（温度/时间/浓度等）是否被准确引用
  ④ 链式连贯性   —— 上一步输出是否正确成为下一步输入（承接关系）
  ⑤ 幻觉抑制率   —— 是否凭空捏造了推理链中不存在的步骤/数据

依赖：openai  jieba  scikit-learn  numpy  matplotlib
运行：python rag_reasoning_evaluator.py
"""

import os, json, re, time, csv, random, datetime, textwrap
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ╔══════════════════════════════════════════════════════════════╗
# ║  ① 配置区  —— 修改此处                                        ║
# ╚══════════════════════════════════════════════════════════════╝
API_KEY    = os.getenv("DEEPSEEK_API_KEY", "")   # ← 填入你的Key
BASE_URL   = "https://api.deepseek.com"
MAIN_MODEL = "deepseek-reasoner"    # R1：主问答模型
EVAL_MODEL = "deepseek-chat"        # V3：LLM评判

DATA_DIR   = ""     # ← JSON文件目录
OUT_DIR    = ""  # ← 输出目录
TOP_K      = 3       # 检索返回文档数
CALL_WAIT  = 3       # API调用间隔秒数
MAX_RETRY  = 3       # API失败最大重试次数

# 推理链问题模板类型（仅保留与推理逻辑相关的维度）
REASONING_DIM_LABELS = {
    "step_order":    "步骤完整性",
    "decision_logic":"决策逻辑性",
    "param_accuracy":"参数精确性",
    "chain_coherence":"链式连贯性",
    "hallucination": "幻觉抑制率",
}


# ╔══════════════════════════════════════════════════════════════╗
# ║  ② 工具函数                                                    ║
# ╚══════════════════════════════════════════════════════════════╝
def nums_in(text: str) -> set:
    """提取文本中所有数值（含单位片段）"""
    return set(re.findall(r"\d+\.?\d*\s*(?:MPa|kJ/m²|℃|°C|K|min|h|%|g|ml|mol|mm|μm|mPa·s|phr)?", text))

def safe_mean(lst) -> Optional[float]:
    vals = [v for v in lst if v is not None]
    return float(np.mean(vals)) if vals else None

def doc2text(doc: Dict) -> str:
    return json.dumps(doc, ensure_ascii=False)

def wrap_print(text: str, width: int = 72, prefix: str = "│   "):
    for line in textwrap.wrap(str(text), width):
        print(prefix + line)


# ╔══════════════════════════════════════════════════════════════╗
# ║  ③ 数据加载                                                    ║
# ╚══════════════════════════════════════════════════════════════╝
def load_data(data_dir: str, max_n: int = None) -> Tuple[List[Dict], List[str]]:
    p = Path(data_dir)
    if not p.exists():
        raise FileNotFoundError(f"数据目录不存在：{p.resolve()}")

    files = sorted(p.glob("*.json"))
    if max_n:
        files = files[:max_n]
    if not files:
        raise FileNotFoundError(f"在 {p} 中未找到任何 .json 文件")

    docs, names = [], []
    for f in files:
        try:
            data = json.loads(f.read_text("utf-8"))
            # 只保留含有推理链的文档
            if data.get("02_推理链"):
                docs.append(data)
                names.append(f.name)
            else:
                print(f"  ⚠ 跳过（无推理链）：{f.name}")
        except Exception as e:
            print(f"  ⚠ 解析失败：{f.name}：{e}")

    print(f"✅ 成功加载 {len(docs)} 个含推理链的文档")
    return docs, names


# ╔══════════════════════════════════════════════════════════════╗
# ║  ④ TF-IDF 检索器（中文，基于 jieba）                            ║
# ╚══════════════════════════════════════════════════════════════╝
class Retriever:
    def __init__(self, docs: List[Dict], names: List[str]):
        self.docs  = docs
        self.names = names
        print("⏳ 构建 TF-IDF 检索索引（仅推理链字段）...")
        # 仅对推理链文本建索引，聚焦检索效果
        tokenized = [
            " ".join(jieba.cut(
                json.dumps(d.get("02_推理链", []), ensure_ascii=False)
            )) for d in docs
        ]
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
# ║  ⑤ 推理链问题生成器                                             ║
# ║                                                              ║
# ║  每篇文档生成5类推理链专项问题，全部聚焦"02_推理链"字段           ║
# ╚══════════════════════════════════════════════════════════════╝
class ReasoningQuestionGenerator:

    def run(self, docs: List[Dict], names: List[str]) -> List[Dict]:
        qs = []
        for doc, fn in zip(docs, names):
            qs += self._from_doc(doc, fn)
        return qs

    def _from_doc(self, doc: Dict, fn: str) -> List[Dict]:
        chain = doc.get("02_推理链", [])
        if not isinstance(chain, list) or not chain:
            return []

        info   = doc.get("01_初始信息", {})
        goal   = info.get("全局目标", "该研究课题") if isinstance(info, dict) else "该研究课题"
        qs     = []
        n_step = len(chain)

        # ── ① 步骤完整性 ────────────────────────────────────────
        # 考察：能否说出完整推理链的步骤数量、每步目标
        step_goals = [
            f"第{s.get('推理序号', i+1)}步：{self._to_str(s.get('当前目标',''))}"
            for i, s in enumerate(chain) if isinstance(s, dict)
        ]
        ref_steps = "\n".join(step_goals)
        qs.append({
            "id":   f"{fn}_step_order",
            "type": "step_order",
            "fn":   fn,
            "q":    f"请详细描述'{goal}'研究的完整实验推理过程，包括共有几个推理步骤、每步的核心目标是什么，以及各步骤之间的逻辑顺序关系。",
            "ref":  ref_steps,
            "ref_nums": [],
            "ref_kws":  step_goals,   # 每步目标作为关键点
            "chain_ref": chain,
        })

        # ── ② 决策逻辑性 ────────────────────────────────────────
        # 考察：给定某步"当前状态"，能否推导出正确的"决策"和"判定"
        for step in chain:
            if not isinstance(step, dict):
                continue
            sn      = step.get("推理序号", 1)
            state   = self._to_str(step.get("当前状态", ""))
            decision= self._to_str(step.get("决策", ""))
            verdict = self._to_str(step.get("判定", ""))
            if state and decision:
                ref = f"决策：{decision}；判定：{verdict}"
                qs.append({
                    "id":   f"{fn}_decision_{sn}",
                    "type": "decision_logic",
                    "fn":   fn,
                    "q":    f"在'{goal}'研究中，当面临「{state[:60]}」这一状况时，研究者采取了什么核心决策？最终结论是什么？请说明其科学依据。",
                    "ref":  ref,
                    "ref_nums": [],
                    "ref_kws":  [w for w in jieba.cut(decision) if len(w) > 1][:6],
                    "chain_ref": [step],
                })

        # ── ③ 参数精确性 ────────────────────────────────────────
        # 考察：工艺参数数值（温度/时间/浓度/强度等）能否被精确引用
        for step in chain:
            if not isinstance(step, dict):
                continue
            sn     = step.get("推理序号", 1)
            inputs = step.get("输入", {})
            if not isinstance(inputs, dict):
                continue
            params = inputs.get("其它参数", {})
            if not isinstance(params, dict):
                continue
            param_items = {
                k: self._to_str(v)
                for k, v in params.items()
                if v and "未明确" not in str(v)
            }
            if param_items:
                ref = "；".join(f"{k}={v}" for k, v in param_items.items())
                ref_nums = list(nums_in(ref))
                step_goal = self._to_str(step.get("当前目标", f"第{sn}步操作"))
                qs.append({
                    "id":   f"{fn}_param_{sn}",
                    "type": "param_accuracy",
                    "fn":   fn,
                    "q":    f"'{goal}'研究的第{sn}步（{step_goal[:40]}）涉及哪些关键工艺参数？请给出所有具体数值（温度、时间、浓度、强度等），并说明这些参数的控制意义。",
                    "ref":  ref,
                    "ref_nums": ref_nums,
                    "ref_kws":  list(param_items.keys()),
                    "chain_ref": [step],
                })

        # ── ④ 链式连贯性 ────────────────────────────────────────
        # 考察：相邻步骤之间"上步输出→下步输入"的承接关系
        for i in range(len(chain) - 1):
            s_prev = chain[i] if isinstance(chain[i], dict) else {}
            s_next = chain[i+1] if isinstance(chain[i+1], dict) else {}
            sn_prev = s_prev.get("推理序号", i+1)
            sn_next = s_next.get("推理序号", i+2)
            out_prev = self._to_str(s_prev.get("输出", {}).get(
                list(s_prev.get("输出", {}).keys())[0], "")
                if isinstance(s_prev.get("输出"), dict) and s_prev.get("输出") else "")
            goal_next = self._to_str(s_next.get("当前目标", ""))
            if out_prev and goal_next:
                ref = f"第{sn_prev}步输出：{out_prev}；第{sn_next}步目标：{goal_next}"
                qs.append({
                    "id":   f"{fn}_coherence_{sn_prev}_{sn_next}",
                    "type": "chain_coherence",
                    "fn":   fn,
                    "q":    f"在'{goal}'研究中，第{sn_prev}步完成后得到了什么结果？这个结果如何支撑和推动了第{sn_next}步的开展？请描述两步之间的承接逻辑。",
                    "ref":  ref,
                    "ref_nums": [],
                    "ref_kws":  [w for w in jieba.cut(out_prev + goal_next) if len(w) > 1][:6],
                    "chain_ref": [s_prev, s_next],
                })

        # ── ⑤ 幻觉抑制率 ────────────────────────────────────────
        # 考察：是否会在推理链中捏造不存在的步骤或数据
        # 构造"反向追问"：让模型说出某步的输出，然后由评判者核查
        for step in chain:
            if not isinstance(step, dict):
                continue
            sn     = step.get("推理序号", 1)
            output = step.get("输出", {})
            if not isinstance(output, dict) or not output:
                continue
            out_val = self._to_str(list(output.values())[0])
            if not out_val or "未明确" in out_val:
                continue
            step_goal = self._to_str(step.get("当前目标", ""))
            qs.append({
                "id":   f"{fn}_halluc_{sn}",
                "type": "hallucination",
                "fn":   fn,
                "q":    f"'{goal}'研究第{sn}步（{step_goal[:40]}）完成后，具体得到了什么产物或结论？请给出精确描述，不要添加文献中没有明确提及的内容。",
                "ref":  out_val,
                "ref_nums": list(nums_in(out_val)),
                "ref_kws":  [w for w in jieba.cut(out_val) if len(w) > 1][:6],
                "chain_ref": [step],
            })

        return qs

    @staticmethod
    def _to_str(val) -> str:
        if val is None:
            return ""
        if isinstance(val, list):
            return "、".join(str(x) for x in val if x)
        if isinstance(val, dict):
            return "；".join(f"{k}={v}" for k, v in val.items())
        return str(val)


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

    def baseline(self, q: str) -> str:
        """基线：不提供知识库，仅凭模型自身推理能力"""
        return self._call([
            {"role": "system", "content": (
                "你是材料科学和化学领域的专家助手，具备严密的逻辑推理能力。"
                "请仅根据自身已有的专业知识和推理能力回答问题。"
                "若涉及具体数值或实验参数，如不确定请如实说明，不要猜测或捏造。"
            )},
            {"role": "user", "content": q},
        ], MAIN_MODEL)

    def rag(self, q: str, ctxs: List[Dict]) -> str:
        """RAG：注入检索到的推理链上下文"""
        # 只提取推理链部分，精准注入
        ctx_parts = []
        for i, c in enumerate(ctxs):
            chain = c["doc"].get("02_推理链", [])
            info  = c["doc"].get("01_初始信息", {})
            goal  = info.get("全局目标", "") if isinstance(info, dict) else ""
            chain_text = json.dumps(chain, ensure_ascii=False, indent=2)
            ctx_parts.append(
                f"【参考资料{i+1}】文件: {c['fn']}  相关度: {c['score']:.3f}\n"
                f"研究目标: {goal}\n"
                f"完整推理链:\n{chain_text}"
            )
        ctx_text = "\n\n" + "─" * 40 + "\n".join(ctx_parts)

        return self._call([
            {"role": "system", "content": (
                "你是材料科学和化学领域的专家助手。"
                "请严格依据提供的推理链知识库资料作答，"
                "精确引用每步的状态、决策、参数和判定结论，"
                "不要添加知识库中未明确出现的步骤或数值。"
            )},
            {"role": "user", "content": (
                f"知识库推理链资料：\n{ctx_text}\n\n"
                f"{'─' * 40}\n"
                f"问题：{q}\n\n"
                f"请从推理链知识库中提取相关步骤、决策逻辑和具体参数来回答问题，"
                f"并明确标注数据来源的步骤编号。"
            )},
        ], MAIN_MODEL)

    def judge_reasoning(self, q: str, ans_a: str, ans_b: str,
                        ref: str, q_type: str) -> Dict:
        """
        针对推理链的5维LLM评判
        每维度1-10分，输出结构化JSON
        """
        dim_instructions = {
            "step_order": "重点评估：步骤数量是否正确、每步目标是否准确、顺序是否合理",
            "decision_logic": "重点评估：因果逻辑是否自洽、决策依据是否充分、判定是否合理",
            "param_accuracy": "重点评估：具体数值（温度/时间/浓度等）是否与参考完全一致，错误或缺失的数值严重扣分",
            "chain_coherence": "重点评估：上下步骤之间的输入输出承接是否正确、逻辑是否连贯",
            "hallucination": "重点评估：是否包含参考答案中未提及的内容（捏造步骤/数值/结论），10=完全无幻觉，1=严重捏造",
        }
        focus = dim_instructions.get(q_type, "综合评估推理链质量")
        dim_name = REASONING_DIM_LABELS.get(q_type, q_type)

        prompt = f"""你是材料科学推理链评审专家，专门评估AI对实验推理逻辑的掌握程度。

【评测维度】{dim_name}
【评测重点】{focus}

【问题】
{q}

【标准参考答案（来自原始推理链JSON）】
{ref}

【答案A】（基线：无知识库，仅凭模型记忆推理）
{ans_a[:1000]}

【答案B】（RAG：注入了推理链知识库）
{ans_b[:1000]}

请对两个答案从以下五个推理维度评分（每项1-10）：
- 步骤完整性：推理步骤数量和目标是否准确
- 决策逻辑性：因果链和科学判断是否正确
- 参数精确性：关键数值是否准确引用（无则N/A=5）
- 链式连贯性：步骤间承接关系是否正确描述
- 幻觉抑制率：是否引入了参考答案外的虚构内容（10=无幻觉）

总分 = 五项均值（参数精确性若N/A则取其余四项均值）

【严格要求】只输出以下JSON格式，不含任何其他文字或Markdown：
{{"A":{{"步骤完整性":X,"决策逻辑性":X,"参数精确性":X,"链式连贯性":X,"幻觉抑制率":X,"总分":X,"核心问题":"一句话指出答案A最大的推理缺陷"}},"B":{{"步骤完整性":X,"决策逻辑性":X,"参数精确性":X,"链式连贯性":X,"幻觉抑制率":X,"总分":X,"核心优势":"一句话指出答案B在推理上相比A的最大提升"}},"推理提升幅度":"RAG在{dim_name}维度相比基线的核心差异（一句话）"}}"""

        resp = self._call([{"role": "user", "content": prompt}], EVAL_MODEL, temp=0)
        try:
            m = re.search(r"\{[\s\S]*\}", resp)
            return json.loads(m.group()) if m else {}
        except Exception:
            return {}


# ╔══════════════════════════════════════════════════════════════╗
# ║  ⑦ 推理链专项定量指标                                           ║
# ╚══════════════════════════════════════════════════════════════╝
def reasoning_param_accuracy(answer: str, ref_nums: List[str]) -> Optional[float]:
    """工艺参数数值准确率（推理链中的具体数值）"""
    ref_set = nums_in(" ".join(ref_nums))
    if not ref_set:
        return None
    ans_set = nums_in(answer)
    return len(ref_set & ans_set) / len(ref_set)

def reasoning_step_coverage(answer: str, step_goals: List[str]) -> Optional[float]:
    """
    推理步骤覆盖率：答案中提到了几个推理步骤目标
    step_goals: 每步目标的文字列表
    """
    if not step_goals:
        return None
    covered = 0
    for goal in step_goals:
        # 提取核心词（jieba取前3个有意义词）作为命中依据
        kws = [w for w in jieba.cut(goal) if len(w) > 1][:3]
        if any(k in answer for k in kws):
            covered += 1
    return covered / len(step_goals)

def hallucination_score_proxy(answer: str, ref: str) -> float:
    """
    幻觉代理指标：基于TF-IDF余弦相似度
    分数越高说明答案越贴近参考（幻觉越少）
    """
    try:
        vect = TfidfVectorizer(min_df=1)
        ref_tok = " ".join(jieba.cut(ref))
        ans_tok = " ".join(jieba.cut(answer))
        mat = vect.fit_transform([ref_tok, ans_tok])
        sim = cosine_similarity(mat[0:1], mat[1:2])[0][0]
        return float(sim)
    except Exception:
        return 0.0


# ╔══════════════════════════════════════════════════════════════╗
# ║  ⑧ 报告生成（JSON + CSV + 可视化雷达图 + 柱状图）               ║
# ╚══════════════════════════════════════════════════════════════╝
class ReasoningReporter:
    def __init__(self, out_dir: str):
        self.d = Path(out_dir)
        self.d.mkdir(exist_ok=True)

    def save_all(self, results: List[Dict], ts: str) -> Dict[str, Path]:
        paths = {}
        paths["json"]  = self._save_json(results, ts)
        paths["csv"]   = self._save_csv(results, ts)
        try:
            paths["radar"] = self._save_radar(results, ts)
            paths["bar"]   = self._save_bar(results, ts)
        except Exception as e:
            print(f"  ⚠ 图表生成失败：{e}")
        return paths

    def _save_json(self, results: List[Dict], ts: str) -> Path:
        p = self.d / f"reasoning_detail_{ts}.json"
        safe = []
        for r in results:
            r2 = {k: v for k, v in r.items() if k != "chain_ref"}
            safe.append(r2)
        p.write_text(json.dumps(safe, ensure_ascii=False, indent=2), "utf-8")
        return p

    def _save_csv(self, results: List[Dict], ts: str) -> Path:
        p = self.d / f"reasoning_summary_{ts}.csv"
        cols = [
            "问题ID", "推理维度", "问题（前60字）", "来源文件",
            "基线_参数准确率", "RAG_参数准确率",
            "基线_步骤覆盖率", "RAG_步骤覆盖率",
            "基线_幻觉相似度", "RAG_幻觉相似度",
            "基线_LLM步骤完整性", "RAG_LLM步骤完整性",
            "基线_LLM决策逻辑性", "RAG_LLM决策逻辑性",
            "基线_LLM参数精确性", "RAG_LLM参数精确性",
            "基线_LLM链式连贯性", "RAG_LLM链式连贯性",
            "基线_LLM幻觉抑制率", "RAG_LLM幻觉抑制率",
            "基线_LLM总分", "RAG_LLM总分", "LLM总分提升",
            "检索命中源文件", "推理提升描述",
        ]

        def f(v): return f"{v:.3f}" if v is not None else "-"
        def diff(a, b):
            try:
                return f"{float(b)-float(a):+.2f}" if a not in (None, "-", "?") and b not in (None, "-", "?") else "-"
            except Exception:
                return "-"

        rows = []
        for r in results:
            j  = r.get("judge", {})
            a  = j.get("A", {}); b = j.get("B", {})
            rows.append({
                "问题ID":          r["id"],
                "推理维度":        REASONING_DIM_LABELS.get(r["type"], r["type"]),
                "问题（前60字）":  r["q"][:60],
                "来源文件":        r["fn"],
                "基线_参数准确率": f(r.get("bn")),
                "RAG_参数准确率":  f(r.get("rn")),
                "基线_步骤覆盖率": f(r.get("bs")),
                "RAG_步骤覆盖率":  f(r.get("rs")),
                "基线_幻觉相似度": f(r.get("bh")),
                "RAG_幻觉相似度":  f(r.get("rh")),
                "基线_LLM步骤完整性": a.get("步骤完整性", "-"),
                "RAG_LLM步骤完整性":  b.get("步骤完整性", "-"),
                "基线_LLM决策逻辑性": a.get("决策逻辑性", "-"),
                "RAG_LLM决策逻辑性":  b.get("决策逻辑性", "-"),
                "基线_LLM参数精确性": a.get("参数精确性", "-"),
                "RAG_LLM参数精确性":  b.get("参数精确性", "-"),
                "基线_LLM链式连贯性": a.get("链式连贯性", "-"),
                "RAG_LLM链式连贯性":  b.get("链式连贯性", "-"),
                "基线_LLM幻觉抑制率": a.get("幻觉抑制率", "-"),
                "RAG_LLM幻觉抑制率":  b.get("幻觉抑制率", "-"),
                "基线_LLM总分":    a.get("总分", "-"),
                "RAG_LLM总分":     b.get("总分", "-"),
                "LLM总分提升":     diff(a.get("总分"), b.get("总分")),
                "检索命中源文件":  "✅" if r.get("hit") else "❌",
                "推理提升描述":    j.get("推理提升幅度", "")[:80],
            })

        with open(p, "w", newline="", encoding="utf-8-sig") as f_:
            w = csv.DictWriter(f_, fieldnames=cols)
            w.writeheader(); w.writerows(rows)
        return p

    def _save_radar(self, results: List[Dict], ts: str) -> Path:
        """雷达图：5维度基线 vs RAG 对比"""
        plt.rcParams.update({
            "font.sans-serif": ["SimHei", "PingFang SC", "Arial Unicode MS"],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
        })

        dims = ["步骤完整性", "决策逻辑性", "参数精确性", "链式连贯性", "幻觉抑制率"]

        def avg_dim(sub, dim):
            vals = []
            for r in results:
                v = r.get("judge", {}).get(sub, {}).get(dim)
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    pass
            return safe_mean(vals) or 0

        base_vals = [avg_dim("A", d) for d in dims]
        rag_vals  = [avg_dim("B", d) for d in dims]

        # 闭合雷达图
        n = len(dims)
        angles = [2 * np.pi * i / n for i in range(n)] + [0]
        base_vals_c = base_vals + [base_vals[0]]
        rag_vals_c  = rag_vals  + [rag_vals[0]]

        fig, ax = plt.subplots(1, 1, figsize=(7, 7),
                               subplot_kw=dict(projection="polar"))
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), dims, fontsize=11)
        ax.set_ylim(0, 10)
        ax.set_yticks([2, 4, 6, 8, 10])
        ax.set_yticklabels(["2", "4", "6", "8", "10"], fontsize=8, color="gray")

        RED, GRN = "#E74C3C", "#27AE60"
        ax.plot(angles, base_vals_c, color=RED, linewidth=2, linestyle="--",
                label="基线（无知识库）")
        ax.fill(angles, base_vals_c, color=RED, alpha=0.15)

        ax.plot(angles, rag_vals_c, color=GRN, linewidth=2.5, linestyle="-",
                label="RAG（含知识库）")
        ax.fill(angles, rag_vals_c, color=GRN, alpha=0.2)

        # 标注数值
        for angle, bv, rv in zip(angles[:-1], base_vals, rag_vals):
            ax.text(angle, bv + 0.5, f"{bv:.1f}", color=RED, fontsize=8, ha="center")
            ax.text(angle, rv + 0.5, f"{rv:.1f}", color=GRN, fontsize=8, ha="center")

        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.12), fontsize=10, ncol=2)
        ax.set_title("推理链能力五维雷达图\n（RAG vs 基线）",
                     fontsize=13, fontweight="bold", pad=20)
        ax.grid(color="gray", linestyle="--", alpha=0.4)

        plt.tight_layout()
        p = self.d / f"reasoning_radar_{ts}.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        return p

    def _save_bar(self, results: List[Dict], ts: str) -> Path:
        """柱状图：按推理维度类型分组的LLM总分对比"""
        plt.rcParams.update({
            "font.sans-serif": ["SimHei", "PingFang SC", "Arial Unicode MS"],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
        })

        q_types = list(REASONING_DIM_LABELS.keys())
        labels  = [REASONING_DIM_LABELS[t] for t in q_types]

        def type_avg(sub, qtype):
            vals = []
            for r in results:
                if r["type"] == qtype:
                    v = r.get("judge", {}).get(sub, {}).get("总分")
                    try:
                        vals.append(float(v))
                    except (TypeError, ValueError):
                        pass
            return safe_mean(vals) or 0

        base_scores = [type_avg("A", t) for t in q_types]
        rag_scores  = [type_avg("B", t) for t in q_types]
        diffs       = [r - b for b, r in zip(base_scores, rag_scores)]

        x  = np.arange(len(q_types))
        w  = 0.33
        RED, GRN, BLU = "#E74C3C", "#27AE60", "#3498DB"

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9),
                                        gridspec_kw={"height_ratios": [2.5, 1]})
        fig.suptitle("推理链评测 —— 各维度 LLM 评判总分（/10）",
                     fontsize=13, fontweight="bold", y=1.01)

        # 上图：基线 vs RAG 各维度得分
        b1 = ax1.bar(x - w/2, base_scores, w, label="基线（无知识库）",
                     color=RED, alpha=0.85, edgecolor="white", linewidth=0.8)
        b2 = ax1.bar(x + w/2, rag_scores,  w, label="RAG（含知识库）",
                     color=GRN, alpha=0.85, edgecolor="white", linewidth=0.8)

        for bar_grp in [b1, b2]:
            for b in bar_grp:
                h = b.get_height()
                ax1.text(b.get_x() + b.get_width()/2, h + 0.08, f"{h:.1f}",
                         ha="center", fontsize=9, color="#333")

        ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=10)
        ax1.set_ylim(0, 12.5); ax1.set_ylabel("LLM 评判总分（/10）", fontsize=11)
        ax1.legend(fontsize=10); ax1.grid(axis="y", alpha=0.3, linestyle="--")

        # 下图：提升幅度（差值）
        colors = [GRN if d >= 0 else RED for d in diffs]
        ax2.bar(x, diffs, color=colors, alpha=0.85, edgecolor="white", linewidth=0.8)
        ax2.axhline(0, color="black", linewidth=0.8)
        for xi, d in zip(x, diffs):
            ax2.text(xi, d + (0.05 if d >= 0 else -0.15), f"{d:+.1f}",
                     ha="center", fontsize=9, color="#333")
        ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=10)
        ax2.set_ylabel("RAG 相对基线提升", fontsize=10)
        ax2.set_title("各维度提升幅度", fontsize=11)
        ax2.grid(axis="y", alpha=0.3, linestyle="--")

        red_p  = mpatches.Patch(color=GRN, label="正向提升")
        blue_p = mpatches.Patch(color=RED, label="负向变化")
        ax2.legend(handles=[red_p, blue_p], fontsize=9)

        plt.tight_layout()
        p = self.d / f"reasoning_bar_{ts}.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        return p

    def print_summary(self, results: List[Dict]):
        """终端打印推理链评测汇总报告"""
        def avg_dim(sub, dim):
            vals = []
            for r in results:
                v = r.get("judge", {}).get(sub, {}).get(dim)
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    pass
            return safe_mean(vals)

        sep = "═" * 62
        print(f"\n{sep}")
        print(f"  📊 推理链专项评测报告  （共 {len(results)} 道推理题）")
        print(sep)

        docs_cnt = len(set(r["fn"] for r in results))
        print(f"  覆盖文档数: {docs_cnt}  |  问题类型: 5种推理维度")
        print(f"  {'─' * 58}")

        # 五维度汇总
        dims = ["步骤完整性", "决策逻辑性", "参数精确性", "链式连贯性", "幻觉抑制率"]
        print(f"\n  {'推理维度':<14} {'基线':^8} {'RAG':^8} {'提升':^8} 变化条")
        print(f"  {'─' * 58}")
        for dim in dims:
            bv = avg_dim("A", dim)
            rv = avg_dim("B", dim)
            if bv is None or rv is None:
                continue
            diff = rv - bv
            sym  = "↑" if diff > 0 else "↓"
            bar  = "█" * max(0, int(abs(diff) / 0.3))
            print(f"  {dim:<14} {bv:^8.2f} {rv:^8.2f} {sym}{abs(diff):.2f}    {bar}")

        # 整体LLM总分
        bl_total = avg_dim("A", "总分")
        rl_total = avg_dim("B", "总分")
        print(f"\n  整体LLM推理总分：基线={bl_total:.2f}/10  RAG={rl_total:.2f}/10  "
              f"{'↑' if rl_total > bl_total else '↓'}{abs(rl_total - bl_total):.2f}")

        # 定量指标均值
        bn = safe_mean([r.get("bn") for r in results])
        rn = safe_mean([r.get("rn") for r in results])
        bs = safe_mean([r.get("bs") for r in results])
        rs = safe_mean([r.get("rs") for r in results])
        bh = safe_mean([r.get("bh") for r in results])
        rh = safe_mean([r.get("rh") for r in results])

        print(f"\n  定量指标（自动计算）：")
        print(f"  {'─' * 58}")
        if bn is not None:
            print(f"  参数数值准确率：基线={bn:.3f} → RAG={rn:.3f}  {'+' if rn>bn else ''}{rn-bn:.3f}")
        if bs is not None:
            print(f"  推理步骤覆盖率：基线={bs:.3f} → RAG={rs:.3f}  {'+' if rs>bs else ''}{rs-bs:.3f}")
        if bh is not None:
            print(f"  语义相似度(反幻觉)：基线={bh:.3f} → RAG={rh:.3f}  {'+' if rh>bh else ''}{rh-bh:.3f}")

        # 检索命中率
        hit_rate = sum(1 for r in results if r.get("hit")) / max(len(results), 1)
        print(f"\n  检索命中源文件率：{hit_rate:.1%}（{sum(1 for r in results if r.get('hit'))}/{len(results)} 题）")

        # 按维度分组
        print(f"\n  按推理维度（LLM总分）：")
        print(f"  {'─' * 58}")
        for qtype, label in REASONING_DIM_LABELS.items():
            tr = [r for r in results if r["type"] == qtype]
            if not tr:
                continue
            ta = avg_dim("A", "总分") if not tr else safe_mean(
                [r.get("judge", {}).get("A", {}).get("总分") for r in tr if r.get("judge",{}).get("A",{}).get("总分") is not None])
            tb = safe_mean(
                [r.get("judge", {}).get("B", {}).get("总分") for r in tr if r.get("judge",{}).get("B",{}).get("总分") is not None])
            if ta and tb:
                diff = tb - ta
                sym  = "↑" if diff > 0 else "↓"
                print(f"  {label:<12}  基线={ta:.2f} → RAG={tb:.2f}  {sym}{abs(diff):.2f}  (n={len(tr)})")

        print(f"\n{sep}\n")


# ╔══════════════════════════════════════════════════════════════╗
# ║  ⑨ 主流程                                                      ║
# ╚══════════════════════════════════════════════════════════════╝
def run(max_docs: int = None, max_qs: int = None, seed: int = 42):
    """
    Parameters
    ----------
    max_docs : int | None
        限制加载的文档数（调试用小数字，完整评测设 None）
    max_qs : int | None
        限制测试问题数（调试用小数字，完整评测设 None）
    seed : int
        随机种子（保证采样可复现）
    """
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    Path(OUT_DIR).mkdir(exist_ok=True)
    progress_file = Path(OUT_DIR) / f"progress_{ts}.json"

    print("\n" + "╔" + "═" * 58 + "╗")
    print(f"  🚀 RAG 推理链专项评测系统  {ts}")
    print(f"  主问答模型 : {MAIN_MODEL}")
    print(f"  评判模型   : {EVAL_MODEL}")
    print(f"  评测维度   : 步骤完整性 / 决策逻辑性 / 参数精确性")
    print(f"               链式连贯性 / 幻觉抑制率")
    print("╚" + "═" * 58 + "╝\n")

    # Step 1 ── 加载数据（只保留有推理链的文档）
    print("─── Step 1 / 5: 加载含推理链的数据集 ───")
    docs, names = load_data(DATA_DIR, max_docs)

    # Step 2 ── 构建推理链检索器
    print("\n─── Step 2 / 5: 构建推理链检索索引 ───")
    retriever = Retriever(docs, names)

    # Step 3 ── 生成推理链专项问题
    print("\n─── Step 3 / 5: 生成推理链专项测试题 ───")
    all_qs = ReasoningQuestionGenerator().run(docs, names)

    if max_qs and len(all_qs) > max_qs:
        random.seed(seed)
        random.shuffle(all_qs)
        questions = all_qs[:max_qs]
        print(f"  随机采样 {max_qs} / {len(all_qs)} 个问题（seed={seed}）")
    else:
        questions = all_qs

    print(f"  共生成 {len(questions)} 道推理链题，维度分布：")
    type_cnt = {}
    for q in questions:
        type_cnt[q["type"]] = type_cnt.get(q["type"], 0) + 1
    for t, c in type_cnt.items():
        print(f"    {REASONING_DIM_LABELS.get(t, t):<12}: {c} 道")

    # Step 4 ── 初始化组件
    llm      = LLM()
    reporter = ReasoningReporter(OUT_DIR)
    results  = []

    # Step 5 ── 逐题评测
    print(f"\n─── Step 4 / 5: 开始推理链评测（{len(questions)} 题）───\n")
    print(f"  预计 API 调用约 {len(questions) * 3} 次，请耐心等待...\n")

    for i, q in enumerate(questions, 1):
        dim_label = REASONING_DIM_LABELS.get(q["type"], q["type"])
        print(f"┌── [{i:02d}/{len(questions):02d}] ▶ {dim_label}")
        wrap_print(q["q"][:80] + ("…" if len(q["q"]) > 80 else ""))

        # ── 基线
        print("│   📤 [基线] 调用（无知识库）...")
        t0 = time.time()
        base_ans = llm.baseline(q["q"])
        print(f"│   ✓ 完成（{time.time()-t0:.0f}s），{len(base_ans)} 字")
        time.sleep(CALL_WAIT)

        # ── 检索推理链上下文
        ctxs = retriever.retrieve(q["q"], TOP_K)
        hit  = q["fn"] in [c["fn"] for c in ctxs]
        print(f"│   🔍 检索：{[c['fn'] for c in ctxs]}")
        print(f"│      命中：{'✅' if hit else '❌（检索未命中源文件）'}")

        # ── RAG
        print("│   📤 [RAG]  调用（含推理链知识库）...")
        t0 = time.time()
        rag_ans = llm.rag(q["q"], ctxs)
        print(f"│   ✓ 完成（{time.time()-t0:.0f}s），{len(rag_ans)} 字")
        time.sleep(CALL_WAIT)

        # ── 定量指标
        bn = reasoning_param_accuracy(base_ans, q.get("ref_nums", []))
        rn = reasoning_param_accuracy(rag_ans,  q.get("ref_nums", []))
        bs = reasoning_step_coverage(base_ans,  q.get("ref_kws", []))
        rs = reasoning_step_coverage(rag_ans,   q.get("ref_kws", []))
        bh = hallucination_score_proxy(base_ans, q["ref"])
        rh = hallucination_score_proxy(rag_ans,  q["ref"])

        def fmt(v): return f"{v:.2f}" if v is not None else "N/A"
        print(f"│   📈 参数准确率: 基线={fmt(bn)} → RAG={fmt(rn)}")
        print(f"│   📈 步骤覆盖率: 基线={fmt(bs)} → RAG={fmt(rs)}")
        print(f"│   📈 语义相似度: 基线={fmt(bh)} → RAG={fmt(rh)}")

        # ── LLM 推理链评判（5维度）
        print(f"│   ⚖️  LLM 推理链评判（{dim_label}）...")
        judge = llm.judge_reasoning(q["q"], base_ans, rag_ans, q["ref"], q["type"])
        a_s = judge.get("A", {}).get("总分", "?")
        b_s = judge.get("B", {}).get("总分", "?")
        tip = judge.get("推理提升幅度", "")[:60]
        print(f"│   ✓ LLM总分：基线={a_s}/10  RAG={b_s}/10")
        wrap_print(f"推理提升：{tip}")
        print("└" + "─" * 58)

        results.append({
            **{k: v for k, v in q.items() if k != "chain_ref"},
            "base_answer": base_ans,
            "rag_answer":  rag_ans,
            "ctxs": [{"fn": c["fn"], "score": round(c["score"], 4)} for c in ctxs],
            "hit": hit,
            "bn": bn, "rn": rn,
            "bs": bs, "rs": rs,
            "bh": bh, "rh": rh,
            "judge": judge,
        })

        # 实时保存进度
        safe = [{k: v for k, v in r.items() if k != "chain_ref"} for r in results]
        progress_file.write_text(json.dumps(safe, ensure_ascii=False, indent=2), "utf-8")
        time.sleep(CALL_WAIT)

    # Step 6 ── 生成报告
    print(f"\n─── Step 5 / 5: 生成推理链评测报告 ───")
    paths = reporter.save_all(results, ts)
    reporter.print_summary(results)

    print("📁 输出文件：")
    labels = {
        "json":  "详细记录 JSON",
        "csv":   "汇总表格 CSV",
        "radar": "五维雷达图 PNG",
        "bar":   "柱状对比图 PNG",
    }
    for k, path in paths.items():
        print(f"  [{labels.get(k, k)}]  {path.name}")
    print(f"  [进度备份]              {progress_file.name}")

    return results


# ╔══════════════════════════════════════════════════════════════╗
# ║  ⑩ 入口                                                        ║
# ╚══════════════════════════════════════════════════════════════╝
if __name__ == "__main__":
    # ┌─────────────────────────────────────────────────────────┐
    # │  调试模式（快速验证）：max_docs=2, max_qs=8              │
    # │  中等规模：          max_docs=10, max_qs=40             │
    # │  完整评测：          max_docs=None, max_qs=None         │
    # │                                                          │
    # │  每篇文档约生成 5~10 道推理链题                            │
    # │  每题 3 次 API 调用（基线+RAG+评判）                      │
    # └─────────────────────────────────────────────────────────┘
    run(
        max_docs=None,   # ← 先用少量文档验证
        max_qs=10,    # ← 先测少量题
    )
