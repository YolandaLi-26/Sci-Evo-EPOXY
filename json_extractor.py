

import json
import re
import sys
import os
from collections import defaultdict


INPUT_JSON = ""
OUTPUT_JSON = ""


def extract_text_from_block(block: dict) -> str:
    """递归提取一个 block 的所有文本内容（拼接为单字符串）。"""
    parts = []
    if "lines" in block:
        for line in block["lines"]:
            for span in line.get("spans", []):
                if span.get("type") == "text":
                    parts.append(span["content"])
    if "blocks" in block:
        for sub in block["blocks"]:
            parts.append(extract_text_from_block(sub))
    return " ".join(p for p in parts if p.strip())


def collect_all_blocks(data: dict) -> list:
    """
    从 pdf_info 的所有页面中提取全部 block，
    返回列表，每项为 (page_index, block_type, text)。
    """
    pages = data.get("pdf_info", [])
    result = []
    for page_i, page in enumerate(pages):
        for block in page.get("preproc_blocks", []):
            btype = block.get("type", "")
            text = extract_text_from_block(block).strip()
            if text:
                result.append({
                    "page": page_i + 1,
                    "type": btype,
                    "text": text,
                    "index": block.get("index", -1),
                })
    return result


def find_title(blocks: list) -> str:
    """取第一个 title 块作为论文标题。"""
    for b in blocks:
        if b["type"] == "title" and len(b["text"]) > 20:
            return b["text"].strip()
    return "未知标题"


def find_authors(blocks: list) -> list:
    """
    启发式：标题后紧跟的短文本块（无句号，长度<80）通常是作者名。
    """
    authors = []
    title_seen = False
    for b in blocks:
        if not title_seen:
            if b["type"] == "title" and len(b["text"]) > 20:
                title_seen = True
            continue
        text = b["text"].strip()
        # 作者行特征：较短、含数字上标、无长句
        if b["type"] in ("text", "list") and len(text) < 120:
            # 过滤掉摘要/关键词/机构等
            if any(kw in text.lower() for kw in
                   ["abstract", "keyword", "introduction", "department",
                    "university", "email", "correspondence", "©", "doi",
                    "received", "accepted", "published"]):
                break
            # 去掉数字上标后仍有内容
            clean = re.sub(r"\d+", "", text).strip(" ,;.")
            if clean and 2 < len(clean) < 80:
                authors.append(clean)
        elif b["type"] == "title":
            break
    return authors if authors else ["未知作者"]


def find_abstract(blocks: list) -> str:
    """定位 Abstract / 摘要 节。"""
    found = False
    parts = []
    for b in blocks:
        if b["type"] == "title" and re.search(r"abstract|摘要", b["text"], re.I):
            found = True
            continue
        if found:
            if b["type"] == "title":
                break
            parts.append(b["text"])
        # 部分论文把摘要直接跟在 Abstract 标题块的下一个 text 块
    return " ".join(parts).strip() if parts else ""


def find_keywords(blocks: list) -> list:
    """定位关键词。"""
    for b in blocks:
        if b["type"] in ("title", "text") and re.search(
                r"key\s*word|关键词", b["text"], re.I):
            # 找下一个 text 块
            idx = blocks.index(b)
            for nb in blocks[idx + 1: idx + 3]:
                if nb["type"] == "text" and len(nb["text"]) < 300:
                    return [k.strip() for k in
                            re.split(r"[;,，；、]", nb["text"]) if k.strip()]
    return []


def find_sections(blocks: list) -> list:
    """
    提取所有章节标题（title 块且含数字编号或常见章节词）。
    返回 [(section_title, [text_blocks])]
    """
    sections = []
    current_title = None
    current_texts = []

    section_pattern = re.compile(
        r"^\s*(\d+[\.\s|]|introduction|experimental|result|discussion|"
        r"conclusion|method|material|characteriz|synthes|preparat|"
        r"实验|结果|讨论|结论|方法|材料|合成|制备|表征)",
        re.I,
    )

    for b in blocks:
        if b["type"] == "title" and section_pattern.match(b["text"]):
            if current_title is not None:
                sections.append((current_title, current_texts))
            current_title = b["text"].strip()
            current_texts = []
        elif current_title is not None and b["type"] in ("text", "list"):
            current_texts.append(b["text"])

    if current_title is not None:
        sections.append((current_title, current_texts))

    return sections


def extract_research_objective(abstract: str, intro_texts: list) -> dict:
    """从摘要和引言中推断研究目标和背景。"""
    # 用摘要前两句作为目标描述
    sentences = re.split(r"(?<=[.!?])\s+", abstract)
    objective = " ".join(sentences[:3]) if sentences else abstract[:300]

    # 从引言找问题/挑战句
    problem_keywords = re.compile(
        r"however|despite|problem|challenge|limit|drawback|"
        r"despite|improve|lack|需要|问题|挑战|限制|不足|然而",
        re.I,
    )
    problem_sentences = []
    for text in intro_texts:
        for sent in re.split(r"(?<=[.!?])\s+", text):
            if problem_keywords.search(sent) and len(sent) > 30:
                problem_sentences.append(sent.strip())
                if len(problem_sentences) >= 2:
                    break
        if len(problem_sentences) >= 2:
            break

    return {
        "研究目标描述": objective,
        "研究背景与问题": " ".join(problem_sentences) if problem_sentences else "见摘要",
    }


def extract_materials(blocks: list) -> list:
    """提取材料/试剂列表。"""
    materials = []
    in_materials = False
    for b in blocks:
        if b["type"] == "title" and re.search(r"material|试剂|原料|reagent", b["text"], re.I):
            in_materials = True
            continue
        if in_materials:
            if b["type"] == "title":
                break
            # 按逗号/句号切分出化学品名
            candidates = re.split(r"[,;，；。.]", b["text"])
            for c in candidates:
                c = c.strip()
                if 3 < len(c) < 80:
                    materials.append(c)
    return materials[:20]  # 最多取20条


def extract_experimental_steps(sections: list) -> list:
    """
    将实验章节解析为步骤列表。
    每个 subsection 对应一个步骤。
    """
    steps = []
    step_no = 1

    exp_pattern = re.compile(
        r"(experimental|method|preparat|synthes|characteriz|analys|测试|"
        r"制备|合成|表征|实验|分析|测量|measurement|evaluat|test)",
        re.I,
    )

    for sec_title, sec_texts in sections:
        if not exp_pattern.search(sec_title):
            continue
        full_text = " ".join(sec_texts)
        if not full_text.strip():
            continue

        # 推断工具/仪器
        tool = infer_tool(sec_title, full_text)
        # 推断动作类型
        action_type = infer_action_type(sec_title, full_text)
        # 推断观察/结果
        observation = infer_observation(full_text)

        steps.append({
            "步骤序号": step_no,
            "步骤名称": sec_title.strip(),
            "动作类型": action_type,
            "工具与仪器": tool,
            "实验描述": full_text[:500].strip(),
            "关键参数": extract_key_params(full_text),
            "观察/预期结果": observation,
        })
        step_no += 1

    return steps


def infer_tool(title: str, text: str) -> dict:
    """从文本中识别仪器/方法名称。"""
    tool_patterns = {
        "TEM": r"\bTEM\b|transmission electron micro",
        "SEM|FE-SEM": r"\bSEM\b|FE-SEM|scanning electron micro",
        "FTIR": r"\bFTIR\b|fourier.transform infrared|infrared spectro",
        "TGA": r"\bTGA\b|thermogravimetric|thermal analysis",
        "XRD": r"\bXRD\b|x-ray diffract",
        "DSC": r"\bDSC\b|differential scanning calori",
        "UV-Vis": r"\bUV.Vis\b|ultraviolet.visible",
        "NMR": r"\bNMR\b|nuclear magnetic resonance",
        "拉伸试验机": r"tensile|universal.*test|拉伸|万能试验",
        "搭接剪切测试": r"lap shear|剪切|adhesion test",
        "Kissinger法": r"kissinger",
        "FWO法": r"flynn.*wall.*ozawa|FWO",
        "Arrhenius方程": r"arrhenius",
    }
    found = []
    combined = title + " " + text
    for name, pattern in tool_patterns.items():
        if re.search(pattern, combined, re.I):
            found.append(name)
    return {"名称": "、".join(found) if found else "未识别", "参数来源": "文本推断"}


def infer_action_type(title: str, text: str) -> str:
    combined = title + " " + text
    if re.search(r"prepar|synthes|制备|合成|grafted|mix|blend|disperse", combined, re.I):
        return "湿实验-样品制备"
    if re.search(r"TEM|SEM|FTIR|TGA|XRD|DSC|spectro|microscop|表征|characteriz", combined, re.I):
        return "湿实验-表征测试"
    if re.search(r"tensile|shear|mechanical|adhesion|拉伸|剪切|力学", combined, re.I):
        return "湿实验-力学性能测试"
    if re.search(r"thermal|TGA|degradation|热稳定|热重|热分解", combined, re.I):
        return "湿实验-热性能测试"
    if re.search(r"calcul|model|equation|kinetic|计算|模型|方程", combined, re.I):
        return "干实验-理论计算"
    return "湿实验-实验操作"


def infer_observation(text: str) -> str:
    """提取实验结果/观察语句（含improve/increase/show等动词的句子）。"""
    result_kw = re.compile(
        r"show|result|reveal|found|observ|indicat|demonstrat|"
        r"improv|increas|decreas|enhanc|显示|表明|发现|结果|观察|改善|增加|提高",
        re.I,
    )
    sentences = re.split(r"(?<=[.!?])\s+", text)
    picked = [s.strip() for s in sentences if result_kw.search(s) and len(s) > 30]
    return " ".join(picked[:3]) if picked else "见实验描述"


def extract_key_params(text: str) -> dict:
    """用正则从文本中提取关键数值参数。"""
    params = {}

    # 温度
    temp = re.findall(r"(\d+(?:\.\d+)?)\s*[°℃]?C\b", text)
    if temp:
        params["温度(°C)"] = temp[:3]

    # 时间
    time_m = re.findall(r"(\d+(?:\.\d+)?)\s*(min|h\b|hour|分钟|小时)", text, re.I)
    if time_m:
        params["时间"] = [f"{v} {u}" for v, u in time_m[:3]]

    # 质量/重量百分比
    wt = re.findall(r"(\d+(?:\.\d+)?)\s*(wt%|phr|%\s*by weight)", text, re.I)
    if wt:
        params["含量"] = [f"{v} {u}" for v, u in wt[:5]]

    # 速率
    speed = re.findall(r"(\d+(?:\.\d+)?)\s*(rpm|mm/min|mm min)", text, re.I)
    if speed:
        params["速率"] = [f"{v} {u}" for v, u in speed[:3]]

    # 波数
    wn = re.findall(r"(\d{3,4})\s*cm[−\-]?1", text)
    if wn:
        params["特征波数(cm⁻¹)"] = wn[:5]

    return params if params else {"说明": "见实验描述"}


def extract_results_metrics(sections: list) -> dict:
    """从结果章节中提取量化指标。"""
    metrics = {}
    result_pattern = re.compile(
        r"(result|evaluat|discussion|conclusion|结果|评价|讨论|结论|analysis|分析)", re.I
    )

    for sec_title, sec_texts in sections:
        if not result_pattern.search(sec_title):
            continue
        full_text = " ".join(sec_texts)

        # 提取百分比提升
        improvements = re.findall(
            r"(\d+(?:\.\d+)?)\s*%\s*(increase|improve|higher|enhance|提升|增加|改善|提高)",
            full_text, re.I,
        )
        if improvements:
            metrics.setdefault("性能提升百分比", [])
            for val, desc in improvements[:5]:
                metrics["性能提升百分比"].append(f"{val}% {desc}")

        # 提取活化能 kJ/mol
        ea = re.findall(r"(\d+(?:\.\d+)?)\s*kJ\s*/?\s*mol", full_text, re.I)
        if ea:
            metrics["活化能(kJ/mol)"] = ea[:5]

        # 提取温度指标
        t5 = re.findall(
            r"T(?:IDT|5|onset|d5|deg)?\s*[=of]*\s*(\d{2,3}(?:\.\d+)?)\s*°?C", full_text
        )
        if t5:
            metrics["降解温度(°C)"] = t5[:5]

        # 提取强度数值 MPa
        mpa = re.findall(r"(\d+(?:\.\d+)?)\s*MPa", full_text)
        if mpa:
            metrics["强度(MPa)"] = mpa[:5]

    return metrics


def extract_conclusion(sections: list, blocks: list) -> str:
    """提取结论章节内容。"""
    concl_pattern = re.compile(r"conclusion|summary|结论|总结", re.I)
    for sec_title, sec_texts in sections:
        if concl_pattern.search(sec_title):
            return " ".join(sec_texts)[:800].strip()

    # 备用：找最后一个 title 后的文本
    for b in reversed(blocks):
        if b["type"] == "title" and concl_pattern.search(b["text"]):
            idx = blocks.index(b)
            parts = [nb["text"] for nb in blocks[idx + 1: idx + 5] if nb["type"] == "text"]
            return " ".join(parts)[:800].strip()

    return ""


def extract_references_count(blocks: list) -> int:
    """粗略统计引用文献数量。"""
    ref_pattern = re.compile(r"\[\d+\]|\[ref|\bref\b", re.I)
    count = 0
    for b in blocks:
        if b["type"] in ("text", "list"):
            matches = ref_pattern.findall(b["text"])
            count += len(matches)
    nums = []
    for b in blocks:
        found = re.findall(r"\[(\d+)\]", b["text"])
        nums.extend(int(n) for n in found if n.isdigit())
    return max(nums) if nums else count


def build_experimental_trajectory(sections: list, blocks: list) -> list:
    """
    核心：将论文的实验章节转换为有序的实验步骤轨迹。
    """
    steps = extract_experimental_steps(sections)

    # 如果没有识别到步骤，降级处理：按主章节分组
    if not steps:
        step_no = 1
        for sec_title, sec_texts in sections:
            full_text = " ".join(sec_texts)
            if len(full_text) < 50:
                continue
            steps.append({
                "步骤序号": step_no,
                "步骤名称": sec_title.strip(),
                "动作类型": infer_action_type(sec_title, full_text),
                "工具与仪器": infer_tool(sec_title, full_text),
                "实验描述": full_text[:500].strip(),
                "关键参数": extract_key_params(full_text),
                "观察/预期结果": infer_observation(full_text),
            })
            step_no += 1

    return steps


def convert(raw_data: dict) -> dict:
    """将原始 PDF 解析 JSON 转换为规范数据集格式。"""

    blocks = collect_all_blocks(raw_data)

    # ── 基本元数据 ──
    title = find_title(blocks)
    authors = find_authors(blocks)
    abstract = find_abstract(blocks)
    keywords = find_keywords(blocks)

    # ── 章节结构 ──
    sections = find_sections(blocks)
    section_names = [s[0] for s in sections]

    # ── 引言内容 ──
    intro_texts = []
    for sec_title, sec_texts in sections:
        if re.search(r"introduction|引言|背景", sec_title, re.I):
            intro_texts = sec_texts
            break

    # ── 研究目标 ──
    research_obj = extract_research_objective(abstract, intro_texts)

    # ── 材料 ──
    materials = extract_materials(blocks)

    # ── 实验脉络（步骤列表）──
    trajectory = build_experimental_trajectory(sections, blocks)

    # ── 结果指标 ──
    result_metrics = extract_results_metrics(sections)

    # ── 结论 ──
    conclusion = extract_conclusion(sections, blocks)

    # ── 文献计量 ──
    ref_count = extract_references_count(blocks)

    # ── 组装规范数据集 ──
    dataset = {
        "01_初始请求": {
            "目标名称": title,
            "作者": authors,
            "输入数据/原料": materials if materials else ["见实验部分"],
            "用户意图": research_obj["研究目标描述"],
            "研究背景与问题": research_obj["研究背景与问题"],
            "关键词": keywords,
            "摘要": abstract[:600] if abstract else "未提取到摘要",
        },
        "02_实验脉络（智能体轨迹）": [
            {
                "步骤序号": step["步骤序号"],
                "步骤名称": step["步骤名称"],
                "思考": (
                    f"[背景] 研究目标：{research_obj['研究目标描述'][:80]}。"
                    f"[动作] {step['步骤名称']}。"
                    f"[预期] {step['观察/预期结果'][:100]}"
                ),
                "动作类型": step["动作类型"],
                "工具与仪器": step["工具与仪器"],
                "关键参数": step["关键参数"],
                "实验描述摘要": step["实验描述"][:300],
                "观察/预期结果": step["观察/预期结果"],
                "有效": True,
            }
            for step in trajectory
        ],
        "03_成功验证": {
            "验证技术": (
                "、".join(
                    s["工具与仪器"]["名称"]
                    for s in trajectory
                    if s["工具与仪器"]["名称"] != "未识别"
                )
                or "见实验章节"
            ),
            "量化指标": result_metrics if result_metrics else {"说明": "见结果章节"},
            "结论": conclusion[:600] if conclusion else "见论文结论章节",
            "引用文献数量（估计）": ref_count,
        },
        "04_章节结构概览": {
            "识别到的章节": section_names,
            "总页数（估计）": len(raw_data.get("pdf_info", [])),
            "总文本块数": len(blocks),
        },
    }

    return dataset


def main():
    input_path = INPUT_JSON

    # 自动生成输出路径（若 OUTPUT_JSON 未指定）
    if OUTPUT_JSON:
        output_path = OUTPUT_JSON
    else:
        base = os.path.splitext(input_path)[0]
        output_path = base + "_dataset.json"

    if not os.path.exists(input_path):
        print(f"错误：找不到输入文件 '{input_path}'")
        print("请在脚本顶部修改 INPUT_JSON 变量为正确的文件路径。")
        sys.exit(1)

    print(f"读取: {input_path}")
    with open(input_path, encoding="utf-8") as f:
        raw_data = json.load(f)

    print("转换中...")
    dataset = convert(raw_data)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"完成！输出: {output_path}")
    print(f"  - 论文标题: {dataset['01_初始请求']['目标名称'][:60]}")
    print(f"  - 实验步骤数: {len(dataset['02_实验脉络（智能体轨迹）'])}")
    print(f"  - 识别章节数: {len(dataset['04_章节结构概览']['识别到的章节'])}")


if __name__ == "__main__":
    main()
