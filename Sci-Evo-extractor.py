#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import requests
import os
import re
from typing import Dict, Any


class PaperJSONConverter:

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_base = "https://api.deepseek.com"
        self.model = "deepseek-reasoner"

    def extract_text_from_json(self, json_data: Dict[str, Any], max_length: int = 50000) -> str:
        text_parts = []
        try:
            if "pdf_info" in json_data:
                for page in json_data["pdf_info"]:
                    if "preproc_blocks" in page:
                        for block in page["preproc_blocks"]:
                            if "lines" in block:
                                for line in block["lines"]:
                                    if "spans" in line:
                                        for span in line["spans"]:
                                            if span.get("type") == "text" and "content" in span:
                                                text_parts.append(span["content"])
                            if len(" ".join(text_parts)) > max_length:
                                break
                    if len(" ".join(text_parts)) > max_length:
                        break
        except Exception as e:
            print(f"提取文本时出错: {e}")

        full_text = " ".join(text_parts)
        full_text = re.sub(r'\s+', ' ', full_text).strip()
        return full_text[:max_length]

    def build_conversion_prompt(self, paper_text: str) -> str:
        prompt = f"""你是一个专业的科研论文分析专家。请仔细阅读以下论文内容，严格按照指定JSON格式提取全部关键信息，尤其要完整提取实验脉络。

论文内容：
{paper_text}

请严格按照以下JSON格式输出，所有字段保持不变，根据论文内容填写：

{{
  "01_初始信息": {{
    "目标名称": "文章题目",
    "输入数据": {{
      "原材料": "列出论文中所有原材料",
      "测试仪器": "列出论文中所有测试仪器"
    }},
    "用户意图": "总结论文的研究目标"
  }},

  "02_思考过程": [
    {{
      "步骤序号": 1,
      "目前进展": "该步骤之前的研究或制备进展",
      "待解决问题": "该步骤要解决的问题",
      "实验目的(解决方案）": "该步骤的实验目的或解决方案",
      "实验仪器": {{
        "实验方法": "实验/仿真",
        "制备仪器": "该步骤中用到的制备仪器。注意和测试仪器不同。"
      }},
      "原料": "该步骤使用的原料",
      "配比": "原料的配比",
      "加料顺序": "加料顺序",
      "混合方式": "混合方式",
      "其它参数": {{
        "": "",
        "": "",
        "": ""
      }},
      "实验结果": "该步骤的实验结果",
      "有效": true
    }}
  ],

  "03_性能分析": [
    {{
      "性能序号": 1,
      "测试原因": "为什么要测试该性能",
      "测试方法": "实验/仿真",
      "测试性能": {{
        "测试性能名称": "性能名称",
        "测试性能类型": "按照固化反应特性、固化结构、力学性能、热学性能进行分类。其中固化反应特性,包括:DSC,凝胶时间，黏度等；固化结构,包括:SEM,FTIR,SAXS等；力学性能（拉伸、剪切等）；热学性能（热分解温度,DMA,玻璃化转变温度等）"
      }},
      "测试仪器": "测试仪器名称",
      "测试对象": "测试的样品或对象。如果出现固化条件或固化工艺之类，应该放入测试对象中",
      "测试标准": {{
        "测试标准": "测试标准名称",
        "测试尺寸": "样品尺寸",
        "测试工艺": "[第一步][第二步]列出具体测试工艺流程路线",
        "测试条件": "测试条件，如温度、湿度等"
      }},
      "其它参数": {{
        "": "",
        "": ""
      }},
      "测试结果": "测试得到的结果",
      "机理阐述": "对测试结果的规律总结或机理阐述",
      "有效": true
    }}
  ],
  "04_应用领域": [
    {{
      "应用序号": 1,
      "应用场景": "应用场景",
      "测试性能": "测试性能",
      "测试仪器": "测试仪器名称",
      "测试对象": "测试的样品或对象",
      "测试标准": {{
        "测试工艺": "[第一步][第二步]列出具体测试工艺流程路线",
        "测试条件": "测试条件，如温度、湿度等"
      }},
      "其它参数": {{
        "": "",
        "": ""
      }},
      "测试结果": "测试得到的结果",
      "机理阐述": "对测试结果的规律总结或机理阐述",
      "有效": true
    }}
  ],
  "05_成功验证": {{
    "指标": {{
      "指标1": {{
        "数值": "具体数值",
        "单位": "单位",
        "方法": "测试方法",
        "解读": "对该指标的解读和意义"
      }}
    }}
  }},

  "06_最终结论和未来展望": {{
    "最终结论和未来展望": "论文主要结论+展望"
  }}
}}

要求：
1. "02_思考过程"按实验顺序完整列出所有制备步骤，每步单独一个对象。从实验部分提取，按什么原材照什么配比、什么顺序、什么工艺制备样品，分步叙述。只要原文实验步骤中出现小标题，自动划分为下一个实验步骤。
2. "03_性能分析"列出论文中所有性能测试，每项单独一个对象。分析文章中对该样品的全部性能测试。只要原文中在性能测试部分出现二级标题，自动划分为一个性能。若FTIR出现在二级标题中,需要单独划分
3. "04_应用领域"列出论文中所有应用，每项单独一个对象。
4. "05_成功验证"提取所有有具体数值的测试结果
5. 参数字段若论文中有其他关键参数（如压力、固化时间等），可替换或新增字段名
6. "03_性能分析"的"测试性能类型"处，根据测试仪器来判断测试性能类型,并按照固化反应特性、固化结构、力学性能、热学性能等进行分类。若无法分类,归为其他性能。其中固化反应特性：包括DSC,凝胶时间、黏度等。固化结构：包括SEM,FTIR,SAXS等。力学性能：拉伸、剪切等。热学性能：热分解温度,DMA,玻璃化转变温度,热重分析仪TG等。老化性能：如果出现耐受、老化、适应等词,不论测试仪器,统一归为老化性能。其他性能：接触角、润湿性、光学透明性等
5. 找不到的信息就填写"未明确"
6. 只输出JSON，不要有任何其他内容，不要有markdown代码块标记
"""
        return prompt

    def call_deepseek_api(self, prompt: str, max_tokens: int = 8000) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个专业的科研论文分析专家，擅长提取和组织论文中的关键信息。请严格按照要求的JSON格式输出，不要添加任何额外的解释或markdown标记。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.3,
            "response_format": {"type": "json_object"}
        }
        try:
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=data,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                raise Exception("API返回格式错误")
        except requests.exceptions.RequestException as e:
            raise Exception(f"API调用失败: {e}")

    def clean_json_response(self, response: str) -> str:
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*', '', response)
        return response.strip()

    def convert(self, input_file: str, output_file: str) -> Dict[str, Any]:
        with open(input_file, 'r', encoding='utf-8') as f:
            original_data = json.load(f)

        paper_text = self.extract_text_from_json(original_data)

        prompt = self.build_conversion_prompt(paper_text)
        response = self.call_deepseek_api(prompt)

        cleaned_response = self.clean_json_response(response)
        try:
            converted_data = json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            print(f"原始响应: {cleaned_response[:500]}...")
            raise

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, ensure_ascii=False, indent=2)

        return converted_data


def main():
    API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

    # ===== 在此配置输入输出路径 =====
    input_file = ""
    output_file = ""
    # ================================

    converter = PaperJSONConverter(API_KEY)
    converter.convert(input_file, output_file)


if __name__ == "__main__":
    main()
