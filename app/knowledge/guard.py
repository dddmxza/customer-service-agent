"""知识库写入侧管控。

用途：data/raw/ 的内容进入索引之前，先过一遍准入检查。
这是投毒载荷进入系统的唯一入口 —— 项目原先在这条链路上没有任何校验。

定位：不是完整的审核系统，是一道"能看见"的闸门。
真正的内容审核要靠人工评审 + 白名单作者，不是正则。
"""

import re
from pathlib import Path
from typing import List, Tuple

_SUSPICIOUS_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("疑似指令注入", re.compile(
        r"忽略(上面|以上|之前|前面).{0,6}(指令|规则|提示)|"
        r"(最高|更高)优先级|以此为准|以本(段|文件|节)为准|"
        r"覆盖.{0,6}(规则|设定|提示词)|"
        r"system\s*prompt|系统提示词"
    )),
    ("疑似索取系统信息", re.compile(
        r"(说出|输出|告诉我|打印).{0,10}(系统提示|prompt|指令|密码|密钥|api[_\s]?key)"
    )),
    ("疑似引导外部联系方式", re.compile(
        r"拨打\s*(400|95|1[3-9])\d|加(微信|QQ|群)|联系.{0,6}(客服|专员).{0,6}(电话|微信)"
    )),
]

_SOURCE_WHITELIST_SUFFIX = {".md", ".txt"}


def scan_text(text: str, source: str = "") -> List[str]:
    """扫描单篇内容，返回命中的问题描述列表（空列表 = 通过）。"""
    findings = []
    for label, pat in _SUSPICIOUS_PATTERNS:
        m = pat.search(text)
        if m:
            findings.append(f"{label}：命中 {m.group(0)!r}")
    return findings


def scan_file(path: Path) -> List[str]:
    """扫描单个文件。文件后缀不在白名单内也记为问题。"""
    findings = []
    if path.suffix.lower() not in _SOURCE_WHITELIST_SUFFIX:
        findings.append(f"非预期文件类型：{path.suffix}")

    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as e:
        return findings + [f"读取失败：{e}"]

    findings.extend(scan_text(text, str(path)))
    return findings


def scan_dir(raw_dir: Path, strict: bool = False) -> bool:
    """扫描整个 data/raw/。返回是否全部通过。

    strict=False（默认）：只报告，不阻断 —— 便于先观察现有语料会命中什么。
    strict=True：有任何命中就返回 False，调用方应中止建库。
    """
    all_clear = True
    files = sorted(p for p in raw_dir.rglob("*") if p.is_file())

    if not files:
        print(f"  [guard] {raw_dir} 下没有文件")
        return True

    for path in files:
        findings = scan_file(path)
        if findings:
            all_clear = False
            print(f"  [guard] 建库前检查未通过：{path.name}")
            for f in findings:
                print(f"           - {f}")
        else:
            print(f"  [guard] 建库前检查通过：{path.name}")

    return all_clear