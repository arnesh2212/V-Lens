# SPDX-License-Identifier: Apache-2.0
"""Alternative verbal routes to the same concept.

The sharpest objection to erasing a J-lens direction is that at late layers the
averaged Jacobian is close to the identity, so the direction is close to the
unembedding row of the word -- subtract it and of course the word stops coming
out.  The way to settle that is to ask for the same concept through a *different*
row of the unembedding matrix.

Chinese is the clean universal route: Qwen models are natively bilingual, every
COCO category has a common Chinese name, and its unembedding rows are entirely
disjoint from the English ones we erase.  If the model can still answer in
Chinese, only the English lexical route was cut and the concept survives; if the
Chinese answer dies too, the erased direction was carrying the concept, not the
spelling.

English near-synonyms cover a smaller set and are noisier (a synonym shares
subword structure and neighbourhood), so they are a secondary route.
"""
from __future__ import annotations

ZH = {
    "person": "人", "bicycle": "自行车", "car": "汽车", "motorcycle": "摩托车",
    "airplane": "飞机", "bus": "公交车", "train": "火车", "truck": "卡车",
    "boat": "船", "bench": "长椅", "bird": "鸟", "cat": "猫", "dog": "狗",
    "horse": "马", "sheep": "羊", "cow": "牛", "elephant": "大象",
    "bear": "熊", "zebra": "斑马", "giraffe": "长颈鹿", "backpack": "背包",
    "umbrella": "雨伞", "handbag": "手提包", "tie": "领带", "suitcase": "行李箱",
    "frisbee": "飞盘", "skis": "滑雪板", "snowboard": "单板", "kite": "风筝",
    "surfboard": "冲浪板", "bottle": "瓶子", "cup": "杯子", "fork": "叉子",
    "knife": "刀", "spoon": "勺子", "bowl": "碗", "banana": "香蕉",
    "apple": "苹果", "sandwich": "三明治", "orange": "橙子", "broccoli": "西兰花",
    "carrot": "胡萝卜", "pizza": "披萨", "donut": "甜甜圈", "cake": "蛋糕",
    "chair": "椅子", "couch": "沙发", "bed": "床", "toilet": "马桶",
    "tv": "电视", "laptop": "笔记本", "mouse": "鼠标", "remote": "遥控器",
    "keyboard": "键盘", "microwave": "微波炉", "oven": "烤箱",
    "toaster": "烤面包机", "sink": "水槽", "refrigerator": "冰箱",
    "book": "书", "clock": "时钟", "vase": "花瓶", "scissors": "剪刀",
    "teddy": "泰迪熊", "toothbrush": "牙刷",
}

SYN = {
    "person": "human", "bicycle": "bike", "car": "automobile",
    "motorcycle": "motorbike", "airplane": "aircraft", "bus": "coach",
    "train": "locomotive", "truck": "lorry", "boat": "vessel",
    "bench": "seat", "bird": "fowl", "cat": "feline", "dog": "canine",
    "horse": "equine", "sheep": "lamb", "cow": "cattle",
    "elephant": "pachyderm", "bear": "bruin", "couch": "sofa",
    "bottle": "flask", "cup": "mug", "bowl": "dish", "chair": "stool",
    "bed": "bunk", "tv": "television", "laptop": "notebook",
    "refrigerator": "fridge", "clock": "timepiece", "vase": "urn",
    "suitcase": "luggage", "handbag": "purse", "backpack": "rucksack",
    "umbrella": "parasol", "knife": "blade", "book": "volume",
    "sink": "basin", "oven": "stove", "toilet": "lavatory",
}


FR = {
    "person": "personne", "bicycle": "vélo", "car": "voiture",
    "motorcycle": "moto", "airplane": "avion", "bus": "autobus",
    "train": "train", "truck": "camion", "boat": "bateau", "bench": "banc",
    "bird": "oiseau", "cat": "chat", "dog": "chien", "horse": "cheval",
    "sheep": "mouton", "cow": "vache", "elephant": "éléphant", "bear": "ours",
    "zebra": "zèbre", "giraffe": "girafe", "backpack": "sac", "umbrella":
    "parapluie", "handbag": "sacoche", "tie": "cravate", "suitcase": "valise",
    "kite": "cerf", "bottle": "bouteille", "cup": "tasse", "fork":
    "fourchette", "knife": "couteau", "spoon": "cuillère", "bowl": "bol",
    "banana": "banane", "apple": "pomme", "sandwich": "sandwich", "orange":
    "orange", "broccoli": "brocoli", "carrot": "carotte", "pizza": "pizza",
    "donut": "beignet", "cake": "gâteau", "chair": "chaise", "couch":
    "canapé", "bed": "lit", "toilet": "toilette", "tv": "téléviseur",
    "laptop": "portable", "mouse": "souris", "remote": "télécommande",
    "keyboard": "clavier", "microwave": "micro", "oven": "four", "toaster":
    "grille", "sink": "évier", "refrigerator": "frigo", "book": "livre",
    "clock": "horloge", "vase": "vase", "scissors": "ciseaux", "teddy":
    "nounours", "toothbrush": "brosse", "surfboard": "planche",
    "skis": "skis", "snowboard": "snowboard", "frisbee": "frisbee",
}

#: Response channels that emit nothing lexical at all.  A digit key is the
#: second arbitrary-symbol channel: if the letter channel survived only because
#: "A" and "B" happen to be robust tokens, a digit key would not survive too.
ARBITRARY = {"letter": ("A", "B"), "digit": ("1", "2")}


def route_word(concept: str, route: str) -> str | None:
    """The surface form of ``concept`` on a given report route."""
    if route == "en":
        return concept
    if route == "zh":
        return ZH.get(concept)
    if route == "syn":
        return SYN.get(concept)
    if route == "fr":
        return FR.get(concept)
    raise ValueError(route)


#: Instruction appended so the model answers on the intended route.
ROUTE_CUE = {
    "en": "Answer with one word.",
    "zh": "\u7528\u4e00\u4e2a\u4e2d\u6587\u8bcd\u56de\u7b54\u3002",  # answer in one Chinese word
    "syn": "Answer with one word.",
    "fr": "R\u00e9pondez par un seul mot en fran\u00e7ais.",
}


def route_token_ids(tokenizer, concept: str, route: str) -> list[int]:
    """First-token ids for ``concept`` on ``route``.

    A leading space is right for English -- the word is emitted mid-sentence --
    but wrong for Chinese, where " \u725b" falls apart into byte fragments while
    "\u725b" is a single token.  Getting this backwards silently scores noise.
    """
    w = route_word(concept, route)
    if not w:
        return []
    forms = [w] if route == "zh" else [f" {w}", w, f" {w.capitalize()}"]
    ids = set()
    for f in forms:
        enc = tokenizer.encode(f, add_special_tokens=False)
        if enc:
            ids.add(enc[0])
    return sorted(ids)


def usable(concept: str, distractor: str, route: str, tokenizer) -> bool:
    """Both options exist on this route and take disjoint first-token sets.

    Without the disjointness check a margin between two words sharing a first
    token is identically zero and quietly pollutes the average.
    """
    ia = set(route_token_ids(tokenizer, concept, route))
    ib = set(route_token_ids(tokenizer, distractor, route))
    return bool(ia) and bool(ib) and not (ia & ib)
