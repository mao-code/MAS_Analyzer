from __future__ import annotations

import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import requests

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PACKAGE_ROOT / "data"

SEARCH_RETURN_N = 50
PRODUCT_WINDOW = 10

END_BUTTON = "buy now"
NEXT_PAGE = "next >"
PREV_PAGE = "< prev"
BACK_TO_SEARCH = "back to search"

ACTION_RE = re.compile(r"(search|click)\[(.*?)\]", re.IGNORECASE | re.DOTALL)
TOKEN_RE = re.compile(r"[a-z0-9]+")

SMALL_FILE_IDS = {
    "items_file": ("items_shuffle_1000.json", "1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib"),
    "attrs_file": ("items_ins_v2_1000.json", "1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu"),
    "human_file": ("items_human_ins.json", "14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O"),
}

FULL_FILE_IDS = {
    "items_file": ("items_shuffle.json", "1A2whVgOO0euk5O13n2iYDM0bQRkkRduB"),
    "attrs_file": ("items_ins_v2.json", "1s2j6NgHljiZzQNL3veZaAiyW_qDEgBNi"),
    "human_file": ("items_human_ins.json", "14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O"),
}

HF_MIRROR_BASE_URL = "https://huggingface.co/datasets/YWZBrandon/webshop-data/resolve/main"

SPLIT_RANGES: dict[str, tuple[int, int | None]] = {
    "test": (0, 500),
    "eval": (500, 1500),
    "train": (1500, None),
    "all": (0, None),
}


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "")]


def _normalized_text(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _ratio(left: str, right: str) -> int:
    return int(round(100.0 * SequenceMatcher(None, left, right).ratio()))


def _token_set_ratio(left: str, right: str) -> int:
    left_tokens = set(_tokenize(left))
    right_tokens = set(_tokenize(right))
    if not left_tokens and not right_tokens:
        return 100
    if not left_tokens or not right_tokens:
        return 0

    common = left_tokens & right_tokens
    left_only = left_tokens - common
    right_only = right_tokens - common

    common_text = " ".join(sorted(common))
    left_text = " ".join(sorted(common | left_only))
    right_text = " ".join(sorted(common | right_only))
    return max(
        _ratio(common_text, left_text),
        _ratio(common_text, right_text),
        _ratio(left_text, right_text),
    )


def _safe_float_price(raw_price: str) -> float:
    cleaned = re.sub(r"[^\d.]", "", raw_price)
    if not cleaned:
        return 100.0
    return float(Decimal(cleaned))


def _google_drive_confirm_token(response: requests.Response) -> str | None:
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    try:
        text = response.text
    except Exception:
        return None

    for pattern in (
        r'confirm=([0-9A-Za-z_]+)&amp;id=',
        r'name="confirm" value="([0-9A-Za-z_]+)"',
        r'"confirm":"([0-9A-Za-z_]+)"',
    ):
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None


def download_google_drive_file(file_id: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    url = "https://drive.google.com/uc?export=download"

    with requests.Session() as session:
        response = session.get(url, params={"id": file_id}, stream=True, timeout=60)
        token = _google_drive_confirm_token(response)
        if token:
            response.close()
            response = session.get(
                url,
                params={"id": file_id, "confirm": token},
                stream=True,
                timeout=60,
            )
        response.raise_for_status()

        with output_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def download_http_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    with output_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)


def _looks_like_json(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open("rb") as handle:
        prefix = handle.read(128).lstrip()
    return prefix.startswith(b"{") or prefix.startswith(b"[")


@dataclass(frozen=True)
class WebShopGoal:
    goal_index: int
    asin: str
    category: str
    query: str
    name: str
    product_category: str
    instruction_text: str
    attributes: tuple[str, ...]
    price_upper: float
    goal_options: dict[str, str] | tuple[str, ...]
    weight: float

    def to_metadata(self) -> dict[str, Any]:
        return {
            "goal_index": self.goal_index,
            "asin": self.asin,
            "category": self.category,
            "query": self.query,
            "name": self.name,
            "product_category": self.product_category,
            "instruction_text": self.instruction_text,
            "attributes": list(self.attributes),
            "price_upper": self.price_upper,
            "goal_options": (
                dict(self.goal_options)
                if isinstance(self.goal_options, dict)
                else list(self.goal_options)
            ),
            "weight": self.weight,
        }


class LightweightProductSearcher:
    def __init__(self, products: list[dict[str, Any]], attribute_to_asins: dict[str, set[str]]) -> None:
        self.products = list(products)
        self.attribute_to_asins = attribute_to_asins
        self._product_by_asin = {str(item["asin"]).lower(): item for item in self.products}
        self._doc_terms: dict[str, Counter[str]] = {}
        self._idf: dict[str, float] = {}
        self._prepare_index()

    def _prepare_index(self) -> None:
        df: Counter[str] = Counter()
        for product in self.products:
            asin = str(product["asin"]).lower()
            option_text = []
            for option_name, option_values in product.get("options", {}).items():
                option_text.append(f"{option_name}: {' '.join(option_values)}")
            contents = " ".join(
                [
                    str(product.get("Title", "")),
                    str(product.get("Description", "")),
                    " ".join(str(item) for item in product.get("BulletPoints", [])),
                    " ".join(str(item) for item in product.get("Attributes", [])),
                    " ".join(option_text),
                    str(product.get("query", "")),
                    str(product.get("product_category", "")),
                ]
            ).lower()
            tf = Counter(_tokenize(contents))
            self._doc_terms[asin] = tf
            df.update(tf.keys())

        total_docs = max(1, len(self.products))
        self._idf = {
            term: math.log((1.0 + total_docs) / (1.0 + count)) + 1.0 for term, count in df.items()
        }

    def search(self, query: str, k: int = SEARCH_RETURN_N) -> list[dict[str, Any]]:
        text = str(query or "").strip().lower()
        if not text:
            return []

        parts = text.split()
        if parts and parts[0] == "<r>":
            rng = random.Random(text)
            pool = list(self.products)
            rng.shuffle(pool)
            return pool[:k]
        if len(parts) > 1 and parts[0] == "<a>":
            attribute = " ".join(parts[1:]).strip()
            asins = self.attribute_to_asins.get(attribute, set())
            return [self._product_by_asin[asin] for asin in sorted(asins)[:k] if asin in self._product_by_asin]
        if len(parts) > 1 and parts[0] == "<c>":
            category = " ".join(parts[1:]).strip()
            return [
                product
                for product in self.products
                if str(product.get("category", "")).strip().lower() == category
            ][:k]
        if len(parts) > 1 and parts[0] == "<q>":
            target_query = " ".join(parts[1:]).strip()
            return [
                product
                for product in self.products
                if str(product.get("query", "")).strip().lower() == target_query
            ][:k]

        query_terms = Counter(_tokenize(text))
        scored: list[tuple[float, dict[str, Any]]] = []
        for product in self.products:
            asin = str(product["asin"]).lower()
            tf = self._doc_terms.get(asin, Counter())
            score = 0.0
            for term, count in query_terms.items():
                doc_count = tf.get(term, 0)
                if doc_count <= 0:
                    continue
                score += count * (1.0 + math.log(1.0 + doc_count)) * self._idf.get(term, 1.0)
            if score > 0.0:
                scored.append((float(score), product))

        if not scored:
            return self.products[:k]

        scored.sort(key=lambda item: item[0], reverse=True)
        return [product for _, product in scored[:k]]


def _goal_option_values(goal_options: dict[str, str] | list[Any] | tuple[Any, ...]) -> list[str]:
    if isinstance(goal_options, dict):
        return [str(value).lower() for value in goal_options.values()]
    values: list[str] = []
    for item in goal_options:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            values.append(str(item[1]).lower())
        else:
            values.append(str(item).lower())
    return values


def _type_reward(purchased_product: dict[str, Any], goal: WebShopGoal) -> dict[str, Any]:
    query_match = str(purchased_product.get("query", "")).strip().lower() == goal.query

    purchased_category = {
        part.strip().lower()
        for part in str(purchased_product.get("product_category", "")).split("›")
        if part.strip()
    }
    goal_category = {part.strip().lower() for part in goal.product_category.split("›") if part.strip()}
    category_match = len(purchased_category & goal_category) >= 2

    purchased_tokens = _tokenize(str(purchased_product.get("name", "")))
    desired_tokens = _tokenize(goal.name)
    if not desired_tokens:
        title_score = 0.2
    else:
        title_score = len(set(purchased_tokens) & set(desired_tokens)) / len(set(desired_tokens))

    reward = 1.0
    match = query_match or category_match or title_score > 0.2
    if not match:
        reward = 0.5
    if title_score < 0.1:
        reward = 0.1
    if title_score == 0.0:
        reward = 0.0

    return {
        "r_type": reward,
        "query_match": query_match,
        "category_match": category_match,
        "title_score": title_score,
    }


def _attribute_reward(purchased_product: dict[str, Any], goal: WebShopGoal) -> tuple[float, int]:
    purchased_attrs = [str(item).lower() for item in purchased_product.get("Attributes", [])]
    title = str(purchased_product.get("Title", "")).lower()
    bullets = " ".join(str(item).lower() for item in purchased_product.get("BulletPoints", []))
    description = str(purchased_product.get("Description", "")).lower()

    matches = 0
    for goal_attr in goal.attributes:
        goal_attr_text = str(goal_attr).lower()
        matched = any(_token_set_ratio(p_attr, goal_attr_text) > 85 for p_attr in purchased_attrs)
        if not matched and (
            goal_attr_text in title or goal_attr_text in bullets or goal_attr_text in description
        ):
            matched = True
        if matched:
            matches += 1

    reward = matches / len(goal.attributes) if goal.attributes else 1.0
    return reward, matches


def _option_reward(purchased_options: list[str], goal_options: list[str]) -> tuple[float | None, int]:
    if not goal_options:
        return None, 0

    normalized_purchased = [str(item).lower() for item in purchased_options]
    normalized_goal = [str(item).lower() for item in goal_options]
    matches = 0
    for goal_option in normalized_goal:
        if any(_token_set_ratio(goal_option, purchased_option) > 85 for purchased_option in normalized_purchased):
            matches += 1
    return matches / len(normalized_goal), matches


def compute_reward(
    purchased_product: dict[str, Any],
    goal: WebShopGoal,
    *,
    price: float,
    selected_options: dict[str, str],
) -> tuple[float, dict[str, Any]]:
    type_details = _type_reward(purchased_product, goal)
    r_price = price <= goal.price_upper if goal.price_upper > 0 else None
    r_attr, attr_matches = _attribute_reward(purchased_product, goal)
    purchased_option_values = [str(value).lower() for value in selected_options.values()]
    goal_option_values = _goal_option_values(goal.goal_options)
    r_option, option_matches = _option_reward(purchased_option_values, goal_option_values)

    denominator = len(goal.attributes) + len(goal_option_values) + 1
    total_reward = (attr_matches + option_matches + int(bool(r_price))) / denominator
    total_reward *= float(type_details["r_type"])

    reward_info = {
        "r_type": type_details["r_type"],
        "r_att": r_attr,
        "r_option": r_option,
        "r_price": r_price,
        "query_match": type_details["query_match"],
        "category_match": type_details["category_match"],
        "title_score": type_details["title_score"],
        "attr_matches": attr_matches,
        "option_matches": option_matches,
        "goal_attributes": list(goal.attributes),
        "goal_options": goal_option_values,
        "purchased_options": purchased_option_values,
        "purchased_attrs": [str(item) for item in purchased_product.get("Attributes", [])],
        "paper_score_100": total_reward * 100.0,
    }
    return float(total_reward), reward_info


def _parse_human_goals(
    all_products: list[dict[str, Any]],
    product_prices: dict[str, float],
) -> list[WebShopGoal]:
    rng = random.Random(233)
    goals: list[WebShopGoal] = []

    for item in all_products:
        asin = str(item["asin"])
        for product_goal in item.get("instructions", []):
            attributes = [str(attr).lower() for attr in product_goal.get("instruction_attributes", [])]
            if not attributes:
                continue

            price = float(product_prices.get(asin, 100.0))
            price_range = [candidate for candidate in [10.0 * i for i in range(1, 100)] if candidate > price][:4]
            if len(price_range) >= 2:
                _, price_upper = sorted(rng.sample(price_range, 2))
                price_text = f", and price lower than {price_upper:.2f} dollars"
            else:
                price_upper = 1_000_000.0
                price_text = ""

            instruction_text = str(product_goal.get("instruction", "")).strip().rstrip(".") + price_text
            raw_goal_options = product_goal.get("instruction_options", {}) or {}
            if isinstance(raw_goal_options, dict):
                goal_options: dict[str, str] | tuple[str, ...] = {
                    str(key).lower(): str(value).lower() for key, value in raw_goal_options.items()
                }
            else:
                goal_options = tuple(str(value).lower() for value in list(raw_goal_options))

            goals.append(
                WebShopGoal(
                    goal_index=-1,
                    asin=asin.lower(),
                    category=str(item.get("category", "")).lower(),
                    query=str(item.get("query", "")).lower(),
                    name=str(item.get("name", "")).lower(),
                    product_category=str(item.get("product_category", "")).lower(),
                    instruction_text=instruction_text,
                    attributes=tuple(attributes),
                    price_upper=float(price_upper),
                    goal_options=goal_options,
                    weight=1.0,
                )
            )

    rng.shuffle(goals)
    return [
        WebShopGoal(
            goal_index=index,
            asin=goal.asin,
            category=goal.category,
            query=goal.query,
            name=goal.name,
            product_category=goal.product_category,
            instruction_text=goal.instruction_text,
            attributes=goal.attributes,
            price_upper=goal.price_upper,
            goal_options=goal.goal_options,
            weight=goal.weight,
        )
        for index, goal in enumerate(goals)
    ]


def load_products(
    *,
    items_path: Path,
    attrs_path: Path,
    human_path: Path,
    num_products: int | None = None,
    human_goals: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, float], dict[str, set[str]]]:
    products = json.loads(items_path.read_text(encoding="utf-8"))
    attrs = json.loads(attrs_path.read_text(encoding="utf-8"))
    human_attrs = json.loads(human_path.read_text(encoding="utf-8"))

    if num_products is not None:
        products = products[: int(num_products)]

    seen_asins: set[str] = set()
    all_products: list[dict[str, Any]] = []
    attribute_to_asins: dict[str, set[str]] = defaultdict(set)

    for product in products:
        asin = str(product.get("asin", "")).strip()
        if asin == "nan" or not asin or len(asin) > 10:
            continue
        asin_lower = asin.lower()
        if asin_lower in seen_asins:
            continue
        seen_asins.add(asin_lower)

        normalized: dict[str, Any] = dict(product)
        normalized["category"] = str(product.get("category", "")).strip().lower()
        normalized["query"] = str(product.get("query", "")).strip().lower()
        normalized["product_category"] = str(product.get("product_category", "")).strip().lower()
        normalized["name"] = str(product.get("name", "")).strip()
        normalized["Title"] = normalized["name"]
        normalized["Description"] = str(product.get("full_description", "")).strip()
        small_description = product.get("small_description", [])
        if isinstance(small_description, list):
            normalized["BulletPoints"] = [str(item).strip() for item in small_description if str(item).strip()]
        elif small_description:
            normalized["BulletPoints"] = [str(small_description).strip()]
        else:
            normalized["BulletPoints"] = []
        normalized["Reviews"] = []
        normalized["Rating"] = "N.A."

        raw_pricing = product.get("pricing")
        if not raw_pricing:
            pricing = [100.0]
            price_tag = "$100.0"
        elif isinstance(raw_pricing, list):
            pricing = [float(item) for item in raw_pricing[:2]]
            price_tag = f"${pricing[0]}" if len(pricing) == 1 else f"${pricing[0]} to ${pricing[-1]}"
        else:
            pricing = [_safe_float_price(item) for item in str(raw_pricing).split("$")[1:]]
            if not pricing:
                pricing = [100.0]
            if len(pricing) == 1:
                price_tag = f"${pricing[0]}"
            else:
                pricing = pricing[:2]
                price_tag = f"${pricing[0]} to ${pricing[1]}"
        normalized["pricing"] = pricing
        normalized["Price"] = price_tag

        options: dict[str, list[str]] = {}
        option_to_image: dict[str, str | None] = {}
        customization_options = product.get("customization_options") or {}
        for option_name, option_contents in customization_options.items():
            if option_contents is None:
                continue
            option_name_lower = str(option_name).strip().lower()
            values: list[str] = []
            for option_content in option_contents:
                value = str(option_content.get("value", "")).strip().replace("/", " | ").lower()
                if not value:
                    continue
                values.append(value)
                option_to_image[value] = option_content.get("image")
            if values:
                options[option_name_lower] = values
        normalized["options"] = options
        normalized["option_to_image"] = option_to_image

        attr_payload = attrs.get(asin) or attrs.get(asin_lower) or {}
        attributes = attr_payload.get("attributes")
        if isinstance(attributes, list) and attributes:
            normalized["Attributes"] = [str(item).lower() for item in attributes]
        else:
            normalized["Attributes"] = ["dummy_attr"]

        if human_goals:
            normalized["instructions"] = list(human_attrs.get(asin, human_attrs.get(asin_lower, [])) or [])
        else:
            normalized["instructions"] = []

        images = product.get("images") or []
        normalized["MainImage"] = str(images[0]) if images else ""
        normalized["asin"] = asin_lower

        all_products.append(normalized)
        for attr in normalized["Attributes"]:
            attribute_to_asins[attr].add(asin_lower)

    product_item_dict = {str(item["asin"]).lower(): item for item in all_products}
    product_prices: dict[str, float] = {}
    rng = random.Random(233)
    for product in all_products:
        asin = str(product["asin"]).lower()
        pricing = list(product.get("pricing", [100.0]))
        if not pricing:
            price = 100.0
        elif len(pricing) == 1:
            price = float(pricing[0])
        else:
            low, high = float(pricing[0]), float(pricing[1])
            price = float(rng.uniform(low, high))
        product_prices[asin] = price

    return all_products, product_item_dict, product_prices, attribute_to_asins


class WebShopDataStore:
    def __init__(
        self,
        *,
        data_root: Path,
        data_mode: str,
        auto_download: bool,
        human_goals: bool,
        num_products: int | None,
        show_attrs: bool,
    ) -> None:
        self.data_root = data_root.expanduser().resolve()
        self.data_mode = data_mode
        self.auto_download = auto_download
        self.human_goals = human_goals
        self.num_products = num_products
        self.show_attrs = show_attrs

        self.all_products: list[dict[str, Any]] = []
        self.product_item_dict: dict[str, dict[str, Any]] = {}
        self.product_prices: dict[str, float] = {}
        self.attribute_to_asins: dict[str, set[str]] = {}
        self.goals: list[WebShopGoal] = []
        self.searcher: LightweightProductSearcher | None = None

    def ensure_loaded(self) -> None:
        if self.goals and self.searcher is not None:
            return

        items_path, attrs_path, human_path = self._ensure_assets()
        (
            self.all_products,
            self.product_item_dict,
            self.product_prices,
            self.attribute_to_asins,
        ) = load_products(
            items_path=items_path,
            attrs_path=attrs_path,
            human_path=human_path,
            num_products=self.num_products,
            human_goals=self.human_goals,
        )
        self.goals = _parse_human_goals(self.all_products, self.product_prices)
        self.searcher = LightweightProductSearcher(self.all_products, self.attribute_to_asins)

    def _ensure_assets(self) -> tuple[Path, Path, Path]:
        if self.data_mode not in {"small", "full"}:
            raise ValueError("webshop.data_mode must be 'small' or 'full'")

        files = SMALL_FILE_IDS if self.data_mode == "small" else FULL_FILE_IDS
        mode_dir = self.data_root / self.data_mode
        paths: dict[str, Path] = {}
        for key, (filename, file_id) in files.items():
            path = mode_dir / filename
            if not _looks_like_json(path):
                if not self.auto_download:
                    raise FileNotFoundError(
                        f"Missing WebShop asset '{path}'. Enable auto_download or place the file manually."
                    )
                mirror_url = f"{HF_MIRROR_BASE_URL}/{filename}"
                try:
                    download_http_file(mirror_url, path)
                except Exception:
                    download_google_drive_file(file_id, path)
                if not _looks_like_json(path):
                    raise RuntimeError(
                        f"Downloaded WebShop asset '{filename}' but the payload does not look like JSON."
                    )
            paths[key] = path
        return paths["items_file"], paths["attrs_file"], paths["human_file"]

    def goals_for_split(self, split: str) -> list[WebShopGoal]:
        self.ensure_loaded()
        if split not in SPLIT_RANGES:
            raise ValueError(f"Unknown WebShop split '{split}'. Use one of: {sorted(SPLIT_RANGES)}")
        start, end = SPLIT_RANGES[split]
        return list(self.goals[start:end])

    def make_episode(self, goal_index: int) -> WebShopEpisode:
        self.ensure_loaded()
        if goal_index < 0 or goal_index >= len(self.goals):
            raise IndexError(f"Goal index out of range: {goal_index}")
        assert self.searcher is not None
        return WebShopEpisode(
            goal=self.goals[goal_index],
            product_item_dict=self.product_item_dict,
            product_prices=self.product_prices,
            searcher=self.searcher,
            show_attrs=self.show_attrs,
        )


class WebShopEpisode:
    def __init__(
        self,
        *,
        goal: WebShopGoal,
        product_item_dict: dict[str, dict[str, Any]],
        product_prices: dict[str, float],
        searcher: LightweightProductSearcher,
        show_attrs: bool,
    ) -> None:
        self.goal = goal
        self.product_item_dict = product_item_dict
        self.product_prices = product_prices
        self.searcher = searcher
        self.show_attrs = show_attrs

        self.page_name = "search"
        self.subpage_name: str | None = None
        self.keywords: list[str] | None = None
        self.page_number = 1
        self.search_results: list[str] = []
        self.current_asin: str | None = None
        self.selected_options: dict[str, str] = {}
        self.visited_asins: set[str] = set()
        self.done = False
        self.final_reward = 0.0
        self.reward_info: dict[str, Any] = {}
        self.invalid_actions = 0
        self.action_counts: dict[str, int] = defaultdict(int)

    def current_product(self) -> dict[str, Any] | None:
        if not self.current_asin:
            return None
        return self.product_item_dict.get(self.current_asin)

    def current_page_products(self) -> list[dict[str, Any]]:
        if not self.search_results:
            return []
        start = max(0, (self.page_number - 1) * PRODUCT_WINDOW)
        end = start + PRODUCT_WINDOW
        return [
            self.product_item_dict[asin]
            for asin in self.search_results[start:end]
            if asin in self.product_item_dict
        ]

    def available_actions_info(self) -> dict[str, Any]:
        if self.done:
            return {"has_search_bar": False, "clickables": []}

        clickables: list[str] = []
        has_search_bar = False
        if self.page_name == "search":
            has_search_bar = True
        elif self.page_name == "results":
            clickables.append(BACK_TO_SEARCH)
            if self.page_number > 1:
                clickables.append(PREV_PAGE)
            if self.page_number * PRODUCT_WINDOW < len(self.search_results):
                clickables.append(NEXT_PAGE)
            clickables.extend(product["asin"] for product in self.current_page_products())
        elif self.page_name == "item":
            clickables.extend([BACK_TO_SEARCH, PREV_PAGE, "description", "features", "reviews"])
            if self.show_attrs:
                clickables.append("attributes")
            product = self.current_product() or {}
            for option_values in product.get("options", {}).values():
                clickables.extend(str(value).lower() for value in option_values)
            clickables.append(END_BUTTON)
        elif self.page_name == "subpage":
            clickables.extend([BACK_TO_SEARCH, PREV_PAGE])
        return {"has_search_bar": has_search_bar, "clickables": clickables}

    def observation_text(self) -> str:
        parts: list[str] = ["WebShop", f"Instruction: {self.goal.instruction_text}"]
        if self.page_name == "search":
            parts.append("Search")
            return " [SEP] ".join(parts)

        if self.page_name == "results":
            parts.extend(
                [
                    BACK_TO_SEARCH.title(),
                    f"Page {self.page_number} (Total results: {len(self.search_results)})",
                ]
            )
            if self.page_number > 1:
                parts.append(PREV_PAGE)
            if self.page_number * PRODUCT_WINDOW < len(self.search_results):
                parts.append(NEXT_PAGE)
            for product in self.current_page_products():
                parts.extend(
                    [
                        product["asin"],
                        str(product.get("Title", "")),
                        str(product.get("Price", "")),
                    ]
                )
            return " [SEP] ".join(parts)

        product = self.current_product() or {}
        if self.page_name == "item":
            parts.extend(
                [
                    BACK_TO_SEARCH.title(),
                    PREV_PAGE,
                    f"ASIN: {product.get('asin', '')}",
                    str(product.get("Title", "")),
                    f"Price: {product.get('Price', '')}",
                    f"Rating: {product.get('Rating', '')}",
                ]
            )
            if self.selected_options:
                selected_text = ", ".join(
                    f"{name}={value}" for name, value in sorted(self.selected_options.items())
                )
                parts.append(f"Selected options: {selected_text}")
            for option_name, option_values in product.get("options", {}).items():
                option_text = ", ".join(option_values)
                parts.append(f"Option {option_name}: {option_text}")
            parts.extend(["Description", "Features", "Reviews"])
            if self.show_attrs:
                parts.append("Attributes")
            parts.append("Buy Now")
            return " [SEP] ".join(parts)

        if self.page_name == "subpage":
            parts.extend([BACK_TO_SEARCH.title(), PREV_PAGE])
            subpage = self.subpage_name or ""
            parts.append(f"Subpage: {subpage}")
            if subpage == "description":
                parts.append(str(product.get("Description", "")))
            elif subpage == "features":
                parts.extend(str(item) for item in product.get("BulletPoints", []))
            elif subpage == "reviews":
                reviews = product.get("Reviews", [])
                if reviews:
                    for review in reviews[:5]:
                        title = str(review.get("title", "")).strip()
                        body = str(review.get("body", "")).strip()
                        score = str(review.get("score", "")).strip()
                        parts.append(f"Review {score}: {title} {body}".strip())
                else:
                    parts.append("No reviews available.")
            elif subpage == "attributes":
                parts.extend(str(item) for item in product.get("Attributes", []))
            return " [SEP] ".join(parts)

        parts.append(f"Final score: {self.final_reward:.4f}")
        return " [SEP] ".join(parts)

    def step(self, action: str) -> tuple[str, float, bool, dict[str, Any]]:
        if self.done:
            return self.observation_text(), self.final_reward, True, self.info()

        action_name, action_arg = parse_action(action)
        valid = self.available_actions_info()
        clickables = {str(item).lower() for item in valid["clickables"]}
        handled = False

        if action_name == "search" and valid["has_search_bar"] and action_arg:
            self._search(action_arg)
            handled = True
        elif action_name == "click" and action_arg and action_arg in clickables:
            self._click(action_arg)
            handled = True

        if not handled:
            self.invalid_actions += 1

        return self.observation_text(), self.final_reward, self.done, self.info()

    def info(self) -> dict[str, Any]:
        return {
            "goal_index": self.goal.goal_index,
            "page_name": self.page_name,
            "keywords": list(self.keywords or []),
            "page_number": self.page_number,
            "asin": self.current_asin,
            "selected_options": dict(self.selected_options),
            "available_actions": self.available_actions_info(),
            "invalid_actions": self.invalid_actions,
            "action_counts": dict(self.action_counts),
            "reward": self.final_reward,
            "reward_info": dict(self.reward_info),
            "paper_score_100": self.final_reward * 100.0,
            "done": self.done,
        }

    def _search(self, query: str) -> None:
        query = str(query).strip().lower()
        if not query:
            return
        self.action_counts["search"] += 1
        self.keywords = query.split()
        self.page_number = 1
        self.current_asin = None
        self.selected_options = {}
        self.page_name = "results"
        self.subpage_name = None
        self.search_results = [
            str(product["asin"]).lower() for product in self.searcher.search(query, k=SEARCH_RETURN_N)
        ]

    def _click(self, clickable_name: str) -> None:
        clickable_name = str(clickable_name).lower()
        if clickable_name == BACK_TO_SEARCH:
            self.action_counts["back_to_search"] += 1
            self.page_name = "search"
            self.subpage_name = None
            self.keywords = None
            self.page_number = 1
            self.current_asin = None
            self.selected_options = {}
            self.search_results = []
            return

        if self.page_name == "results":
            if clickable_name == PREV_PAGE and self.page_number > 1:
                self.action_counts["paginate"] += 1
                self.page_number -= 1
                return
            if clickable_name == NEXT_PAGE and self.page_number * PRODUCT_WINDOW < len(self.search_results):
                self.action_counts["paginate"] += 1
                self.page_number += 1
                return
            if clickable_name in self.product_item_dict:
                self.action_counts["asin"] += 1
                self.current_asin = clickable_name
                self.visited_asins.add(clickable_name)
                self.page_name = "item"
                self.subpage_name = None
                return
            return

        if self.page_name == "item":
            if clickable_name == PREV_PAGE:
                self.page_name = "results"
                self.subpage_name = None
                return
            if clickable_name == END_BUTTON:
                self._finish_purchase()
                return
            if clickable_name in {"description", "features", "reviews"} or (
                self.show_attrs and clickable_name == "attributes"
            ):
                self.action_counts[clickable_name] += 1
                self.page_name = "subpage"
                self.subpage_name = clickable_name
                return

            product = self.current_product() or {}
            for option_name, option_values in product.get("options", {}).items():
                if clickable_name in {str(value).lower() for value in option_values}:
                    self.action_counts["options"] += 1
                    self.selected_options[option_name] = clickable_name
                    return
            return

        if self.page_name == "subpage" and clickable_name == PREV_PAGE:
            self.page_name = "item"
            self.subpage_name = None

    def _finish_purchase(self) -> None:
        if not self.current_asin:
            return
        product = self.current_product()
        if product is None:
            return
        self.action_counts["purchase"] += 1
        price = float(self.product_prices.get(self.current_asin, 100.0))
        self.final_reward, self.reward_info = compute_reward(
            product,
            self.goal,
            price=price,
            selected_options=self.selected_options,
        )
        self.done = True
        self.page_name = "done"
        self.subpage_name = None


def parse_action(action: str) -> tuple[str, str | None]:
    match = ACTION_RE.search(str(action or ""))
    if match is None:
        cleaned = str(action or "").strip().lower()
        return cleaned, None
    action_name, action_arg = match.groups()
    return action_name.strip().lower(), action_arg.strip().lower()
