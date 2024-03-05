"""Placeholder file template.
"""

import heapq
import itertools
from collections import Counter

import msgspec
import pandas


def main() -> int:

    # Load database using msgspec
    class Card(msgspec.Struct, dict=True):
        name: str | None = None
        mana_cost: str | None = None
        cmc: float | None = None
        colors: list[str] | None = None
        power: str | None = None
        toughness: str | None = None
        keywords: list[str] | None = None
        type_line: str | None = None
        oracle_text: str | None = None
        # legalities: dict[str, str] | None = None

    with open("data/oracle-cards-20240301220151.json", "rb") as json_file:
        dataset = msgspec.json.decode(json_file.read(), type=list[Card])

    df = pandas.DataFrame([msgspec.structs.asdict(card) for card in dataset])
    features = df[["name"]].copy()

    # Oracle text
    df["oracle_text_no_reminder"] = (
        df["oracle_text"]
        .str.replace(r"[\(].*?[\)]", "", regex=True)
        .str.lower()
    )
    cares = [
        "creature",
        "token",
        "instant",
        "sorcery",
        "legend",
        "land",
        "artifact",
        "enchantment",
        "damage",
        "prevent",
        "dies",
        "destroy",
        "life",
    ]
    for care in cares:
        features[f"oracle_cares_{care}"] = df[
            "oracle_text_no_reminder"
        ].str.contains(care)

    # Numerical (cmc, power, toughness)
    def is_float(s: str):
        try:
            f = float(s)
            return (True, f)
        except:
            return (False, float("nan"))

    features["power_is_numeric"], features["power_float"] = zip(
        *df["power"].apply(is_float)
    )
    features["toughness_is_numeric"], features["toughness_float"] = zip(
        *df["toughness"].apply(is_float)
    )
    features["cmc"] = df["cmc"]

    # Card types
    df["card_types"] = df["type_line"].str.split(" ")

    unique_types_counts = Counter(
        itertools.chain.from_iterable(df["card_types"])
    )
    unique_types_counts.pop("—")

    def get_types(list_of_types: list):
        out = []
        for card_type in unique_types_counts.keys():
            if card_type in list_of_types:
                out.append(1.0)
            else:
                out.append(0.0)
        return out

    df["is_type"] = df["card_types"].apply(get_types)
    df_types = pandas.DataFrame(df["is_type"].to_list(), index=df.index)
    df_types.columns = [
        f"is_type_{key}" for key in list(unique_types_counts.keys())
    ]

    features = features.join(df_types)

    # Colors
    unique_colors_counts = Counter(
        itertools.chain.from_iterable(df["colors"].dropna())
    )

    def get_colors(list_of_colors: list):
        if not list_of_colors:
            return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        out = [0.0]
        for color in unique_colors_counts.keys():
            if color in list_of_colors:
                out.append(1.0)
            else:
                out.append(0.0)
        return out

    df["is_color"] = df["colors"].apply(get_colors)
    df_colors = pandas.DataFrame(df["is_color"].to_list(), index=df.index)
    df_colors.columns = [
        f"is_color_{key}" for key in ["C"] + list(unique_colors_counts.keys())
    ]

    features = features.join(df_colors)

    # Keywords
    unique_keywords_counts = Counter(
        itertools.chain.from_iterable(df["keywords"])
    )

    def get_keywords(list_of_keywords: list):
        if len(list_of_keywords) == 0:
            return [0.0] * len(unique_keywords_counts)
        out = []
        for keyword in unique_keywords_counts.keys():
            if keyword in list_of_keywords:
                out.append(1.0)
            else:
                out.append(0.0)
        return out

    df["is_keyword"] = df["keywords"].apply(get_keywords)
    df_keywords = pandas.DataFrame(df["is_keyword"].to_list(), index=df.index)
    df_keywords.columns = [
        f"is_keyword_{key}" for key in list(unique_keywords_counts.keys())
    ]

    features = features.join(df_keywords)

    query = features
    query = query.drop(columns=["power_is_numeric", "toughness_is_numeric"])
    query = query.set_index("name")
    question_number = 0
    while len(query) > 1:
        # calculate max median split for all features
        num_cards = len(query)
        split_ratios: list[tuple] = []
        for feature in query.columns:
            median_value = query[feature].median()
            split_above = (query[feature] > median_value).sum()
            split_equal = (query[feature] == median_value).sum()
            split_below = (query[feature] < median_value).sum()

            op_above = ">"
            op_below = "<"
            if split_above < split_below:
                split_above += split_equal
                op_above += "="
            else:
                split_below += split_equal
                op_below += "="

            if split_above < split_below:
                heapq.heappush(
                    split_ratios,
                    (
                        -split_above / num_cards,
                        feature,
                        op_above,
                        median_value,
                    ),
                )
            else:
                heapq.heappush(
                    split_ratios,
                    (
                        -split_below / num_cards,
                        feature,
                        op_below,
                        median_value,
                    ),
                )

        feat, op, med = heapq.heappop(split_ratios)[1:]
        question_number += 1
        answer = input(
            f"Q{question_number}: Is {feat} {op} {med}? (y/n/exit)> "
        )
        if answer == "exit":
            break
        elif ((op == ">=") and (answer == "y")) or (
            (op == "<") and (answer == "n")
        ):
            mask = query[feat] >= med
        elif ((op == ">") and (answer == "y")) or (
            (op == "<=") and (answer == "n")
        ):
            mask = query[feat] > med
        elif ((op == "<=") and (answer == "y")) or (
            (op == ">") and (answer == "n")
        ):
            mask = query[feat] <= med
        elif ((op == "<") and (answer == "y")) or (
            (op == ">=") and (answer == "n")
        ):
            mask = query[feat] < med
        else:
            break
        query = query[mask]

        # remove singular columns
        query = query.loc[:, ~(query == query.iloc[0]).all()]
        print(f"{len(query)} cards remaining...")

        if query.empty:
            print(f"You must have chosen {query.index.to_list()}")
            return 0

    print(f"You must have chosen {query.index[0]}!")

    return 0


if __name__ == "__main__":
    main()
