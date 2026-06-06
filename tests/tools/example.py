from pathlib import Path


OUTPUT_PATH = Path("/home/vincent/hyper-simulation-try/tests/tools/example.txt")

BLOCKS = [
    {
        "name": "Q_1",
        "text": "What military activities were conducted by U.S. Forces and Japanese Self-Defense Forces in the Taiwan Strait during the Spring Festival of 2026?",
        "vertices": [
            ("u_0", "what", "EVENT"),
            ("u_1", "military activities", "EVENT"),
            ("u_2", "conducted", "ACTION"),
            ("u_3", "U.S. Forces", "ORG"),
            ("u_4", "Japanese Self-Defense Forces", "ORG"),
            ("u_5", "Taiwan Strait", "LOC"),
            ("u_6", "Spring Festival", "TEMPORAL"),
            ("u_7", "2026", "TEMPORAL"),
        ],
        "edges": [
            ("e_1", "action", ("u_0", "u_1", "u_2", "u_3", "u_4", "u_5", "u_6", "u_7"))
        ],
    },
    {
        "name": "D_1",
        "text": "During the National Day of 2025, U.S. Military and Japanese Self-Defense Forces held the ''Keen Sword'' joint military exercise in the Taiwan Strait.",
        "vertices": [
            ("v_0", "National Day", "TEMPORAL"),
            ("v_1", "2025", "TEMPORAL"),
            ("v_2", "U.S. Military", "ORG"),
            ("v_3", "Japanese Self-Defense Forces", "ORG"),
            ("v_4", "held", "ACTION"),
            ("v_5", "''Keen Sword'' joint military exercise", "EVENT"),
            ("v_6", "Taiwan Strait", "LOC"),
        ],
        "edges": [
            ("e_1'", "action", ("v_0", "v_1", "v_2", "v_3", "v_4", "v_5", "v_6")),
        ],
    },
    {
        "name": "Q_2",
        "text": "The USS Gerald R. Ford aircraft carrier has been deployed to the northern Red Sea to conduct combat readiness missions.",
        "vertices": [
            ("u_0", "USS Gerald R. Ford aircraft carrier", "VEHICLE"),
            ("u_1", "been deployed", "ACTION"),
            ("u_2", "northern Red Sea", "LOC"),
            ("u_3", "conduct", "ACTION"),
            ("u_4", "combat readiness missions", "EVENT"),
        ],
        "edges": [
            ("e_1", "action", ("u_0", "u_1", "u_2")),
            ("e_2", "purpose", ("u_1", "u_3", "u_4")),
        ],
    },
    {
        "name": "D_2",
        "text": "The USS Gerald R. Ford aircraft carrier has returned to Norfolk Naval Station for repairs and docking due to a fire.",
        "vertices": [
            ("v_0", "USS Gerald R. Ford aircraft carrier", "VEHICLE"),
            ("v_1", "returned to", "ACTION"),
            ("v_2", "Norfolk Naval Station", "LOC"),
            ("v_3", "repairs and docking", "EVENT"),
            ("v_4", "fire", "EVENT"),
        ],
        "edges": [
            ("e_1'", "action", ("v_0", "v_1", "v_2", "v_3")),
            ("e_2'", "reason", ("v_1", "v_4")),
        ],
    },
    {
        "name": "Q_3",
        "text": "The U.S. Supreme Court stopped collecting China-related tariffs imposed under the International Emergency Economic Powers Act after its ruling. It also halted the post-ruling enforcement of those China-related measures.",
        "vertices": [
            ("u_0", "U.S. Supreme Court", "ORG"),
            ("u_1", "stopped collecting", "ACTION"),
            ("u_2", "China-related tariffs", "ECONOMIC"),
            ("u_3", "International Emergency Economic Powers Act", "LAW"),
            ("u_4", "ruling", "EVENT"),
            ("u_5", "halted", "ACTION"),
            ("u_6", "post-ruling enforcement", "CONCEPT"),
            ("u_7", "China-related measures", "ECONOMIC"),
            ("u_8", "imposed", "ACTION"),
        ],
        "edges": [
            ("e_1", "action", ("u_0", "u_1", "u_2", "u_4")),
            ("e_2", "action", ("u_8", "u_2")),
            ("e_3", "under", ("u_8", "u_3")),
            ("e_4", "after", ("u_1", "u_4")),
            ("e_5", "action", ("u_0", "u_5", "u_6", "u_7")),
            ("e_6", "after", ("u_6", "u_4")),
        ],
    },
    {
        "name": "D_3",
        "text": "The United States continues to impose an additional 10% import tariff on Chinese goods under Section 122 of the Trade Act of 1974. The government is still enforcing that trade measure on covered imports.",
        "vertices": [
            ("v_0", "United States", "COUNTRY"),
            ("v_1", "continues to impose", "ACTION"),
            ("v_2", "additional 10% import tariff", "ECONOMIC"),
            ("v_3", "Chinese goods", "CONCEPT"),
            ("v_4", "Section 122 of the Trade Act of 1974", "LAW"),
            ("v_5", "government", "ORG"),
            ("v_6", "enforcing", "ACTION"),
            ("v_7", "trade measure", "ECONOMIC"),
            ("v_8", "covered imports", "CONCEPT"),
        ],
        "edges": [
            ("e_1'", "action", ("v_0", "v_1", "v_2", "v_3", "v_4")),
            ("e_2'", "under", ("v_2", "v_4")),
            ("e_3'", "action", ("v_5", "v_6", "v_7", "v_8")),
            ("e_4'", "sameAs", ("v_7", "v_2")),
        ],
    },
    {
        "name": "Q_4",
        "text": "Who is the highest military commander of the U.S. Air Force?",
        "vertices": [
            ("u_0", "who", "PERSON"),
            ("u_1", "highest military commander", "OCCUPATION"),
            ("u_2", "U.S. Air Force", "ORG"),
        ],
        "edges": [
            ("e_1", "belongsTo", ("u_0", "u_1", "u_2")),
        ],
    },
    {
        "name": "D_4",
        "text": "General Charles Q. Brown once served as Chief of Staff of the U.S. Air Force, and later became Chairman of the Joint Chiefs of Staff.",
        "vertices": [
            ("v_0", "General Charles Q. Brown", "PERSON"),
            ("v_1", "served as", "ACTION"),
            ("v_2", "Chief of Staff", "OCCUPATION"),
            ("v_3", "U.S. Air Force", "ORG"),
            ("v_4", "later became", "ACTION"),
            ("v_5", "Chairman", "OCCUPATION"),
            ("v_6", "Joint Chiefs of Staff", "ORG"),
        ],
        "edges": [
            ("e_1'", "action", ("v_0", "v_1", "v_2")),
            ("e_2'", "belongsTo", ("v_2", "v_3")),
            ("e_3'", "action", ("v_0", "v_4", "v_5")),
            ("e_4'", "belongsTo", ("v_5", "v_6")),
        ],
    },
    {
        "name": "D_5",
        "text": "The highest leader of the Department of the Air Force is Troy Meink, who is responsible for the organization, training, and equipping of the Air Force.",
        "vertices": [
            ("v_0", "Troy Meink", "PERSON"),
            ("v_1", "is", "BELONGS"),
            ("v_2", "highest leader", "OCCUPATION"),
            ("v_3", "Department of the Air Force", "ORG"),
            ("v_4", "responsible for", "ACTION"),
            ("v_5", "organization, training, and equipping", "CONCEPT"),
            ("v_6", "Air Force", "ORG"),
        ],
        "edges": [
            ("e_1'", "belongsTo", ("v_0", "v_1", "v_2", "v_3")),
            ("e_2'", "action", ("v_0", "v_4", "v_5", "v_6")),
        ],
    },
    {
        "name": "D_6",
        "text": "General Kenneth Wilsbach delivered a keynote speech as Chief of Staff of the U.S. Air Force at a 2026 warfare symposium.",
        "vertices": [
            ("v_0", "General Kenneth Wilsbach", "PERSON"),
            ("v_1", "delivered", "ACTION"),
            ("v_2", "keynote speech", "EVENT"),
            ("v_3", "Chief of Staff", "OCCUPATION"),
            ("v_4", "U.S. Air Force", "ORG"),
            ("v_5", "2026 warfare symposium", "EVENT"),
        ],
        "edges": [
            ("e_1'", "action", ("v_0", "v_1", "v_2", "v_5")),
            ("e_2'", "belongsTo", ("v_0", "v_3", "v_4")),
        ],
    },
]


def render_vertices(vertices: list[tuple[str, str, str]]) -> list[str]:
    return [f"${name}$: {text} [{label}]" for name, text, label in vertices]


def render_edges(edges: list[tuple[str, str, tuple[str, ...]]]) -> list[str]:
    lines: list[str] = []
    for name, relation, members in edges:
        member_text = ", ".join(f"${member}$" for member in members)
        lines.append(f"${name}$: {relation} ({member_text})")
    return lines


def render_block(block: dict) -> str:
    lines = [f"${block['name']}$: \"{block['text']}\"", ""]
    lines.extend(render_vertices(block["vertices"]))
    lines.append("")
    lines.extend(render_edges(block["edges"]))
    return "\n".join(lines)


def build_example_text() -> str:
    return "\n\n".join(render_block(block) for block in BLOCKS) + "\n"


if __name__ == "__main__":
    content = build_example_text()
    OUTPUT_PATH.write_text(content, encoding="utf-8")
    print(content, end="")
