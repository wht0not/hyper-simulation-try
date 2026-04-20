import os
import json
import time
from nltk.corpus import wordnet as wn
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_ollama import ChatOllama
from hyper_simulation.utils.chat_completion import get_generate
from tqdm import tqdm

OUTPUT_FILE = "qwen_ontology_mapping.json"
PROGRESS_FILE = "qwen_ontology_progress.json"
BATCH_SIZE = 20  # 姣忔浼犵粰 get_generate 鐨?prompt 鏁伴噺 (鍙栧喅浜庝綘鐨勬樉瀛樺ぇ灏忓拰 Ollama 骞跺彂璁剧疆)

VALID_CATEGORIES = [
    "PERSON", "COUNTRY", "LOC", "ORG", "FAC", "NORP", "PRODUCT", 
    "WORK_OF_ART", "LAW", "LANGUAGE", "OCCUPATION", "EVENT", 
    "TEMPORAL", "NUMBER", "CONCEPT", "NOT_ENT"
]

CANDIDATE = [
    "ORGANISM: Living being, such as animal, plant, or microorganism.",
    "FOOD: Edible substance, dish, or cuisine.",
    "MEDICAL: Medical condition, disease, symptom, or treatment.",
    "ANATOMY: Body part, organ, or anatomical structure.",
    "SUBSTANCE: Chemical element, compound, or material.",
    "ASTRO: Astronomical object, such as a star, planet, or galaxy.",
    "AWARD: Prize, honor, or recognition given to a person or organization.",
    "VEHICLE: Means of transportation, such as a car, airplane, or bicycle.",
]

# 閽堝鍗曟潯鏁版嵁鐨勭簿绠€ Prompt锛屾瀬鍏堕€傚悎 9B 妯″瀷
SINGLE_PROMPT_TEMPLATE = """
You are an expert Ontologist. Classify the following WordNet synset into EXACTLY ONE of these categories:
PERSON: Human being, individual, or specific character.
COUNTRY: A nation with its own government.
LOC: Geographical location, natural region, body of water.
ORG: Organization, institution, company, government body.
FAC: Physical building, facility, structure.
鈥婫PE鈥嬧€? Geopolitical entity, such as cities, states, provinces (but not countries).
NORP: Nationalities, religious or political groups.
PRODUCT: Physical object, vehicle, device, manufactured good.
WORK_OF_ART: Piece of art, publication, show.
LAW: Legal document, binding agreement.
LANGUAGE: Spoken or written human language.
OCCUPATION: Job, profession, trade.
EVENT: Phenomenon, historical event, sports match.
TEMPORAL: Time period, specific date, unit of time.
NUMBER: Mathematical number, quantity.
CONCEPT: Abstract idea, theoretical concept.
ORGANISM: Living being, such as animal, plant, or microorganism.
FOOD: Edible substance, dish, or cuisine.
MEDICAL: Medical condition, disease, symptom, or treatment.
ANATOMY: Body part, organ, or anatomical structure.
SUBSTANCE: Chemical element, compound, or material.
ASTRO: Astronomical object, such as a star, planet, or galaxy.
AWARD: Prize, honor, or recognition given to a person or organization.
VEHICLE: Means of transportation, such as a car, airplane, or bicycle.
NOT_ENT: Use this if the synset does not fit any of the above 24 categories.

Synset Label: {label}
Meaning & Examples: {text}

Output ONLY the category name from the list above. Do not output any other words or explanations.
"""

# ================= 3. 杈呭姪瑙ｆ瀽鍑芥暟 =================
def extract_category(response_text: str) -> str:
    """
    娓呮礂鏈湴妯″瀷鐨勮緭鍑恒€傚嵆浣?Qwen 鍟板棪浜嗭紙姣斿杈撳嚭浜?"The category is: ORG"锛夛紝
    涔熻兘瀹夊叏鎻愬彇鍑哄搴旂殑绫诲埆銆?
    """
    response_upper = response_text.strip().upper()
    for cat in VALID_CATEGORIES:
        # 鍖归厤鍒颁换浣曚竴涓悎娉曠被鍒嵆杩斿洖
        if cat in response_upper:
            return cat
    return "NOT_ENT" # 濡傛灉妯″瀷鑳¤█涔辫锛岄粯璁ゅ綊涓?NOT_ENT


def load_json_file(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def atomic_json_dump(path: str, data) -> None:
    temp_path = f"{path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    os.replace(temp_path, path)


def save_state(mapping: dict, total_pending: int, done_in_run: int) -> None:
    atomic_json_dump(OUTPUT_FILE, mapping)
    progress_payload = {
        "updated_at": int(time.time()),
        "mapped_total": len(mapping),
        "current_run_total_pending": total_pending,
        "current_run_done": done_in_run,
    }
    atomic_json_dump(PROGRESS_FILE, progress_payload)

# ================= 4. 涓绘帶娴佺▼ (鍖呭惈鏂偣缁紶) =================
def main():
    # 鍒濆鍖栦綘鐨勬湰鍦版ā鍨?
    print("姝ｅ湪杩炴帴鏈湴 Qwen 妯″瀷...")
    llm = ChatOllama(model="qwen3.5:9b", top_p=0.95, reasoning=False)

    # 璇诲彇宸叉湁鐨勮繘搴︼紙鏂偣缁紶鏍稿績閫昏緫锛?
    existing_mapping = load_json_file(OUTPUT_FILE, default={})
    if existing_mapping:
        print(f"鉁?鎵惧埌鏈湴瀛樻。锛屽凡鍔犺浇 {len(existing_mapping)} 涓凡澶勭悊鏍囩锛屽噯澶囩户缁?..")
    elif os.path.exists(OUTPUT_FILE):
        print("鈿狅笍 瀛樻。鏂囦欢涓嶅彲璇伙紝灏嗕粠澶村紑濮嬨€?)

    progress = load_json_file(PROGRESS_FILE, default={})
    if progress:
        print(
            "馃搶 涓婃璁板綍: "
            f"mapped_total={progress.get('mapped_total', 0)}, "
            f"current_run_done={progress.get('current_run_done', 0)}/{progress.get('current_run_total_pending', 0)}"
        )

    print("姝ｅ湪浠?WordNet 鎻愬彇寰呭鐞嗗悕璇?..")
    pending_tasks = []
    
    # 閬嶅巻鎵€鏈夊悕璇?
    for syn in wn.all_synsets(pos='n'):
        label = syn.name()
        
        # 銆愬叧閿€戯細濡傛灉杩欎釜鏍囩宸茬粡鍦ㄥ瓨妗ｉ噷浜嗭紝灏辫烦杩囷紒
        if label in existing_mapping:
            continue
            
        text = syn.definition()
        if syn.examples():
            text += ". " + " ".join(syn.examples())
            
        pending_tasks.append({"label": label, "text": text})

    total_pending = len(pending_tasks)
    if total_pending == 0:
        print("馃帀 鎵€鏈?WordNet 鍚嶈瘝閮藉凡缁忓鐞嗗畬姣曚簡锛?)
        return

    print(f"鍏辨湁 {total_pending} 涓瘝鏉￠渶瑕佸鐞嗐€傚紑濮嬫壒澶勭悊 (Batch Size = {BATCH_SIZE})...")

    done_in_run = 0

    # 鍒嗘壒娆″鐞?
    try:
        for i in tqdm(range(0, total_pending, BATCH_SIZE)):
            batch = pending_tasks[i : i + BATCH_SIZE]

            # 1. 涓烘壒娆′腑鐨勬瘡涓瘝缁勮涓撳睘 Prompt
            batch_prompts = []
            for item in batch:
                prompt = SINGLE_PROMPT_TEMPLATE.format(label=item['label'], text=item['text'])
                batch_prompts.append(prompt)

            # 2. 璋冪敤浣犵殑鍑芥暟杩涜鎺ㄧ悊
            try:
                responses = get_generate(batch_prompts, llm)

                # 3. 鎸夋潯瀹炴椂钀界洏锛岄伩鍏嶄腑鏂椂涓㈠け鏁翠釜 batch
                for item, response_text in zip(batch, responses):
                    category = extract_category(response_text)
                    existing_mapping[item['label']] = category
                    done_in_run += 1
                    save_state(existing_mapping, total_pending, done_in_run)

            except Exception as e:
                print(f"鉂?鎵规 {i} 鍒?{i+BATCH_SIZE} 鎺ㄧ悊澶辫触: {e}")
                print("浼戠湢 5 绉掑悗缁х画涓嬩竴涓壒娆?..")
                time.sleep(5)
                continue

    except KeyboardInterrupt:
        print("\n鉀?妫€娴嬪埌涓柇锛屾鍦ㄤ繚瀛樺綋鍓嶈繘搴?..")
        save_state(existing_mapping, total_pending, done_in_run)
        print("鉁?宸蹭繚瀛橈紝鍙笅娆＄户缁墽琛屻€?)
        return

    # 缁撴潫鏃跺啀淇濆瓨涓€娆★紝纭繚鏈€缁堢姸鎬佷竴鑷?
    save_state(existing_mapping, total_pending, done_in_run)
    print("鉁?鍏ㄩ儴澶勭悊瀹屾垚锛屽凡淇濆瓨鏈€缁堟槧灏勪笌杩涘害鏂囦欢銆?)
            
if __name__ == "__main__":
    main()
