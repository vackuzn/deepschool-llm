import re, os, asyncio
from typing import Iterator
import pandas as pd
from datetime import datetime
from dateutil import parser
from dotenv import load_dotenv
from together import AsyncTogether
from together.error import RateLimitError, ServiceUnavailableError
from tqdm import tqdm


# Settings
RU_MONTHS = {
    'января': '01', 'февраля': '02', 'марта': '03',
    'апреля': '04', 'мая': '05', 'июня': '06',
    'июля': '07', 'августа': '08', 'сентября': '09',
    'октября': '10', 'ноября': '11', 'декабря': '12',
}
model = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
max_in_tokens = 50000
max_out_tokens = 30
temperature = 0.0
intermediate_results_file = "intermediate_results.csv"
generation_column = "generation"
categories = ["politics", "economy", "culture"]


def parse_timestamp(ts: str) -> datetime:
    ts = ts.strip()

    # Case 1: Unix timestamp
    if ts.isdigit():
        return datetime.fromtimestamp(int(ts))

    # Case 2: Strip updated parts like "(обновлено: 21:13 31.08.2020)"
    ts = re.sub(r"\(обновлено:.*?\)", "", ts).strip()

    # Case 3: Russian date with month name (with or without comma)
    if any(month in ts for month in RU_MONTHS):
        parts = ts.replace(",", "").split()
        if len(parts) >= 4:
            time_part = parts[0]
            day, month_ru, year = parts[1:4]
            month = RU_MONTHS.get(month_ru)
            if not month:
                raise ValueError(f"Unrecognized month: '{month_ru}'")
            return datetime.strptime(f"{year}-{month}-{day} {time_part}", "%Y-%m-%d %H:%M")

    # Case 4: DD.MM.YYYY format
    match = re.search(r"(\d{1,2}:\d{2})\s+(\d{1,2})\.(\d{1,2})\.(\d{4})", ts)
    if match:
        time_str, day, month, year = match.groups()
        return datetime.strptime(f"{year}-{month}-{day} {time_str}", "%Y-%m-%d %H:%M")

    # Case 5: ISO or fallback
    try:
        return parser.parse(ts)
    except Exception as e:
        raise ValueError(f"Unrecognized timestamp format: {ts}") from e
    

def normalize_text(text: str) -> str:
    # Replace non-breaking space and other similar characters
    text = text.replace('\xa0', ' ')
    
    # Remove leading/trailing whitespace including newlines
    text = text.strip()

    # Collapse multiple whitespace and newlines into a single space
    text = re.sub(r'\s+', ' ', text)

    return text


def resolve_file_path(file_name: str) -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, file_name)


def load_content_from_file() -> list[dict]:
    with open(file=resolve_file_path("news.tsv"), mode='r', encoding='utf-8') as f:
        entries = f.read().split("\t")

    parsed = []

    for entry_idx in range(0, len(entries) - 1, 3):
        if '\n' in entries[entry_idx]:
            src = entries[entry_idx].split("\n")[1]
        else:
            src = entries[entry_idx]

        title = entries[entry_idx + 1]
        content = entries[entry_idx + 2]
        posted_ts = entries[entry_idx + 3].split("\n")[0]  

        parsed.append(
            {
                "source": src,
                "title": normalize_text(title),
                "content": normalize_text(content),
                "posted_ts": parse_timestamp(posted_ts).strftime("%Y-%m-%d %H:%M:%S")
            }
        )
    
    return parsed


def to_dataframe(data: list[dict]) -> pd.DataFrame:
    # Create DataFrame
    df = pd.DataFrame(data)
    # Ensure column types
    df = df.astype({
        "source": "string",
        "title": "string",
        "content": "string"
    })
    df["posted_ts"] = pd.to_datetime(df["posted_ts"], errors="coerce")  # handle invalid timestamps gracefully

    return df


async def classify_article(client: AsyncTogether, categories: list[str], title: str, content: str) -> str:
    prompt = f"""
    Given article title and article text, determine which categories does the article fall into.

    There are available categories:
    {",".join(categories)}

    Each article can fall into multiple categories. If no categories apply, return "none".

    Answer should contain only comma separated category names, nothing else
    """.strip()

    # Combine title and content for classification
    text_for_classification = f"Title: ```{title}```\nContent: ```{content}```"

    # Send to LLM for classification
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text_for_classification}
    ]

    while True:
        try:
            chat_completion = await client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False,
                temperature=temperature,
                max_tokens=max_out_tokens,
                max_in_tokens=max_in_tokens
            )
            break
        except RateLimitError as e:
            print(f"Error: {str(e)}")
            await asyncio.sleep(60)
        except ServiceUnavailableError as e:
            print(f"Error: {str(e)}")
            await asyncio.sleep(60)

    response_text = chat_completion.choices[0].message.content
    return response_text


def split_into_batches(df: pd.DataFrame, batch_size: int) -> Iterator[pd.DataFrame]:
    for i in range(0, len(df), batch_size):
        yield df[i:i+batch_size]

    if len(df) % batch_size != 0:
        yield df[len(df) - len(df) % batch_size:]


async def annotate_articles(df: pd.DataFrame, limit: int | None = None) -> pd.DataFrame:
    concurrency_limit = 10

    client = AsyncTogether()

    # Initialize binary classification columns (0/1) for each category if they don't exist
    for c in categories:
        if c not in df.columns:
            df[c] = 0

    if generation_column not in df.columns:
        df[generation_column] = None

    # Get count of non-empty generation results
    non_empty_count = df[df[generation_column].notna()].shape[0]
    if limit is None:
        limit = len(df)

    print(f"Number of articles already classified: {non_empty_count}")

    if non_empty_count == len(df):
        return df
    
    # Create list of coroutines for empty rows up to limit
    to_process_rows = df[df[generation_column].isna()].iloc[:limit]
    
    for batch in tqdm(split_into_batches(to_process_rows, concurrency_limit), total=len(to_process_rows)):
        coroutines = [
            classify_article(client, categories, row["title"], row["content"])
            for _, row in batch.iterrows()
        ]

        batch_results = await asyncio.gather(*coroutines)
        for idx, result in zip(batch.index, batch_results):
            df.loc[idx, generation_column] = result

        df.to_csv(resolve_file_path(intermediate_results_file), index=False)
    
    return df

def apply_categories(df: pd.DataFrame) -> pd.DataFrame:
    for c in categories:
        df[c] = df[generation_column].str.contains(c).astype(int)

    df.to_csv(resolve_file_path(intermediate_results_file), index=False)
    return df

async def main():
    load_dotenv()
    articles_to_annotate_limit: int | None = None

    # Load and process data
    if os.path.exists(resolve_file_path("intermediate_results.csv")):
        print("Loading intermediate results...")
        df = pd.read_csv(resolve_file_path("intermediate_results.csv"))
    else:
        print("Reloading data...")
        data_json = load_content_from_file()
        df = to_dataframe(data_json)

    print(df.dtypes)
    print(df.head())
        
    # Annotate articles
    print("\nAnnotating articles...")
    df = await annotate_articles(df, articles_to_annotate_limit)
    print("\nApplying categories...")
    df = apply_categories(df)
    
    return df

if __name__ == "__main__":
    asyncio.run(main())
