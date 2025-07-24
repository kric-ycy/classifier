# %%

### filter the obect columns from the raw data
import pandas as pd

raw = pd.read_excel("data/raw/NB_패스트푸드/24H2_NBCI_패스트푸드_코딩완료데이터_20240614_F.xlsx",sheet_name="DATA")
raw = raw.set_index("SEQ")
# after 3rd row, define the types and get only the obj columns :
def get_object_columns(df):
    """
    Get columns with object data type from a DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to analyze.
    
    Returns:
    list: A list of column names with object data type.
    """
    df_part = df.iloc[3:]
    object_columns = [col for col in df_part.columns if df_part[col].dropna().dtype == 'object']
    filtered_columns = [col for col in object_columns if col.startswith("Q") or col.startswith("sq")]
    return filtered_columns
object_columns = get_object_columns(raw)

obj_data = raw[object_columns].iloc[2:]

obj_data.to_csv(r".\data\temp_storage\object_columns_data.csv")

def get_question_columns(df, q_num , grouped_dict):
    columns = grouped_dict.get(q_num  , [])
    r_df = df[columns]
    return r_df


q_numbers = list(set([question.replace("z","_").split("_")[0] for question in object_columns]))

grouped_dict = {prefix: [] for prefix in q_numbers}

for q_num in q_numbers:
    grouped_columns = [col for col in obj_data.columns if col.startswith(q_num)]
    grouped_dict[q_num] = grouped_columns


df = get_question_columns(obj_data, "Q5", grouped_dict).dropna()


# %%
# vector search 
import psycopg2
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer
import numpy as np

# 모델 준비
model = SentenceTransformer('jhgan/ko-sbert-sts')

def search_top_k(word, k=3):
    # DB 커넥션 (스레드마다 새로 만들어야 함)
    conn = psycopg2.connect(
        dbname='net_classifier',
        user='aihc',
        password='aihc8520',
        host='116.125.140.82',
        port='5432'
    )
    cur = conn.cursor()

    # 벡터 임베딩 및 정규화
    embedding = model.encode([word])[0]
    vector_str = str(embedding.tolist())

    # 쿼리 실행
    query = """
    SELECT word, code, embedding <=> %s::vector AS distance
    FROM test_schema.embed_test
    ORDER BY embedding <=> %s::vector
    LIMIT %s;
    """
    cur.execute(query, (vector_str, vector_str, k))
    results = cur.fetchall()
    if results is None:
        return {
            'word': word,
            'classified': None,
            'code': None,
            'distance': None,
            'match': False
        }
    cur.close()
    conn.close()

    return (word, results)

whole = []
for col in df.columns:
    df[col] = df[col].astype(str)
    search_words = df[col].tolist()
    # 병렬 실행
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(search_top_k, word) for word in search_words]

        all_results = [future.result() for future in futures]
    # 결과 통합
    for word, results in all_results:
        if results:
            for res in results:
                whole.append({
                    'column': col,
                    'word': word,
                    'classified': res[0],
                    'code': res[1],
                    'distance': res[2]
                })

# %%
whole_df = pd.DataFrame(whole)


fallback_threshold = 0.01
whole_df['distance'] = whole_df.apply(lambda x: 1 if x['distance'] < fallback_threshold and x['classified'] != x['word'] else x['distance'], axis=1)


# %%
whole_df[whole_df['distance'] < 0.24]
# %%

# %%
col_index = raw.columns.get_loc('Q5_1_1')
print(col_index)
col_index = raw.columns.get_loc('Q5_4_2')
print(col_index)
# %%
df_cleaned = raw.iloc[:, 137: 153]  
df_clean_cols = df_cleaned.columns.tolist()
df_cleaned.columns = [col.replace(".", "_") for col in df_clean_cols]

df_cleaned = df_cleaned.dropna(how='all').reset_index()
word_cols = [col for col in df_cleaned.columns if col.startswith("Q5")]
pairs = [(col, df_cleaned.columns[i + 1]) for i, col in enumerate(df_cleaned.columns) if col in word_cols]

# %%
records = []
for row in df_cleaned.itertuples(index=False):
    for word_col, code_col in pairs:
        word = getattr(row, word_col)
        code = getattr(row, code_col)
        if pd.notna(word) and pd.notna(code):
            records.append({
                'column': word_col,
                'word': word,
                'code_validate': int(code)
            })

long_df = pd.DataFrame(records)
# %%
match = pd.merge(whole_df, long_df, on=['column', 'word'], how='right')
match['code_validate'] = match['code_validate'].fillna(0).astype(int)
match['match'] = match['code'] == match['code_validate']

match = match.drop_duplicates()
match[match['match'] == False]['distance'].describe()
# 5768 
# %%
match[(match['distance'] < 0.20) & (match['match'] == False)]
# %%
match[match['match'] == False].sort_values(by='distance', ascending=True).head(10)
# %%


# %%
