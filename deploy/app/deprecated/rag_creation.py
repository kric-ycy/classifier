#%%
import code
import pandas as pd

raw = pd.read_excel('data/raw/NB_패스트푸드/-2024 NBCI 하반기_코드프레임 1.xlsx', sheet_name='공통문항_문4,5이미지', header=None)
raw = raw.dropna(axis=1, how='all')
raw.columns = [*range(len(raw.columns))]
meta = raw.iloc[:10]

import numpy as np
meta_cols = meta.columns.tolist()
meta = meta.replace("NaN", np.nan)
store = pd.DataFrame()
for i in range(len(meta_cols)//2):

    for row in meta.index:

        code_col = meta_cols[i*2]
        text_col = meta_cols[i*2 + 1]

        meta_code = meta.iloc[row, code_col]
        meta_text = meta.iloc[row, text_col]
        print(meta_code, meta_text)
        metaname = None
        if pd.isna(meta_code) and not pd.isna(meta_text):
            metaname = meta_text
        elif not pd.isna(meta_code) and pd.isna(meta_text):
            metaname = meta_code
        elif not pd.isna(meta_code) and not pd.isna(meta_text):
            print('all filled')
        else:
            print('all empty')
        temp = pd.DataFrame([{
            'metaname': metaname,
            'row': f'{int(row)}',
            'column': f'{int(i)}'
        }])
        store = pd.concat([store, temp], ignore_index=True)

# %%
store = store.dropna(subset=['metaname'])
store = store[store['metaname'].apply(type) == str]
# %%
first_key = store[store['row'] == '0']
second_key = store[store['row'] == '1']
third_key = store[store['row'] == '2']


# %%
db_parsed = pd.DataFrame()

for i in range(len(meta_cols)//2):
    
        code_col = meta_cols[i*2]
        text_col = meta_cols[i*2 + 1]

        meta_code = raw[code_col]
        meta_text = raw[text_col]
        temp = pd.DataFrame({
            'text': meta_text,
            'code': meta_code,
            'row': [*range(len(meta_text))],
            'column': [f'{int(i)}'] * len(meta_text)
        })


        db_parsed = pd.concat([db_parsed, temp], ignore_index=True)
# %%
db_parsed = db_parsed.dropna()
db_parsed['first_key'] = db_parsed['column'].apply(lambda x: first_key[first_key['column'] == x]['metaname'].values[0] if x in first_key['column'].values else None)
db_parsed['second_key'] = db_parsed['column'].apply(lambda x: second_key[second_key['column'] == x]['metaname'].values[0] if x in second_key['column'].values else None)
db_parsed['third_key'] = db_parsed['column'].apply(lambda x: third_key[third_key['column'] == x]['metaname'].values[0] if x in third_key['column'].values else None) 

db_parsed['first_key'] = db_parsed['first_key'].fillna(method ='ffill')
db_parsed['second_key'] = db_parsed['second_key'].fillna(method ='ffill')
db_parsed['third_key'] = db_parsed['third_key'].fillna(method ='ffill')
# %%
def split_text_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # text → 여러 개면 리스트로 변환
    df['text_split'] = df['text'].apply(
        lambda x: [t.strip() for t in str(x).split('/')] if '/' in str(x) else [x]
    )

    # explode → 한 text마다 한 row
    df_exploded = df.explode('text_split')
    
    # 원래 text 컬럼을 대체
    df_exploded['text'] = df_exploded['text_split']
    df_exploded = df_exploded.drop(columns=['text_split'])

    return df_exploded.reset_index(drop=True)


db_parsed = split_text_rows(db_parsed)


# %%

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch
# 모델 로드
model = SentenceTransformer('jhgan/ko-sbert-sts')

# 예시 데이터프레임
df = db_parsed.copy()

df = df[df['text'] != ' ']
df = df[df['text'].str.strip() != '']
df['code'] = df['code'].astype('Int64')
# 리스트 형태로 batch encode
embeddings = model.encode(df['text'].tolist(), batch_size=32, show_progress_bar=True)
# normed = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)  # ✅ 행 단위 정규화
def normalize_torch_cpu(embeddings_np):
    t = torch.from_numpy(embeddings_np.astype(np.float32))
    t = t / t.norm(dim=1, keepdim=True)
    return t.numpy()
normed = normalize_torch_cpu(embeddings)
# DataFrame에 numpy array 컬럼으로 추가
df['embedding'] = list(normed)
df = df.sort_values(by='text').reset_index(drop=True)


print(normed.shape)  # (n, 768)
print(np.linalg.norm(normed[0]))  # ✅ 1.0 이어야 함

# %%
import psycopg2
import numpy as np

# DB 연결
conn = psycopg2.connect(
    dbname='net_classifier',
    user='aihc',
    password='aihc8520',
    host='116.125.140.82',
    port='5432'
)
cur = conn.cursor()


cur.execute("""SET search_path to test_schema, public;
""" )
conn.commit()


cur.execute("""Drop table if exists embed_test;""")
conn.commit()

cur.execute("""Create table if not exists embed_test (
    id serial primary key, 
    word text, 
    first_key text,
    second_key text,
    third_key text,
    code integer,
    embedding public.vector(768))""")
conn.commit() 



# %%
# INSERT 시 id는 제외하고 나머지만 삽입
from psycopg2.extras import execute_values

import numpy as np
import pandas as pd
from psycopg2.extras import execute_values

def safe_row(row):
    def safe(val):
        if pd.isna(val) or val == 'None':
            return None
        return val

    emb = row['embedding']
    if isinstance(emb, np.ndarray):
        emb = emb.astype(np.float32).tolist()  
    elif pd.isna(emb) or emb is None:
        emb = None

    return (
        safe(row['text']),
        safe(row['first_key']),
        safe(row['second_key']),
        safe(row['third_key']),
        int(row['code']) if not pd.isna(row['code']) else None,
        emb
    )

rows = [safe_row(row) for _, row in df.iterrows()]

insert_query = """
INSERT INTO test_schema.embed_test (
    word, first_key, second_key, third_key, code, embedding
) VALUES %s
"""

execute_values(cur, insert_query, rows)
conn.commit()


# %%

# %%
####d이히 search test

search_word = "곰팡이"
# 임베딩 + 정규화
embedding = model.encode([search_word])[0]
query_vec_normed = normalize_torch_cpu(np.array([embedding]))[0]

# 문자열로 변환
embedding_str = str(embedding.tolist())

# %%
search_query = """
SELECT word,
       embedding <=> %s::vector AS distance
FROM test_schema.embed_test
ORDER BY embedding <=> %s::vector
LIMIT 5;
"""

cur.execute(search_query, (embedding_str, embedding_str))
results = cur.fetchall()
print(results)
# %%


norm_check_query = """SELECT id, vector_norm(embedding) AS norm
FROM vectors
WHERE ABS(vector_norm(embedding) - 1.0) > 0.01;

"""

cur.execute(norm_check_query)
norm_results = cur.fetchall()
print(norm_results)
# %%
