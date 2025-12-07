import os, re, random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from xgboost import XGBRanker
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# ================= CONFIG =================
SEED = 993
NFOLDS = 3
DIM_WORD = 80
DIM_CHAR = 40
DIM_DESC = 40
DIM_BULLET = 30

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"

random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# ================= SAVE =================
def create_submission(predictions, ids):
    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame({"id": ids, "prediction": predictions})
    path = "results/submission.csv"
    df.to_csv(path, index=False)
    print(f" Saved: {path}")
    return path

# ================= NLP =================
try:
    nltk.data.find("corpora/stopwords")
except:
    nltk.download("stopwords")

STOP = set(stopwords.words("english"))
stemmer = SnowballStemmer("english")
_rx = re.compile(r"[a-z0-9]+")

def clean(x):
    if pd.isna(x): return ""
    x = re.sub(r"<[^>]+>", " ", str(x).lower())
    x = re.sub(r"\s+"," ",x)
    return x.strip()

def tok(x):
    return [stemmer.stem(t) for t in _rx.findall(clean(x)) if t not in STOP]

def col(df,c):
    return df[c].fillna("").astype(str) if c in df.columns else pd.Series([""]*len(df))

# ================= MAIN =================
def main():
    print(" Start")

    df = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    for c in ["query","product_title","product_description","product_bullet_point","product_brand","product_color"]:
        df[c] = col(df,c).map(clean)
        test[c] = col(test,c).map(clean)

    for d in [df,test]:
        d["qt"] = d["query"].map(tok)
        d["tt"] = d["product_title"].map(tok)
        d["dt"] = d["product_description"].map(tok)
        d["bt"] = d["product_bullet_point"].map(tok)
        d["full"] = (d["product_title"]+" "+d["product_description"]+" "+d["product_bullet_point"]).map(clean)

    # ===== TFIDF + SVD blocks =====
    all_full = pd.concat([df["full"], test["full"]])
    all_title = pd.concat([df["product_title"], test["product_title"]])
    all_desc = pd.concat([df["product_description"], test["product_description"]])
    all_bullet = pd.concat([df["product_bullet_point"], test["product_bullet_point"]])

    def tfidf_svd(texts, dim, char=False):
        if char:
            vec = TfidfVectorizer(
                max_features=40000,
                analyzer="char_wb",
                ngram_range=(1,3),
                min_df=2
            )
        else:
            vec = TfidfVectorizer(
                max_features=40000,
                analyzer="word",
                ngram_range=(1,2),
                min_df=2
            )
        svd = TruncatedSVD(dim, random_state=SEED)
        X = vec.fit_transform(texts)
        return vec, svd, svd.fit_transform(X)

    vec_full, svd_full, X_full = tfidf_svd(all_full, DIM_WORD)
    vec_title, svd_title, X_title = tfidf_svd(all_title, DIM_WORD)
    vec_desc, svd_desc, X_desc = tfidf_svd(all_desc, DIM_DESC)
    vec_bullet, svd_bullet, X_bullet = tfidf_svd(all_bullet, DIM_BULLET)
    _, svd_char, X_char = tfidf_svd(all_full, DIM_CHAR, char=True)

    ntr = len(df)
    def split(X): return X[:ntr], X[ntr:]

    ftr, fte = split(X_full)
    ttr, tte = split(X_title)
    dtr, dte = split(X_desc)
    btr, bte = split(X_bullet)
    ctr, cte = split(X_char)

    # ===== BM25 =====
    def build_idf(col_tokens):
        cnt = defaultdict(int)
        for toks in col_tokens:
            for t in set(toks): cnt[t]+=1
        N=len(col_tokens)
        return {t:np.log((N-v+0.5)/(v+0.5)+1) for t,v in cnt.items()}

    idf_title = build_idf(df["tt"])
    idf_desc = build_idf(df["dt"])
    avg_t = np.mean(df["tt"].map(len))
    avg_d = np.mean(df["dt"].map(len))

    def bm25(q,d,idf,avg):
        tf=Counter(d)
        s=0; k1=1.5; b=0.75
        norm=k1*(1-b+b*len(d)/(avg+1e-9))
        for t in q:
            if t in idf:
                f=tf.get(t,0)
                s+=idf[t]*(f*(k1+1))/(f+norm+1e-9)
        return s

    def bm25_block(d):
        out=np.zeros((len(d),6))
        for i,(q,t,dd,bb) in enumerate(zip(d["qt"],d["tt"],d["dt"],d["bt"])):
            st=bm25(q,t,idf_title,avg_t)
            sd=bm25(q,dd,idf_desc,avg_d)
            sb=bm25(q,bb,idf_desc,avg_d)
            out[i]=[st,sd,sb,np.log1p(st),np.log1p(sd),np.log1p(sb)]
        return out

    bm_tr=bm25_block(df); bm_te=bm25_block(test)

    # ===== Similarity block =====
    def sim_block(d):
        Qt=vec_title.transform(d["query"])
        Tt=vec_title.transform(d["product_title"])
        cos=np.array([cosine_similarity(Qt[i],Tt[i])[0,0] for i in range(len(d))])
        jac=np.array([len(set(q)&set(t))/max(1,len(set(q)|set(t))) for q,t in zip(d["qt"],d["tt"])])
        cov=np.array([len(set(q)&set(t))/max(1,len(q)) for q,t in zip(d["qt"],d["tt"])])
        return np.vstack([cos,jac,cov]).T

    sim_tr=sim_block(df); sim_te=sim_block(test)

    # ===== Length + interaction =====
    def len_block(d):
        ql=d["qt"].map(len).values
        tl=d["tt"].map(len).values
        dl=d["dt"].map(len).values
        bl=d["bt"].map(len).values
        return np.vstack([ql,tl,dl,bl,np.log1p(ql),np.log1p(tl),ql/(tl+1)]).T

    def inter_block(d):
        br=(d["query"]==d["product_brand"]).astype(float).values
        co=(d["query"]==d["product_color"]).astype(float).values
        return np.vstack([br,co]).T

    len_tr,len_te=len_block(df),len_block(test)
    int_tr,int_te=inter_block(df),inter_block(test)

    # ===== Merge all features =====
    X_train=np.hstack([ftr,ttr,dtr,btr,ctr,bm_tr,sim_tr,len_tr,int_tr])
    X_test =np.hstack([fte,tte,dte,bte,cte,bm_te,sim_te,len_te,int_te])

    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)

    print("Features shape:",X_train.shape)

    # ===== Strong ensemble ranking models =====
    test_pred=np.zeros(len(test))
    gkf=GroupKFold(n_splits=NFOLDS)

    for fold,(tr,va) in enumerate(gkf.split(X_train,groups=df["query_id"]),1):
        print("Fold",fold)
        Xt,Xv=X_train[tr],X_train[va]
        yt=df.iloc[tr]["relevance"].values
        g=df.iloc[tr].groupby("query_id").size().values

        m1=XGBRanker(
            n_estimators=1200,learning_rate=0.03,max_depth=9,
            subsample=0.85,colsample_bytree=0.85,random_state=SEED
        )
        m2=XGBRanker(
            n_estimators=900,learning_rate=0.04,max_depth=7,
            subsample=0.9,colsample_bytree=0.9,random_state=SEED+1
        )

        m1.fit(Xt,yt,group=g,verbose=False)
        m2.fit(Xt,yt,group=g,verbose=False)

        p = 0.6*m1.predict(X_test)+0.4*m2.predict(X_test)
        test_pred+=p/NFOLDS

    create_submission(test_pred,test["id"].values)
    print(" Done")

if __name__=="__main__":
    main()
