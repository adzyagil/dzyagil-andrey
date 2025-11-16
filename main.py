import os
import random
import math
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder
from catboost import CatBoostClassifier, Pool
import torch
import torch.nn as nn
import torch.optim as optim

RSEED = 42
np.random.seed(RSEED)
random.seed(RSEED)

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
SUB_PATH = "results/submission.csv"

N_FOLDS = 5
MU = 1.0 / 3.0
W_CLIP = 50.0
HPO_TRIALS = 8
CEM_POP = 40
CEM_ITERS = 6
CEM_ELITE = 0.2
W_Q = 0.6
W_PI = 0.4
POLICY_EPOCHS = 40
POLICY_BATCH = 1024
ENT_REG = 0.01
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def normalize_action(s):
    if not isinstance(s, str):
        return "no_email"
    x = s.lower()
    if "mens" in x:
        return "mens_email"
    if "women" in x or "womens" in x:
        return "womens_email"
    if "no" in x:
        return "no_email"
    return "no_email"

def sample_catboost_params():
    return {
        "iterations": random.choice([400, 600, 800]),
        "learning_rate": float(10 ** np.random.uniform(np.log10(0.01), np.log10(0.08))),
        "depth": int(random.choice([4, 5, 6])),
        "l2_leaf_reg": float(10 ** np.random.uniform(np.log10(1), np.log10(20))),
        "random_seed": RSEED
    }

def softmax_policy(q, tau):
    L = q / tau
    L = L - L.max(axis=1, keepdims=True)
    ex = np.exp(L)
    return ex / ex.sum(axis=1, keepdims=True)

def compute_snips_ips_dr(df, policy, q_hat, mu=MU, clip=W_CLIP):
    amap = {"mens_email":0,"womens_email":1,"no_email":2}
    ai = df["action"].map(amap).values
    pi_ai = policy[np.arange(len(df)), ai]
    w = pi_ai / mu
    if clip is not None:
        w = np.minimum(w, clip)
    r = df["visit"].astype(float).values
    ips = np.mean(w * r)
    snips = (np.sum(w * r) / np.sum(w)) if np.sum(w)!=0 else 0.0
    q_pi = (policy * q_hat).sum(axis=1)
    q_ai = q_hat[np.arange(len(df)), ai]
    dr = np.mean(q_pi + w * (r - q_ai))
    return {"ips": ips, "snips": snips, "sum_w": float(w.sum()), "mean_w": float(w.mean()), "dr": dr}

def compute_best_static(df, mu=MU):
    names = ["Mens E-Mail","Womens E-Mail","No E-Mail"]
    res={}
    for s in names:
        mask = df.get("segment", df.get("action","")) == s
        ri = df.loc[mask, "visit"].astype(float).values
        if len(ri)==0:
            res[s]=0.0
        else:
            res[s] = float(np.sum(ri*(1.0/mu))/ (len(ri)*(1.0/mu)))
    best = max(res, key=res.get)
    return res, res[best], best

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
test_ids = test["id"].astype(int).values

train["action"] = train["segment"].apply(normalize_action)
if "segment" in test.columns:
    test["action"] = test["segment"].apply(normalize_action)
train["visit"] = train["visit"].astype(int)

id_col = "id"
target_col = "visit"
exclude = {id_col, target_col, "segment", "action"}
feature_cols = [c for c in train.columns if c not in exclude]

cat_candidates=[]
num_candidates=[]
for c in feature_cols:
    if train[c].dtype==object:
        cat_candidates.append(c)
    else:
        if train[c].nunique()<=12:
            cat_candidates.append(c)
        else:
            num_candidates.append(c)

for c in feature_cols:
    if set(train[c].dropna().unique()).issubset({0,1}):
        train[c]=train[c].astype(int)
        if c in test.columns:
            test[c]=test[c].astype(int)

cat_features = [c for c in cat_candidates if c in train.columns and c in test.columns]
num_features = [c for c in num_candidates if c in train.columns]

if len(cat_features)>0:
    enc = OrdinalEncoder()
    train[cat_features] = enc.fit_transform(train[cat_features].astype(str))
    test[cat_features] = enc.transform(test[cat_features].astype(str))
    train[cat_features] = train[cat_features].astype("int32")
    test[cat_features] = test[cat_features].astype("int32")

features_final = [c for c in feature_cols if c in train.columns]
for c in features_final:
    if c not in test.columns:
        test[c]=0
test = test[features_final]

arms = ["mens_email","womens_email","no_email"]
K = len(arms)
n = len(train)
q_oof = np.zeros((n,K))
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RSEED)
for tr_idx, val_idx in kf.split(train):
    df_tr = train.iloc[tr_idx].reset_index(drop=True)
    df_val = train.iloc[val_idx].reset_index(drop=True)
    for ai, arm in enumerate(arms):
        mask_tr_arm = (df_tr["action"]==arm)
        X_tr_arm = df_tr.loc[mask_tr_arm, features_final]
        y_tr_arm = df_tr.loc[mask_tr_arm, "visit"].astype(int)
        if len(X_tr_arm) < 60:
            Xs_tr = df_tr[features_final].copy()
            Xs_tr["action_tmp"]=df_tr["action"].astype(str).values
            Xs_val = df_val[features_final].copy()
            Xs_val["action_tmp"]=arm
            cat_idx_local = [i for i,c in enumerate(Xs_tr.columns) if c in cat_features + ["action_tmp"]]
            m = CatBoostClassifier(**sample_catboost_params())
            m.fit(Pool(Xs_tr, df_tr["visit"].astype(int).values, cat_features=cat_idx_local), verbose=False)
            preds = m.predict_proba(Pool(Xs_val, cat_features=cat_idx_local))[:,1]
        else:
            cat_idx_local = [i for i,c in enumerate(X_tr_arm.columns) if c in cat_features]
            m = CatBoostClassifier(**sample_catboost_params())
            m.fit(Pool(X_tr_arm, y_tr_arm, cat_features=cat_idx_local), verbose=False)
            preds = m.predict_proba(Pool(df_val[features_final], cat_features=cat_idx_local))[:,1]
        q_oof[val_idx, ai] = preds

per_arm_models = {}
for ai, arm in enumerate(arms):
    mask = (train["action"]==arm)
    X_af = train.loc[mask, features_final]
    y_af = train.loc[mask, "visit"].astype(int)
    if len(X_af) < 60:
        Xs_full = train[features_final].copy()
        Xs_full["action_tmp"]=train["action"].astype(str)
        cat_idx_local = [i for i,c in enumerate(Xs_full.columns) if c in cat_features + ["action_tmp"]]
        m = CatBoostClassifier(**sample_catboost_params())
        m.fit(Pool(Xs_full, train["visit"].astype(int).values, cat_features=cat_idx_local), verbose=False)
        per_arm_models[arm] = ("s", m)
    else:
        cat_idx_local = [i for i,c in enumerate(X_af.columns) if c in cat_features]
        m = CatBoostClassifier(**sample_catboost_params())
        m.fit(Pool(X_af, y_af, cat_features=cat_idx_local), verbose=False)
        per_arm_models[arm] = ("t", m)

Xs_full = train[features_final].copy()
Xs_full["action_tmp"]=train["action"].astype(str)
cat_idx_global = [i for i,c in enumerate(Xs_full.columns) if c in cat_features + ["action_tmp"]]
global_model = CatBoostClassifier(**sample_catboost_params())
global_model.fit(Pool(Xs_full, train["visit"].astype(int).values, cat_features=cat_idx_global), verbose=False)

X_test_feats = test[features_final].copy()
q_test_perarm = np.zeros((len(X_test_feats), K))
for ai, arm in enumerate(arms):
    typ, model = per_arm_models[arm]
    if typ=="t":
        cat_idx_local = [i for i,c in enumerate(X_test_feats.columns) if c in cat_features]
        q_test_perarm[:, ai] = model.predict_proba(Pool(X_test_feats, cat_features=cat_idx_local))[:,1]
    else:
        Xtmp = X_test_feats.copy()
        Xtmp["action_tmp"]=arm
        cat_idx_local = [i for i,c in enumerate(Xtmp.columns) if c in cat_features + ["action_tmp"]]
        q_test_perarm[:, ai] = model.predict_proba(Pool(Xtmp, cat_features=cat_idx_local))[:,1]

q_test_global = np.zeros_like(q_test_perarm)
for ai, arm in enumerate(arms):
    Xtmp = X_test_feats.copy()
    Xtmp["action_tmp"]=arm
    cat_idx_local = [i for i,c in enumerate(Xtmp.columns) if c in cat_features + ["action_tmp"]]
    q_test_global[:, ai] = global_model.predict_proba(Pool(Xtmp, cat_features=cat_idx_local))[:,1]

q_hat_test = 0.6 * q_test_perarm + 0.4 * q_test_global

q_oof_global = np.zeros_like(q_oof)
for ai, arm in enumerate(arms):
    Xg = train[features_final].copy()
    Xg["action_tmp"]=arm
    cat_idx_local = [i for i,c in enumerate(Xg.columns) if c in cat_features + ["action_tmp"]]
    q_oof_global[:, ai] = global_model.predict_proba(Pool(Xg, cat_features=cat_idx_local))[:,1]

q_oof_ensemble = 0.6 * q_oof + 0.4 * q_oof_global

arm_means = np.array([ float(train.loc[train["action"]==a, "visit"].mean()) if (train["action"]==a).sum()>0 else float(train["visit"].mean()) for a in arms ])
q_oof_ensemble = 0.85 * q_oof_ensemble + 0.15 * arm_means.reshape(1,-1)
q_hat_test = 0.85 * q_hat_test + 0.15 * arm_means.reshape(1,-1)

def compute_dr_scalar(df, policy, qhat):
    d = compute_snips_ips_dr(df, policy, qhat, mu=MU, clip=W_CLIP)
    return d["dr"]

tau_grid = [0.2,0.3,0.5,0.7,1.0,1.5,2.0]
best_tau, best_score = None, -1e9
for tau in tau_grid:
    pi = softmax_policy(q_oof_ensemble, tau)
    sc = compute_dr_scalar(train, pi, q_oof_ensemble)
    if sc > best_score:
        best_score, best_tau = sc, tau

def cem_tau(df, q_oof, init_mean, pop=CEM_POP, iters=CEM_ITERS):
    pop_arr = np.clip(np.random.normal(init_mean, 0.5, size=pop), 0.01, 10.0)
    best_t, best_s = init_mean, -1e9
    for it in range(iters):
        scores = np.array([ compute_dr_scalar(df, softmax_policy(q_oof, t), q_oof) for t in pop_arr ])
        elite = pop_arr[np.argsort(scores)[-max(1,int(0.2*pop)) : ]]
        mu_e, sigma_e = float(elite.mean()), float(elite.std())
        if elite.std()>1e-6:
            pop_arr = np.clip(np.random.normal(mu_e, sigma_e, size=pop), 0.01, 10.0)
        if scores.max() > best_s:
            best_s = float(scores.max())
            best_t = float(pop_arr[np.argmax(scores)])
    return best_t, best_s

tau_cem, _ = cem_tau(train, q_oof_ensemble, best_tau)
tau_final = tau_cem if tau_cem is not None else best_tau

pi_train_net = np.ones((n,K))/K
pi_test_net = np.ones((len(test),K))/K

X_train_tensor = torch.tensor(train[features_final].values.astype(np.float32), device=DEVICE)
q_oof_tensor = torch.tensor(q_oof_ensemble.astype(np.float32), device=DEVICE)
ai_map = {"mens_email":0,"womens_email":1,"no_email":2}
ai_idx = train["action"].map(ai_map).values
ai_idx_tensor = torch.tensor(ai_idx, dtype=torch.long, device=DEVICE)
r_tensor = torch.tensor(train["visit"].astype(np.float32).values, device=DEVICE)

net = nn.Sequential(nn.Linear(len(features_final),128), nn.ReLU(), nn.Linear(128,64), nn.ReLU(), nn.Linear(64,K)).to(DEVICE)
opt = optim.Adam(net.parameters(), lr=1e-3)
best_dr = -1e9
cur_pat = 0
patience = 6

for epoch in range(POLICY_EPOCHS):
    net.train()
    perm = np.random.permutation(n)
    for i in range(0, n, POLICY_BATCH):
        idx = perm[i:i+POLICY_BATCH]
        xb = X_train_tensor[idx]
        qb = q_oof_tensor[idx]
        ai_b = ai_idx_tensor[idx]
        r_b = r_tensor[idx]
        logits = net(xb)
        pi_b = torch.softmax(logits, dim=1)
        q_pi = (pi_b * qb).sum(dim=1)
        q_ai = qb.gather(1, ai_b.view(-1,1)).squeeze(1)
        pi_ai = pi_b.gather(1, ai_b.view(-1,1)).squeeze(1)
        w = torch.clamp(pi_ai / MU, max=W_CLIP)
        dr_vals = q_pi + w * (r_b - q_ai)
        loss = -dr_vals.mean()
        entropy = - (pi_b * torch.log(pi_b + 1e-12)).sum(dim=1).mean()
        loss = loss - ENT_REG * entropy
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        pi_train_net = torch.softmax(net(X_train_tensor), dim=1).cpu().numpy()
        dr_val = compute_dr_scalar(train, pi_train_net, q_oof_ensemble)
        if dr_val > best_dr + 1e-8:
            best_dr = dr_val
            best_state = {k:v.cpu().clone() for k,v in net.state_dict().items()}
            cur_pat=0
        else:
            cur_pat +=1
            if cur_pat >= patience:
                break
net.load_state_dict({k:best_state[k].to(DEVICE) for k in best_state})
with torch.no_grad():
    pi_test_net = torch.softmax(net(torch.tensor(test[features_final].values.astype(np.float32), device=DEVICE)), dim=1).cpu().numpy()
    pi_train_net = torch.softmax(net(X_train_tensor), dim=1).cpu().numpy()

pi_test_q = softmax_policy(q_hat_test, tau_final)
pi_test_final = W_Q * pi_test_q + W_PI * pi_test_net
pi_test_final = np.clip(pi_test_final, 1e-12, 1.0)
pi_test_final = pi_test_final / pi_test_final.sum(axis=1, keepdims=True)

def create_submission(predictions):
    os.makedirs('results', exist_ok=True)
    submission = pd.DataFrame({
        "id": test_ids,
        "p_mens_email": predictions[:,0],
        "p_womens_email": predictions[:,1],
        "p_no_email": predictions[:,2]
    })
    submission.to_csv(SUB_PATH, index=False)
    print(f"Submission файл сохранен: {SUB_PATH}")
    return SUB_PATH

def main():
    print("=" * 50)
    print("Запуск решения соревнования")
    print("=" * 50)
    create_submission(pi_test_final)
    pi_train_q = softmax_policy(q_oof_ensemble, tau_final)
    pi_train_final = W_Q * pi_train_q + W_PI * pi_train_net
    pi_train_final = np.clip(pi_train_final, 1e-12, 1.0)
    pi_train_final = pi_train_final / pi_train_final.sum(axis=1, keepdims=True)
    metrics = compute_snips_ips_dr(train, pi_train_final, q_oof_ensemble, mu=MU, clip=W_CLIP)
    print("Train diagnostics: SNIPS=", metrics["snips"], "IPS=", metrics["ips"], "DR=", metrics["dr"], "sum_w=", metrics["sum_w"])
    static_res, static_val, static_arm = compute_best_static(train, mu=MU)
    print("Best static (legacy-style):", static_res, "best_arm:", static_arm, "value:", static_val)
    print("Final competition-like SCORE =", metrics["snips"] - static_val)
    print("=" * 50)
    print("Выполнение завершено успешно!")
    print("=" * 50)

if __name__ == "__main__":
    main()
