import os
import time
import random
import numpy as np
import pandas as pd
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, average_precision_score

parser = argparse.ArgumentParser()
parser.add_argument('--mode', required=True, choices=['oracle', 'bandit', 'epsilon_greedy'],
                    help="Mode: oracle, bandit, or epsilon_greedy")
parser.add_argument('--epsilon', default="None", help="Epsilon for epsilon_greedy mode")
parser.add_argument('--seed', type=int, default=None, help="Random seed for reproducibility")
args = parser.parse_args()

mode = args.mode
epsilon = None if args.epsilon == "None" else float(args.epsilon)
seed = args.seed 

np.random.seed(seed)
random.seed(seed)
data_path = os.path.join(os.environ.get("TMPDIR", "/rds/general/user/ms2524/home/MSc-project/notebooks"), "oldmerged.parquet")
df = pd.read_parquet(data_path)

top_100 = [
 'avgdpdtolclosure24_3658938P', 'dateofcredstart_739D_days_from_decision', 'dpdmax_139P',
 'pmtnum_254L', 'numberofoverdueinstlmax_1039L', 'price_1097A', 'age_at_decision', 'sex_738L',
 'mobilephncnt_593L', 'overdueamountmax_155A', 'numrejects9m_859L', 'numberofcontrsvalue_358L',
 'pmtssum_45A', 'incometype_1044T', 'amount_4527230A', 'education_927M', 'days120_123L',
 'pmtaverage_3A', 'numinstlswithdpd10_728L', 'cntpmts24_3658933L', 'dpdmaxdateyear_596T',
 'firstclxcampaign_1125D_days_from_decision', 'totaldebtoverduevalue_178A', 'pmtscount_423L',
 'annuity_780A', 'isbidproduct_1095L', 'residualamount_856A', 'pctinstlsallpaidlate1d_3546856L',
 'credamount_770A', 'eir_270L', 'totalamount_996A', 'numinstpaidearly3d_3546850L', 'dpdmax_757P',
 'days90_310L', 'days30_165L', 'dateofbirth_337D_days_from_decision', 'days180_256L',
 'numinsttopaygr_769L', 'maxdbddpdtollast12m_3658940P', 'pctinstlsallpaidearl3d_427L',
 'validfrom_1069D_days_from_decision', 'currdebt_22A', 'interestrate_311L', 'overdueamountmax2_14A',
 'familystate_447L', 'birthdate_574D_days_from_decision', 'empl_employedfrom_271D_days_from_decision',
 'applicationscnt_867L', 'totaldebt_9A', 'disbursedcredamount_1113A', 'maxdpdlast24m_143P',
 'totaloutstanddebtvalue_39A', 'dateofcredstart_181D_days_from_decision', 'lastrejectcredamount_222A',
 'maxdpdlast3m_392P', 'maxdpdlast9m_1059P', 'avgdbddpdlast24m_3658932P', 'language1_981M',
 'credtype_322L', 'totalamount_6A', 'inittransactionamount_650A', 'approvaldate_319D_days_from_decision',
 'lastdelinqdate_224D_days_from_decision', 'pmtnum_8L', 'days360_512L', 'numberofoverdueinstlmax_1151L',
 'lastrejectreasonclient_4145040M', 'numinstlallpaidearly3d_817L',
 'lastrejectdate_50D_days_from_decision', 'maxdpdlast12m_727P', 'homephncnt_628L',
 'education_1138M', 'education_1103M', 'debtoutstand_525A', 'totalsettled_863A',
 'numinstunpaidmax_3546851L', 'lastrejectreason_759M', 'pmtaverage_4527227A', 'overdueamountmax_35A',
 'monthsannuity_845L', 'maxannuity_159A', 'maxdbddpdlast1m_3658939P', 'maxdpdlast6m_474P',
 'responsedate_4527233D_days_from_decision', 'instlamount_768A', 'dtlastpmt_581D_days_from_decision',
 'numinstlsallpaid_934L', 'tenor_203L', 'maxdpdtolerance_577P', 'numinstunpaidmaxest_4493212L',
 'numberofoverdueinstlmaxdat_641D_days_from_decision', 'familystate_726L', 'maxdpdtolerance_374P',
 'numinstlswithoutdpd_562L', 'disbursementtype_67L', 'lastst_736L', 'numinstlswithdpd5_4187116L',
 'cntincpaycont9m_3716944L', 'requesttype_4525192L', 'avgdbddpdlast3m_4187120P'
]

for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype('category')
df = df[top_100 + ["WEEK_NUM", "target", "case_id"]]


def add_flag(df, threshold=0.05):
    df = df.copy()
    missing = df.isna().mean()
    cols = missing[missing > threshold].index.tolist()
    for col in cols:
        df[col + '_missing'] = df[col].isna().astype(int)
    return df


def encode_features(df, categorical_features):
    df = df.copy()
    return pd.get_dummies(df, columns=categorical_features, drop_first=True)

df = encode_features(df, df.select_dtypes(include=['object', 'category']).columns.tolist())
df = add_flag(df)
features_with_missing = [col for col in df.columns if df[col].isna().any()]


def impute_features(train_df, test_df, features, sentinel=-99999):

    train_df = train_df.copy()
    test_df = test_df.copy()

    for feature in features:
        is_int_feature = (
            pd.api.types.is_integer_dtype(train_df[feature].dtype) or
            pd.api.types.is_integer_dtype(test_df[feature].dtype)
        )

        valid_train_vals = train_df[feature][(train_df[feature].notna())
        ]
        median_val = valid_train_vals.median() if not valid_train_vals.empty else None

        if median_val is not None and is_int_feature:
            median_val = int(round(median_val))

        if is_int_feature:
            train_df[feature] = train_df[feature].astype("Int32")
            test_df[feature] = test_df[feature].astype("Int32")
        else:
            train_df[feature] = train_df[feature].where(train_df[feature].notna(), np.nan).astype(float)
            test_df[feature] = test_df[feature].where(test_df[feature].notna(), np.nan).astype(float)

        if median_val is not None:
            train_df[feature] = train_df[feature].fillna(median_val)
            test_df[feature] = test_df[feature].fillna(median_val)
        else:
            train_df[feature] = train_df[feature].fillna(sentinel)
            test_df[feature] = test_df[feature].fillna(sentinel)

    return train_df, test_df

def make_utility(fn_weight=-1, tn_weight=0.06):
    return lambda tn, fn: tn_weight * tn + fn_weight * fn

def find_best_threshold_vectorized(y_true, y_proba, thresholds, utility_fn):
    thresholds = np.array(thresholds)
    utilities = []
    for thr in thresholds:
        pred = (y_proba >= thr).astype(int)
        tn = np.sum((pred == 0) & (y_true == 0))
        fn = np.sum((pred == 0) & (y_true == 1))
        utilities.append(utility_fn(tn, fn))
    best_idx = np.argmax(utilities)
    return thresholds[best_idx], utilities[best_idx]

def split_by_week(df, n):
    min_week = df["WEEK_NUM"].min()
    max_week = df["WEEK_NUM"].max()
    bins = np.linspace(min_week, max_week + 1, n + 1, dtype=int)
    segments = {}
    for i in range(n):
        lower = bins[i]
        upper = bins[i + 1]
        key = f"split_{i}"
        segment = df[(df["WEEK_NUM"] >= lower) & (df["WEEK_NUM"] < upper)].copy()
        segments[key] = segment.sort_values(by=["WEEK_NUM"])
    return segments
    
def cf(final_segments, thresholds=np.linspace(0.01, 0.5, 40),
       random_state=seed, mode='oracle', epsilon=0.01, x=0.06):
    results = []
    exclude_cols = {'case_id', 'WEEK_NUM', 'target'}
    features = [c for c in final_segments['split_0'].columns if c not in exclude_cols]

    segment_keys = sorted(final_segments.keys(), key=lambda x: int(x.split('_')[-1]))
    X_splits = [final_segments[k][features] for k in segment_keys]
    y_splits = [final_segments[k]['target'].to_numpy() for k in segment_keys]

    accepted_X = [X_splits[0]]
    accepted_y = [y_splits[0]]
    utility_fn = make_utility(fn_weight=-1, tn_weight=x)

    for t in range(1, len(segment_keys)):
        test_key = segment_keys[t]

        if mode == 'oracle':
            X_train = pd.concat(X_splits[:t], ignore_index=True)
            y_train = np.concatenate(y_splits[:t])
        else:
            X_train = pd.concat(accepted_X, ignore_index=True)
            y_train = np.concatenate(accepted_y)

        X_test = X_splits[t]
        y_test = y_splits[t]

        reference_data = pd.concat(X_splits[:t], ignore_index=True)

        ref_filled, _ = impute_features(reference_data, reference_data, features_with_missing)
        _, train_filled = impute_features(reference_data, X_train, features_with_missing)
        _, test_filled  = impute_features(reference_data, X_test,  features_with_missing)

        scaler = StandardScaler().fit(ref_filled.values)
        train_scaled = scaler.transform(train_filled.values)
        test_scaled  = scaler.transform(test_filled.values)

        model = LogisticRegression(max_iter=1000, random_state=random_state, penalty=None)
        model.fit(train_scaled, y_train)

        train_scores = model.predict_proba(train_scaled)[:, 1]
        best_threshold, _ = find_best_threshold(y_train, train_scores, thresholds, utility_fn)

        test_scores = model.predict_proba(test_scaled)[:, 1]

        if mode == 'epsilon_greedy' and epsilon is not None:
            rand = np.random.rand(len(test_scores))
            follow_policy = rand >= epsilon
            random_accept = np.random.rand(len(test_scores)) > 0.5
            policy_accept = test_scores < best_threshold
            accept_decisions = np.where(follow_policy, policy_accept, random_accept)
        else:
            accept_decisions = test_scores < best_threshold

        accepted = np.asarray(accept_decisions, dtype=bool)
        tn = np.sum(accepted & (y_test == 0))
        fn = np.sum(accepted & (y_test == 1))
        utility = utility_fn(tn, fn)

        results.append({
            "segment": test_key,
            "segment_num": int(test_key.split("_")[1]),
            "threshold": float(best_threshold),
            "utility": float(utility),
            "true_default_rate": float(y_test.mean()),
            "pred_default_rate": float(test_scores.mean()),
            "pred_default_se": float(test_scores.std(ddof=1) / np.sqrt(len(test_scores))),
            "accept_rate": float(accepted.mean()),
            "exploration_rate": float((rand < epsilon).mean()) if mode == 'epsilon_greedy' else 0.0
        })

        if mode != 'oracle':
            accepted_X.append(X_test.loc[accepted])
            accepted_y.append(y_test[accepted])

    return pd.DataFrame(results)


final_segments = split_by_week(df, 30)

results = cf(final_segments, mode=mode, epsilon=epsilon)

results['seed'] = seed
results['epsilon'] = epsilon if epsilon is not None else "None"
results['mode'] = mode
output_name = f"cf_{mode}"
if epsilon is not None:
    output_name += f"_eps{epsilon}"
output_name += f"_seed{seed}.csv" 

results.to_csv(output_name, index=False) 


